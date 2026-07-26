# -*- coding: utf-8 -*-
"""Google Maps Reviews — data source (v15), prepared per the product spec.

The spec asks to PREPARE the system to receive and analyze Google Maps
reviews now; the API key/place list get connected later by the site owner.
This module supports BOTH paths so nothing has to change later:

  1. AUTOMATIC (once configured): set GOOGLE_MAPS_API_KEY (a Places API key)
     and GOOGLE_MAPS_PLACE_IDS (comma-separated Google Place IDs — hotels,
     transport companies, Hajj/Umrah campaigns, etc.) and fetch_reviews()
     pulls each place's public reviews via the official Places Details API
     (https://maps.googleapis.com/maps/api/place/details/json), no scraping.
     Google's Places API only exposes a handful of "most relevant" reviews
     per place — this is a real, documented limitation of the official API,
     not a bug here.

  2. MANUAL IMPORT (works today, no key needed): admins can POST a JSON list
     of reviews (e.g. exported from Google Takeout, a spreadsheet, or any
     tool) to POST /api/admin/import-reviews — normalize_review() maps each
     item's fields (however named) onto the platform's internal schema.

Every field the spec asked the system to be ready to receive:
    place_name, place_type, country, city, place_url, rating,
    review_date, language, text, username, review_id
"""
import os
import re
from datetime import datetime, timezone

import requests

TIMEOUT = 15
PLACES_DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"

# Flexible key aliases so a manually-imported JSON list doesn't have to match
# our internal names exactly (spreadsheets/exports vary a lot in practice).
_ALIASES = {
    "place_name": ("place_name", "placeName", "name", "location_name"),
    "place_type": ("place_type", "placeType", "type", "category"),
    "country": ("country", "place_country"),
    "city": ("city", "place_city"),
    "place_url": ("place_url", "placeUrl", "url", "link", "maps_url"),
    "rating": ("rating", "stars", "star_rating"),
    "review_date": ("review_date", "reviewDate", "date", "time", "created_at"),
    "language": ("language", "lang"),
    "text": ("text", "review_text", "reviewText", "comment", "body"),
    "username": ("username", "author", "author_name", "authorName", "user"),
    "review_id": ("review_id", "reviewId", "id", "external_id"),
}


def configured() -> bool:
    return bool(os.environ.get("GOOGLE_MAPS_API_KEY") and os.environ.get("GOOGLE_MAPS_PLACE_IDS"))


def _pick(raw: dict, key: str):
    for alias in _ALIASES[key]:
        if alias in raw and raw[alias] not in (None, ""):
            return raw[alias]
    return None


def _parse_date(value) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    if isinstance(value, (int, float)):  # unix timestamp (seconds)
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
        except (ValueError, OSError):
            return datetime.now(timezone.utc).isoformat()
    s = str(value).strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.isoformat()
        except ValueError:
            continue
    return datetime.now(timezone.utc).isoformat()


def normalize_review(raw: dict, default_place: dict = None) -> dict:
    """Map one loosely-structured review dict onto the internal schema used
    for storage. default_place (optional) fills place_name/type/country/
    city/url when a batch import shares one place (e.g. all reviews for a
    single hotel) and the per-row data omits them."""
    default_place = default_place or {}
    text = (_pick(raw, "text") or "").strip()
    rating = _pick(raw, "rating")
    try:
        rating = int(round(float(rating))) if rating is not None else None
        if rating is not None and not 1 <= rating <= 5:
            rating = None
    except (TypeError, ValueError):
        rating = None
    review_id = _pick(raw, "review_id") or None
    return {
        "text": text,
        "author": _pick(raw, "username") or "Google Maps user",
        "created_at": _parse_date(_pick(raw, "review_date")),
        "source": "google_maps",
        "external_id": "gmaps:" + str(review_id) if review_id else None,
        "rating": rating,
        "language": _pick(raw, "language"),
        "place_name": _pick(raw, "place_name") or default_place.get("place_name"),
        "place_type": _pick(raw, "place_type") or default_place.get("place_type"),
        "place_country": _pick(raw, "country") or default_place.get("country"),
        "place_city": _pick(raw, "city") or default_place.get("city"),
        "place_url": _pick(raw, "place_url") or default_place.get("place_url"),
    }


def normalize_batch(raw_list: list, default_place: dict = None) -> list:
    out = []
    for raw in raw_list or []:
        if not isinstance(raw, dict):
            continue
        item = normalize_review(raw, default_place=default_place)
        if item["text"]:
            out.append(item)
    return out


def _extract_city_country(address_components: list) -> tuple:
    city = country = None
    for comp in address_components or []:
        types = comp.get("types") or []
        if "locality" in types and not city:
            city = comp.get("long_name")
        if "country" in types and not country:
            country = comp.get("long_name")
    return city, country


def fetch_reviews() -> list:
    """AUTOMATIC path: pull reviews for every configured Place ID via the
    official Places Details API. Returns [] gracefully if not configured or
    on any network/auth error — a source that isn't set up must never break
    the app. NEVER raises."""
    key = os.environ.get("GOOGLE_MAPS_API_KEY")
    place_ids = [p.strip() for p in (os.environ.get("GOOGLE_MAPS_PLACE_IDS") or "").split(",") if p.strip()]
    if not (key and place_ids):
        return []
    out = []
    for place_id in place_ids:
        try:
            r = requests.get(
                PLACES_DETAILS_URL,
                params={"place_id": place_id, "key": key,
                        "fields": "name,url,reviews,address_component"},
                timeout=TIMEOUT,
            )
            r.raise_for_status()
            result = (r.json() or {}).get("result") or {}
            place_name = result.get("name")
            place_url = result.get("url")
            city, country = _extract_city_country(result.get("address_component"))
            for rv in result.get("reviews") or []:
                text = (rv.get("text") or "").strip()
                if not text:
                    continue
                rid = f"{place_id}:{rv.get('time') or rv.get('author_name')}"
                out.append({
                    "text": text,
                    "author": rv.get("author_name") or "Google Maps user",
                    "created_at": _parse_date(rv.get("time")),
                    "source": "google_maps",
                    "external_id": "gmaps:" + re.sub(r"\s+", "_", rid),
                    "rating": rv.get("rating"),
                    "language": rv.get("language"),
                    "place_name": place_name,
                    "place_type": None,  # Places API doesn't classify type; set via import if needed
                    "place_country": country,
                    "place_city": city,
                    "place_url": place_url,
                })
        except Exception as e:  # one bad place must not stop the rest
            print(f"[google_maps] fetch failed for place_id={place_id}: {e}")
    return out
