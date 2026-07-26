# -*- coding: utf-8 -*-
"""Server-side language detection + translation (v13).

Every comment — written on the site or fetched from ANY external source
(YouTube / Reddit / X / anything added later) — goes through this module:

    detect_and_translate(text, target="ar")
        -> {"translated": "...", "detected_lang": "en", "ok": True}

* Works for ALL languages (English, Bengali, Urdu, Turkish, Indonesian, ...)
  because it uses Google's public translate endpoint, which auto-detects the
  source language and returns it alongside the translation.
* The ORIGINAL text is never modified — callers store the translation in a
  separate column (comments.text_ar) next to the original.
* Every failure is soft: on any network/parse error the function returns
  ok=False and the app simply keeps showing the original text (and the
  browser-side translation remains as a further fallback), so translation
  problems can never break comment display or ingestion.

An in-process cache avoids re-translating identical texts (e.g. when the
admin re-analyzes all comments).
"""
import json

import requests

TRANSLATE_URL = "https://translate.googleapis.com/translate_a/single"
TIMEOUT = 12
_cache = {}  # (text, target) -> result dict


def _has_arabic(s: str) -> bool:
    return any("\u0600" <= ch <= "\u06FF" for ch in s or "")


def detect_and_translate(text: str, target: str = "ar") -> dict:
    """Detect the language of `text` and translate it to `target`.

    Returns {"translated": str|None, "detected_lang": str|None, "ok": bool}.
    If the text is already in the target language the translation equals the
    original and detected_lang is the target code."""
    text = (text or "").strip()
    if not text:
        return {"translated": None, "detected_lang": None, "ok": False}
    key = (text, target)
    if key in _cache:
        return _cache[key]
    # Cheap local shortcut: clearly-Arabic text asked to become Arabic.
    if target == "ar" and _has_arabic(text) and not any(c.isascii() and c.isalpha() for c in text):
        result = {"translated": text, "detected_lang": "ar", "ok": True}
        _cache[key] = result
        return result
    try:
        r = requests.get(
            TRANSLATE_URL,
            params={"client": "gtx", "sl": "auto", "tl": target, "dt": "t", "q": text},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        translated = "".join(seg[0] for seg in (data[0] or []) if seg and seg[0])
        detected = data[2] if len(data) > 2 and isinstance(data[2], str) else None
        ok = bool(translated)
        result = {"translated": translated or None, "detected_lang": detected, "ok": ok}
    except (requests.RequestException, ValueError, KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"[translate] failed ({target}): {e}")
        result = {"translated": None, "detected_lang": None, "ok": False}
    if result["ok"]:
        _cache[key] = result
        if len(_cache) > 5000:  # keep the cache bounded
            _cache.clear()
    return result


def to_english(text: str) -> str:
    """Translate to English for the sentiment engine; falls back to the
    original text if translation is unavailable (never raises)."""
    res = detect_and_translate(text, target="en")
    return res["translated"] if res["ok"] else text
