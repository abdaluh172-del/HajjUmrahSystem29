# -*- coding: utf-8 -*-
"""Fetch public comments/opinions about Hajj & Umrah — OFFICIAL APIs ONLY.

v12, extended in v15 with richer Reddit metadata (title, community, votes,
comment count, permalink — stored in the comments.source_meta JSON column
so they can be shown in the UI and used in analytics' source comparison).
Google Maps Reviews are a separate source — see google_maps_source.py.

No scraping anywhere in this module: every request goes to the
platform's documented, official API using credentials the site owner
provides via environment variables. If a credential is missing, that
source is simply skipped (the app keeps working normally without it).

Environment variables (all optional):
    YOUTUBE_API_KEY        Google API key with YouTube Data API v3 enabled
    YOUTUBE_CHANNEL_IDS    comma-separated channel IDs to follow — their
                           latest uploads are checked automatically every
                           run, IN ADDITION to the keyword search below
    YOUTUBE_MAX_VIDEOS_PER_CHANNEL  latest videos to check per followed
                           channel each run (default 5)
    YOUTUBE_MAX_PAGES_PER_VIDEO     comment pages (100/page) fetched per
                           video per run — raise for a full backlog, lower
                           to conserve API quota (default 5)
    REDDIT_CLIENT_ID       Reddit "script" app credentials
    REDDIT_CLIENT_SECRET   (OAuth2 client-credentials flow)
    REDDIT_USER_AGENT      e.g. "HajjUmrahSystem/1.0 by <reddit-username>"
    X_BEARER_TOKEN         X (Twitter) API v2 Bearer token (recent search)
    EXTERNAL_SEARCH_QUERY  override the default search query
    FETCH_INTERVAL_MINUTES auto-refresh period (default 60 = hourly)
    AUTO_FETCH_EXTERNAL    set to "0" to disable the hourly auto-refresh

Compliance notes:
  * Only public content returned by the official endpoints is stored.
  * Each stored comment keeps its source ('youtube'/'reddit'/'x') and the
    platform's own item id (external_id) so it is never duplicated and its
    origin is always visible in the UI.
  * Request volumes are tiny (a few calls per run) — far below every
    platform's rate limits.

Each fetcher returns a list of dicts:
    {"external_id", "text", "author", "created_at", "source"}
and NEVER raises — network/auth problems are logged and yield [].
"""
import os
from datetime import datetime, timezone

import requests

DEFAULT_QUERY = "Hajj Umrah experience"
TIMEOUT = 15  # seconds per HTTP request


def _query() -> str:
    return os.environ.get("EXTERNAL_SEARCH_QUERY") or DEFAULT_QUERY


def configured_sources() -> dict:
    """Which sources have credentials set (used by the admin status panel)."""
    return {
        "youtube": bool(os.environ.get("YOUTUBE_API_KEY")),
        "reddit": bool(os.environ.get("REDDIT_CLIENT_ID") and os.environ.get("REDDIT_CLIENT_SECRET")),
        "x": bool(os.environ.get("X_BEARER_TOKEN")),
    }


# ------------------------------------------------------------------ #
# YouTube Data API v3 (official): search videos, then read their
# top-level comment threads.
# ------------------------------------------------------------------ #
# ------------------------------------------------------------------ #
# YouTube Data API v3 (official): discover videos (by keyword search
# and/or by following specific channels), then paginate their top-level
# comment threads.
# ------------------------------------------------------------------ #
def youtube_channel_ids() -> list:
    raw = os.environ.get("YOUTUBE_CHANNEL_IDS", "")
    return [c.strip() for c in raw.split(",") if c.strip()]


def _youtube_latest_videos_from_channel(channel_id: str, key: str, max_videos: int) -> list:
    """Latest uploads from one followed channel. Returns [{"video_id","title"}]."""
    try:
        r = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={"part": "id,snippet", "channelId": channel_id, "type": "video",
                    "order": "date", "maxResults": max_videos, "key": key},
            timeout=TIMEOUT)
        r.raise_for_status()
        out = []
        for it in r.json().get("items", []):
            vid = it.get("id", {}).get("videoId")
            if vid:
                out.append({"video_id": vid, "title": (it.get("snippet") or {}).get("title"),
                           "channel_id": channel_id})
        return out
    except Exception as e:  # one bad/unavailable channel must not stop the rest
        print(f"[external] youtube channel {channel_id} lookup failed: {e}")
        return []


def _youtube_video_comments(video_id: str, key: str, max_pages: int,
                             known_ids: set, per_page: int = 100) -> list:
    """Paginate a video's top-level comment threads, newest first
    (order=time), stopping as soon as a comment already in `known_ids` is
    seen (everything after it, in time order, was already collected on a
    previous run) or after `max_pages` pages (safety cap on API quota use
    per run — a video with a huge backlog is picked up incrementally over
    several refresh cycles rather than exhausting the day's quota at once).
    """
    out = []
    page_token = None
    for _ in range(max(1, max_pages)):
        try:
            params = {"part": "snippet", "videoId": video_id, "textFormat": "plainText",
                      "order": "time", "maxResults": per_page, "key": key}
            if page_token:
                params["pageToken"] = page_token
            cr = requests.get("https://www.googleapis.com/youtube/v3/commentThreads",
                              params=params, timeout=TIMEOUT)
            cr.raise_for_status()
            data = cr.json()
            hit_known = False
            for it in data.get("items", []):
                sn = it["snippet"]["topLevelComment"]["snippet"]
                cid = "youtube:" + it["snippet"]["topLevelComment"]["id"]
                if cid in known_ids:
                    hit_known = True
                    break  # everything from here on (older) is already stored
                out.append({
                    "external_id": cid,
                    "text": (sn.get("textDisplay") or "").strip(),
                    "author": sn.get("authorDisplayName") or "YouTube user",
                    "created_at": sn.get("publishedAt") or datetime.now(timezone.utc).isoformat(),
                    "source": "youtube",
                    "video_id": video_id,
                })
            if hit_known:
                break
            page_token = data.get("nextPageToken")
            if not page_token:
                break  # no more pages — collected everything available
        except Exception as e:  # one bad page must not stop the rest of the run
            print(f"[external] youtube comments for {video_id} failed: {e}")
            break
    return out


def fetch_youtube(max_videos: int = 3, max_comments_per_video: int = 15,
                  known_ids: set = None) -> list:
    """Fetch comments from (a) a keyword search for new/relevant videos and
    (b) every channel configured in YOUTUBE_CHANNEL_IDS (their latest
    uploads, fetched automatically every run). `known_ids` — external_ids
    already stored in the DB — lets pagination stop early instead of
    re-walking comments already collected on a previous run; pass None to
    always fetch from the first page only (legacy/simple behavior).

    Env vars (all optional, sensible defaults):
      YOUTUBE_CHANNEL_IDS            comma-separated channel IDs to follow
      YOUTUBE_MAX_VIDEOS_PER_CHANNEL latest videos to check per channel (default 5)
      YOUTUBE_MAX_PAGES_PER_VIDEO    comment pages (100/page) per video per run (default 5)
    """
    key = os.environ.get("YOUTUBE_API_KEY")
    if not key:
        return []
    known_ids = known_ids or set()
    max_pages = max(1, int(os.environ.get("YOUTUBE_MAX_PAGES_PER_VIDEO", "5")))
    max_per_channel = max(1, int(os.environ.get("YOUTUBE_MAX_VIDEOS_PER_CHANNEL", "5")))

    # -- discover videos: keyword search + every followed channel's latest --
    videos = []  # [{"video_id","title","channel_id"(optional)}]
    try:
        r = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={"part": "id,snippet", "q": _query(), "type": "video",
                    "maxResults": max_videos, "key": key},
            timeout=TIMEOUT)
        r.raise_for_status()
        for it in r.json().get("items", []):
            vid = it.get("id", {}).get("videoId")
            if vid:
                videos.append({"video_id": vid, "title": (it.get("snippet") or {}).get("title"),
                              "channel_id": None})
    except Exception as e:
        print(f"[external] youtube search failed: {e}")

    for channel_id in youtube_channel_ids():
        videos.extend(_youtube_latest_videos_from_channel(channel_id, key, max_per_channel))

    # de-duplicate videos discovered by both search and channel-following
    seen_vids = set()
    unique_videos = []
    for v in videos:
        if v["video_id"] not in seen_vids:
            seen_vids.add(v["video_id"])
            unique_videos.append(v)

    out = []
    for v in unique_videos:
        try:
            comments = _youtube_video_comments(
                v["video_id"], key, max_pages, known_ids,
                per_page=min(100, max(max_comments_per_video, 1)))
            for c in comments:
                c["video_title"] = v.get("title")
                c["channel_id"] = v.get("channel_id")
            out.append({"video_id": v["video_id"], "title": v.get("title"),
                       "channel_id": v.get("channel_id"), "comments": comments})
        except Exception as e:  # one bad video must not stop the rest
            print(f"[external] youtube video {v['video_id']} failed: {e}")
    return out


# ------------------------------------------------------------------ #
# v15.5 — analyze ONE specific video by URL (on-demand, like the Google
# Maps manual import): pull ~150 real top-level comments from that exact
# video, ordered by relevance, and hand them back for analysis. No comments
# are ever generated — only what real users actually posted is returned.
# ------------------------------------------------------------------ #
import re as _re

_YT_ID_PATTERNS = [
    _re.compile(r"(?:v=|/v/|youtu\.be/|/embed/|/shorts/|/live/)([A-Za-z0-9_-]{11})"),
]


def extract_video_id(url_or_id: str) -> str:
    """Accept a full YouTube URL (any common shape) or a bare 11-char id and
    return the video id. Returns '' if none can be found."""
    s = (url_or_id or "").strip()
    if not s:
        return ""
    if _re.fullmatch(r"[A-Za-z0-9_-]{11}", s):
        return s
    for pat in _YT_ID_PATTERNS:
        m = pat.search(s)
        if m:
            return m.group(1)
    return ""


class YouTubeFetchError(Exception):
    """Raised by fetch_video_comments so the caller can show a precise reason
    (comments disabled, video not found, quota exceeded, key missing)."""
    def __init__(self, message_en, message_ar, status=400):
        super().__init__(message_en)
        self.message_en = message_en
        self.message_ar = message_ar
        self.status = status


def fetch_video_comments(url_or_id: str, max_comments: int = 150) -> dict:
    """Fetch up to `max_comments` REAL top-level comments from ONE video.

    Returns {"video_id", "comments": [ {external_id, text, author,
    created_at, likes, source}, ... ]}. If the video has fewer comments than
    requested, returns all available. Raises YouTubeFetchError with a clear
    bilingual reason on failure (never a silent empty result)."""
    key = os.environ.get("YOUTUBE_API_KEY")
    if not key:
        raise YouTubeFetchError(
            "YouTube isn't configured on the server (missing YOUTUBE_API_KEY)",
            "خدمة YouTube غير مُفعّلة على الخادم (مفتاح YOUTUBE_API_KEY مفقود)", 501)
    video_id = extract_video_id(url_or_id)
    if not video_id:
        raise YouTubeFetchError("Could not read a video id from that link",
                                "تعذّر التعرّف على معرّف الفيديو من الرابط", 400)

    comments = []
    seen = set()
    page_token = None
    max_comments = max(1, min(int(max_comments or 150), 500))
    while len(comments) < max_comments:
        params = {"part": "snippet", "videoId": video_id, "textFormat": "plainText",
                  "order": "relevance", "maxResults": min(100, max_comments - len(comments)),
                  "key": key}
        if page_token:
            params["pageToken"] = page_token
        try:
            r = requests.get("https://www.googleapis.com/youtube/v3/commentThreads",
                             params=params, timeout=TIMEOUT)
        except Exception as e:
            print(f"[external] fetch_video_comments network error: {e}")
            raise YouTubeFetchError("Network error contacting YouTube",
                                    "خطأ في الاتصال بخدمة YouTube", 502)
        if r.status_code != 200:
            body = r.text[:400]
            print(f"[external] fetch_video_comments HTTP {r.status_code}: {body}")
            if r.status_code == 403 and "commentsDisabled" in body:
                raise YouTubeFetchError("Comments are disabled on this video",
                                        "التعليقات معطّلة على هذا الفيديو", 400)
            if r.status_code == 403 and ("quota" in body.lower()):
                raise YouTubeFetchError("YouTube API daily quota exceeded — try again later",
                                        "تم تجاوز حصة YouTube اليومية — حاول لاحقًا", 429)
            if r.status_code == 404:
                raise YouTubeFetchError("Video not found", "الفيديو غير موجود", 404)
            raise YouTubeFetchError(f"YouTube API error (HTTP {r.status_code})",
                                    f"خطأ من YouTube (HTTP {r.status_code})", 502)
        data = r.json()
        for it in data.get("items", []):
            sn = it["snippet"]["topLevelComment"]["snippet"]
            cid = "youtube:" + it["snippet"]["topLevelComment"]["id"]
            if cid in seen:
                continue
            seen.add(cid)
            text = (sn.get("textDisplay") or "").strip()
            if not text:
                continue
            comments.append({
                "external_id": cid,
                "text": text,
                "author": sn.get("authorDisplayName") or "YouTube user",
                "created_at": sn.get("publishedAt") or datetime.now(timezone.utc).isoformat(),
                "likes": int(sn.get("likeCount", 0) or 0),
                "source": "youtube",
                "video_id": video_id,
            })
            if len(comments) >= max_comments:
                break
        page_token = data.get("nextPageToken")
        if not page_token:
            break  # fewer comments than requested — return all available
    return {"video_id": video_id, "comments": comments}


def flatten_youtube_comments(video_results: list) -> list:
    """video_results is fetch_youtube()'s per-video shape; flatten to the
    plain comment-list shape every other fetcher (fetch_reddit/fetch_x)
    returns, for callers that just want the comments."""
    flat = []
    for v in video_results or []:
        flat.extend([c for c in v.get("comments", []) if c.get("text")])
    return flat


# ------------------------------------------------------------------ #
# Reddit API (official OAuth2 client-credentials flow).
# ------------------------------------------------------------------ #
def fetch_reddit(max_posts: int = 20) -> list:
    cid = os.environ.get("REDDIT_CLIENT_ID")
    secret = os.environ.get("REDDIT_CLIENT_SECRET")
    agent = os.environ.get("REDDIT_USER_AGENT") or "HajjUmrahSystem/1.0"
    if not (cid and secret):
        return []
    out = []
    try:
        tok = requests.post(
            "https://www.reddit.com/api/v1/access_token",
            auth=(cid, secret), data={"grant_type": "client_credentials"},
            headers={"User-Agent": agent}, timeout=TIMEOUT)
        tok.raise_for_status()
        access = tok.json()["access_token"]
        r = requests.get(
            "https://oauth.reddit.com/search",
            params={"q": _query(), "limit": max_posts, "sort": "new", "type": "link"},
            headers={"Authorization": f"Bearer {access}", "User-Agent": agent},
            timeout=TIMEOUT)
        r.raise_for_status()
        for child in r.json().get("data", {}).get("children", []):
            d = child.get("data", {})
            text = (d.get("selftext") or d.get("title") or "").strip()
            if not text:
                continue
            created = d.get("created_utc")
            permalink = ("https://reddit.com" + d["permalink"]) if d.get("permalink") else None
            out.append({
                "external_id": "reddit:" + str(d.get("id")),
                "text": text[:2000],
                "author": "u/" + (d.get("author") or "reddit"),
                "created_at": (datetime.fromtimestamp(created, tz=timezone.utc).isoformat()
                               if created else datetime.now(timezone.utc).isoformat()),
                "source": "reddit",
                # v15: extra fields for display + analytics (stored in source_meta)
                "title": d.get("title"),
                "community": d.get("subreddit"),
                "votes": d.get("ups") if d.get("ups") is not None else d.get("score"),
                "num_comments": d.get("num_comments"),
                "permalink": permalink,
            })
    except Exception as e:
        print(f"[external] reddit fetch failed: {e}")
    return out


# ------------------------------------------------------------------ #
# X (Twitter) API v2 (official): recent search. Requires a plan whose
# permissions include recent search — if the token lacks access the
# request simply fails and this source yields nothing.
# ------------------------------------------------------------------ #
def fetch_x(max_results: int = 25) -> list:
    bearer = os.environ.get("X_BEARER_TOKEN")
    if not bearer:
        return []
    out = []
    try:
        r = requests.get(
            "https://api.twitter.com/2/tweets/search/recent",
            params={"query": f"({_query()}) -is:retweet",
                    "max_results": max(10, min(max_results, 100)),
                    "tweet.fields": "created_at,author_id"},
            headers={"Authorization": f"Bearer {bearer}"}, timeout=TIMEOUT)
        r.raise_for_status()
        for tw in r.json().get("data", []) or []:
            out.append({
                "external_id": "x:" + str(tw["id"]),
                "text": (tw.get("text") or "").strip(),
                "author": "X user " + str(tw.get("author_id") or ""),
                "created_at": tw.get("created_at") or datetime.now(timezone.utc).isoformat(),
                "source": "x",
            })
    except Exception as e:
        print(f"[external] x fetch failed: {e}")
    return [c for c in out if c["text"]]


def fetch_all(known_youtube_ids: set = None) -> list:
    """Fetch from every configured source (flat comment list, same shape
    for all sources). Unconfigured sources yield []. `known_youtube_ids`
    lets the YouTube fetcher stop paginating early — see fetch_youtube()."""
    yt = flatten_youtube_comments(fetch_youtube(known_ids=known_youtube_ids))
    return yt + fetch_reddit() + fetch_x()


# ================================================================== #
# v15.6 — On-demand keyword search for the admin "X Analytics" and
# "Reddit Analytics" pages (mirrors fetch_video_comments' contract:
# return a structured result, or raise a bilingual *FetchError with a
# clear reason — never a silent empty result). These power the two new
# admin pages that work exactly like the YouTube analysis page, but
# search by the Hajj/Umrah keywords instead of taking a single URL.
# ================================================================== #

# The keywords the two new pages search for (English + Arabic), per spec.
HAJJ_UMRAH_KEYWORDS = ["Hajj", "Umrah", "الحج", "العمرة"]


def _keyword_query_x() -> str:
    """X (Twitter) API v2 recent-search query built from the Hajj/Umrah
    keywords. Overridable via X_SEARCH_QUERY, else EXTERNAL_SEARCH_QUERY."""
    override = os.environ.get("X_SEARCH_QUERY") or os.environ.get("EXTERNAL_SEARCH_QUERY")
    if override:
        return override
    # (Hajj OR Umrah OR الحج OR العمرة) -is:retweet
    terms = " OR ".join(HAJJ_UMRAH_KEYWORDS)
    return f"({terms}) -is:retweet"


def _keyword_query_reddit() -> str:
    """Reddit search query from the Hajj/Umrah keywords. Overridable via
    REDDIT_SEARCH_QUERY, else EXTERNAL_SEARCH_QUERY."""
    override = os.environ.get("REDDIT_SEARCH_QUERY") or os.environ.get("EXTERNAL_SEARCH_QUERY")
    if override:
        return override
    return " OR ".join(HAJJ_UMRAH_KEYWORDS)


class XFetchError(Exception):
    """Raised by search_x_posts so the X Analytics page can show a precise,
    bilingual reason (token missing, plan lacks recent-search, rate limit)."""
    def __init__(self, message_en, message_ar, status=400):
        super().__init__(message_en)
        self.message_en = message_en
        self.message_ar = message_ar
        self.status = status


class RedditFetchError(Exception):
    """Raised by search_reddit_posts so the Reddit Analytics page can show a
    precise, bilingual reason (credentials missing, auth failed, rate limit)."""
    def __init__(self, message_en, message_ar, status=400):
        super().__init__(message_en)
        self.message_en = message_en
        self.message_ar = message_ar
        self.status = status


def search_x_posts(max_results: int = 50) -> dict:
    """Search X (Twitter) for recent public posts about Hajj/Umrah and return
    them as analyzable items. Also pulls a few public metrics per tweet.

    Returns {"query": str, "items": [ {external_id, text, author, created_at,
    likes, source, url, kind}, ... ]}. Raises XFetchError (never a silent
    empty) when the source isn't configured or the API rejects the request."""
    bearer = os.environ.get("X_BEARER_TOKEN")
    if not bearer:
        raise XFetchError(
            "X isn't configured on the server (missing X_BEARER_TOKEN)",
            "خدمة X غير مُفعّلة على الخادم (مفتاح X_BEARER_TOKEN مفقود)", 501)

    max_results = max(10, min(int(max_results or 50), 100))
    query = _keyword_query_x()
    items = []
    try:
        r = requests.get(
            "https://api.twitter.com/2/tweets/search/recent",
            params={
                "query": query,
                "max_results": max_results,
                "tweet.fields": "created_at,author_id,public_metrics,lang",
                "expansions": "author_id",
                "user.fields": "username,name",
            },
            headers={"Authorization": f"Bearer {bearer}"}, timeout=TIMEOUT)
    except Exception as e:
        print(f"[external] search_x_posts network error: {e}")
        raise XFetchError("Network error contacting X",
                          "خطأ في الاتصال بخدمة X", 502)

    if r.status_code != 200:
        body = r.text[:400]
        print(f"[external] search_x_posts HTTP {r.status_code}: {body}")
        if r.status_code in (401, 403):
            raise XFetchError(
                "X rejected the token (invalid, or the plan lacks recent-search access)",
                "رفضت X المفتاح (غير صالح أو الباقة لا تدعم البحث الحديث)", 403)
        if r.status_code == 429:
            raise XFetchError("X rate limit reached — try again shortly",
                              "تم تجاوز حد طلبات X — حاول بعد قليل", 429)
        raise XFetchError(f"X API error (HTTP {r.status_code})",
                          f"خطأ من X (HTTP {r.status_code})", 502)

    payload = r.json() or {}
    # Map author_id -> username for nicer display.
    users = {}
    for u in (payload.get("includes", {}) or {}).get("users", []) or []:
        users[u.get("id")] = u.get("username") or u.get("name")

    for tw in payload.get("data", []) or []:
        text = (tw.get("text") or "").strip()
        if not text:
            continue
        tid = str(tw.get("id"))
        author_id = tw.get("author_id")
        uname = users.get(author_id)
        metrics = tw.get("public_metrics", {}) or {}
        items.append({
            "external_id": "x:" + tid,
            "text": text,
            "author": ("@" + uname) if uname else ("X user " + str(author_id or "")),
            "created_at": tw.get("created_at") or datetime.now(timezone.utc).isoformat(),
            "likes": int(metrics.get("like_count", 0) or 0),
            "source": "x",
            "url": ("https://twitter.com/i/web/status/" + tid),
            "kind": "post",
        })
    return {"query": query, "items": items}


def _reddit_access_token():
    """OAuth2 client-credentials token for Reddit, or raise RedditFetchError."""
    cid = os.environ.get("REDDIT_CLIENT_ID")
    secret = os.environ.get("REDDIT_CLIENT_SECRET")
    agent = os.environ.get("REDDIT_USER_AGENT") or "HajjUmrahSystem/1.0"
    if not (cid and secret):
        raise RedditFetchError(
            "Reddit isn't configured on the server (missing REDDIT_CLIENT_ID/SECRET)",
            "خدمة Reddit غير مُفعّلة على الخادم (مفاتيح REDDIT_CLIENT_ID/SECRET مفقودة)", 501)
    try:
        tok = requests.post(
            "https://www.reddit.com/api/v1/access_token",
            auth=(cid, secret), data={"grant_type": "client_credentials"},
            headers={"User-Agent": agent}, timeout=TIMEOUT)
    except Exception as e:
        print(f"[external] reddit token network error: {e}")
        raise RedditFetchError("Network error contacting Reddit",
                               "خطأ في الاتصال بخدمة Reddit", 502)
    if tok.status_code != 200:
        print(f"[external] reddit token HTTP {tok.status_code}: {tok.text[:300]}")
        raise RedditFetchError(
            "Reddit rejected the credentials (check REDDIT_CLIENT_ID/SECRET)",
            "رفضت Reddit بيانات الاعتماد (تحقق من REDDIT_CLIENT_ID/SECRET)", 403)
    access = (tok.json() or {}).get("access_token")
    if not access:
        raise RedditFetchError("Reddit did not return an access token",
                               "لم تُرجع Reddit رمز الوصول", 502)
    return access, agent


def search_reddit_posts(max_posts: int = 20, max_comments_per_post: int = 8) -> dict:
    """Search Reddit for posts about Hajj/Umrah AND fetch a few top comments on
    each, returning everything as analyzable items.

    Returns {"query": str, "items": [ {external_id, text, author, created_at,
    likes, source, url, kind, title?, community?}, ... ]} where kind is
    'post' or 'comment'. Raises RedditFetchError (never a silent empty) when
    the source isn't configured or the API rejects the request."""
    access, agent = _reddit_access_token()
    max_posts = max(1, min(int(max_posts or 20), 50))
    max_comments_per_post = max(0, min(int(max_comments_per_post or 8), 20))
    query = _keyword_query_reddit()
    headers = {"Authorization": f"Bearer {access}", "User-Agent": agent}
    items = []

    try:
        r = requests.get(
            "https://oauth.reddit.com/search",
            params={"q": query, "limit": max_posts, "sort": "new", "type": "link"},
            headers=headers, timeout=TIMEOUT)
    except Exception as e:
        print(f"[external] search_reddit_posts network error: {e}")
        raise RedditFetchError("Network error contacting Reddit",
                               "خطأ في الاتصال بخدمة Reddit", 502)
    if r.status_code != 200:
        print(f"[external] search_reddit_posts HTTP {r.status_code}: {r.text[:300]}")
        if r.status_code == 429:
            raise RedditFetchError("Reddit rate limit reached — try again shortly",
                                   "تم تجاوز حد طلبات Reddit — حاول بعد قليل", 429)
        raise RedditFetchError(f"Reddit API error (HTTP {r.status_code})",
                               f"خطأ من Reddit (HTTP {r.status_code})", 502)

    for child in (r.json().get("data", {}) or {}).get("children", []) or []:
        d = child.get("data", {}) or {}
        pid = str(d.get("id") or "")
        text = (d.get("selftext") or d.get("title") or "").strip()
        created = d.get("created_utc")
        permalink = ("https://reddit.com" + d["permalink"]) if d.get("permalink") else None
        subreddit = d.get("subreddit")
        if text:
            items.append({
                "external_id": "reddit:" + pid,
                "text": text[:2000],
                "author": "u/" + (d.get("author") or "reddit"),
                "created_at": (datetime.fromtimestamp(created, tz=timezone.utc).isoformat()
                               if created else datetime.now(timezone.utc).isoformat()),
                "likes": int(d.get("ups") or d.get("score") or 0),
                "source": "reddit",
                "url": permalink,
                "kind": "post",
                "title": d.get("title"),
                "community": subreddit,
            })
        # Pull a few top comments for this post (best-effort — a failure on one
        # post must never abort the whole run).
        if max_comments_per_post and pid:
            try:
                cr = requests.get(
                    f"https://oauth.reddit.com/comments/{pid}",
                    params={"limit": max_comments_per_post, "sort": "top", "depth": 1},
                    headers=headers, timeout=TIMEOUT)
                if cr.status_code == 200:
                    listing = cr.json()
                    # listing[1] holds the comment tree; listing[0] is the post.
                    if isinstance(listing, list) and len(listing) > 1:
                        for c in (listing[1].get("data", {}) or {}).get("children", []) or []:
                            cd = c.get("data", {}) or {}
                            if c.get("kind") != "t1":
                                continue  # skip "more" placeholders
                            ctext = (cd.get("body") or "").strip()
                            if not ctext:
                                continue
                            ccreated = cd.get("created_utc")
                            items.append({
                                "external_id": "reddit:" + str(cd.get("id") or ""),
                                "text": ctext[:2000],
                                "author": "u/" + (cd.get("author") or "reddit"),
                                "created_at": (datetime.fromtimestamp(ccreated, tz=timezone.utc).isoformat()
                                               if ccreated else datetime.now(timezone.utc).isoformat()),
                                "likes": int(cd.get("ups") or cd.get("score") or 0),
                                "source": "reddit",
                                "url": (("https://reddit.com" + cd["permalink"]) if cd.get("permalink") else permalink),
                                "kind": "comment",
                                "community": subreddit,
                            })
                else:
                    print(f"[external] reddit comments {pid} HTTP {cr.status_code}")
            except Exception as e:
                print(f"[external] reddit comments {pid} failed: {e}")

    return {"query": query, "items": items}
