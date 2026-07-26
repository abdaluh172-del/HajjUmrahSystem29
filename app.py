# -*- coding: utf-8 -*-
"""
Hajj & Umrah Sentiment Analysis System — Flask backend API.

Run in VS Code:
    1) python -m venv venv
    2) venv\\Scripts\\activate      (Windows)   or   source venv/bin/activate   (Mac/Linux)
    3) pip install -r requirements.txt
    4) python app.py
    -> API runs on http://localhost:5000

Endpoints:
    GET    /api/health
    POST   /api/analyze                body: {"text": "..."}
    GET    /api/comments               query: search, sentiment, category, sort, page, per_page
    POST   /api/comments               body: {"text": "...", "category": "..."}
    PUT    /api/comments/<id>          body: {"text": "..."}
    DELETE /api/comments/<id>
    GET    /api/comments/export        query: format=csv   (ADMIN ONLY)
    GET    /api/login-logs             query: search, sort, page, per_page (ADMIN ONLY)
    GET    /api/dashboard/stats
    POST   /api/auth/login             body: {"email": "...", "password": "..."}
    POST   /api/auth/signup            body: {"name": "...", "email": "...", "password": "..."}
    POST   /api/auth/google            body: {"id_token": "..."}  (real — needs GOOGLE_OAUTH_CLIENT_ID)
    GET    /api/auth/google-config     (public — tells the frontend whether Google Sign-In is enabled)
    PUT    /api/me/comment-lang        body: {"comment_lang": "original"|"ar"|"en"}   (any signed-in account)
    POST   /api/auth/guest             (Continue as Guest — no credentials)
    POST   /api/auth/forgot-password   body: {"email": "..."}  (simulated — no email is actually sent)
    GET    /api/users
    POST   /api/users                  body: {"name","email","role","password"}
    PUT    /api/users/<id>             body: {"name","email","role"}
    DELETE /api/users/<id>

Roles (3 levels):
    admin -> the ONE fixed admin email only. Exclusive rights: users page
             (see registered emails), delete/edit comments, manage users.
             The admin role can never be granted to any other email.
    user  -> anyone who signs up. Can view everything AND add comments.
    guest -> "Continue as Guest" (no account). View only — cannot add
             comments until they create an account.
Fixed admin (cannot be deleted or demoted): see ADMIN_EMAIL below / README.
Admin-only endpoints (require "Authorization: Bearer <token>" from login):
    /api/users (all methods), PUT/DELETE /api/comments/<id>
Write endpoint (admin or registered user token required):
    POST /api/comments
Deployment: init_db() runs at import time, so the app works out of the box
under Gunicorn on Render — no manual database creation needed.

PERMANENT STORAGE (v10):
    Set the DATABASE_URL environment variable to a PostgreSQL connection
    string (e.g. from Neon / Supabase / any managed Postgres) and ALL data —
    user accounts, the admin account, comments and login logs — lives in
    that external database. It is completely independent of the app's
    filesystem, so nothing is ever lost on app shutdown, Restart or
    Redeploy on Render's free plan (no Persistent Disk needed).
    Without DATABASE_URL the app falls back to a local SQLite file, which
    is intended for local development only.
"""
import os
import io
import re
import csv
import ssl
import json
import sqlite3
import secrets
import smtplib
import threading
import time
from email.mime.text import MIMEText
from email.utils import formataddr
from datetime import datetime, timedelta, timezone

from functools import wraps

import requests
from flask import Flask, request, jsonify, g, Response, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
import joblib

from lexicon import find_keywords
from train_model import train_and_save, MODEL_PATH
from dataset import TRAIN_DATA
import external_sources     # v12: official-API fetchers (YouTube / Reddit / X)
import translation          # v13: server-side language detection + translation
import sentiment            # v13: multilingual tiered sentiment engine
import ai_pipeline          # v14: unified pipeline (LLM/moderation/relevance); v15: +category
import google_maps_source   # v15: Google Maps Reviews — auto fetch (once configured) + manual import
import dedup                # v15: cross-source de-duplication
import assistant            # v15: "تعليمات الحج والعمرة" — the Hajj/Umrah AI assistant
import knowledge_base       # v15: curated grounding facts for the assistant

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ---- Persistent storage for the database (v10: external PostgreSQL) ----
# All data — accounts (including the admin), comments and login logs — is
# stored in a real database, NEVER in variables or app memory.
#
#   * DATABASE_URL set (postgres://... or postgresql://...):
#       everything is stored in that external PostgreSQL database. Because
#       it lives OUTSIDE the app's filesystem, closing the app, Restart and
#       Redeploy on Render's free plan never touch it — no Persistent Disk
#       is required. This is the mode to use in production.
#   * DATABASE_URL not set:
#       local-development fallback — a SQLite file next to the code
#       (optionally at DATABASE_PATH). Do NOT rely on this on Render's
#       free plan: its filesystem is wiped on every redeploy.
DATABASE_URL = (os.environ.get("DATABASE_URL") or "").strip()
USE_POSTGRES = DATABASE_URL.startswith(("postgres://", "postgresql://"))
if USE_POSTGRES:
    import psycopg2
    import psycopg2.extras

DB_PATH = os.environ.get("DATABASE_PATH") or os.path.join(BASE_DIR, "hajj_umrah.db")
os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
# v15: the classification taxonomy now lives in ai_pipeline.py (shared with
# the AI classifier) — old category strings already stored on existing rows
# ("Services", "Crowd Management", ...) are left completely untouched; new
# comments are classified into the new taxonomy going forward.
CATEGORIES = ai_pipeline.CATEGORIES
LEGACY_CATEGORIES = ["Services", "Crowd Management", "Transportation", "Food",
                     "Staff Behavior", "Accommodation", "General"]

# ---- Roles & fixed admin account ----
# The system has exactly three roles:
#   admin -> full access — reserved EXCLUSIVELY for ADMIN_EMAIL, never assignable
#   user  -> registered account: view + add comments
#   guest -> no account: view only
VALID_ROLES = ("admin", "user", "guest")
ASSIGNABLE_ROLES = ("user", "guest")  # 'admin' can never be granted to anyone
ADMIN_EMAIL = "abdullah2222@ghjj.sa"  # fixed admin — cannot be deleted or demoted
# The fixed admin's password. Changed to "Abdullah" per request; the account's
# email/name are unchanged. It is NEVER stored as plain text — init_db() hashes
# it with werkzeug's generate_password_hash() and login verifies with
# check_password_hash(). On an existing database (e.g. Render's persistent DB)
# the stored hash is re-synced to this value on startup (see init_db) so the new
# password takes effect on redeploy without wiping any other data. Overridable
# via the ADMIN_PASSWORD env var if the owner wants to set it from the dashboard.
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "Abdullah")
ADMIN_NAME = "Abdullah Alharbi"  # display name of the fixed admin
# Previous default names — renamed automatically; a custom name set later
# through the users page is left untouched (credentials are never reset).
LEGACY_ADMIN_NAMES = ("Admin User", "عبدالله الحربي")
# Old admin emails from previous versions — migrated automatically to the new one.
LEGACY_ADMIN_EMAILS = ("abdullah1222@gmail.com", "admin@hajj.sa")
SECRET_KEY = os.environ.get("SECRET_KEY", "hajj-umrah-dev-secret-change-in-production")
TOKEN_MAX_AGE = 60 * 60 * 12          # 12 hours — default (not "remembered") session
TOKEN_MAX_AGE_REMEMBER = 60 * 60 * 24 * 30  # 30 days — v15 "remember me"
_serializer = URLSafeTimedSerializer(SECRET_KEY, salt="auth-token")

# ---- Email OTP verification (v15.5) ----
# Real email sign-in: a 6-digit code is emailed to the address the person
# types, and the account is created / signed in ONLY after that exact code
# is entered — so nobody can register or sign in with an email they don't
# actually control. SMTP settings come from the environment (set them in the
# Render dashboard). SMTP_PASSWORD for Gmail must be a 16-char App Password
# (Google Account -> Security -> App passwords), NOT the normal password.
SMTP_HOST = (os.environ.get("SMTP_HOST") or "").strip()
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = (os.environ.get("SMTP_USER") or "").strip()
SMTP_PASSWORD = (os.environ.get("SMTP_PASSWORD") or "").strip()
SMTP_FROM = (os.environ.get("SMTP_FROM") or SMTP_USER).strip()
OTP_TTL_SECONDS = int(os.environ.get("OTP_TTL_SECONDS", "600"))      # 10 minutes
OTP_RESEND_SECONDS = int(os.environ.get("OTP_RESEND_SECONDS", "45"))  # min gap between sends
OTP_MAX_ATTEMPTS = int(os.environ.get("OTP_MAX_ATTEMPTS", "5"))


def smtp_configured() -> bool:
    return bool(SMTP_HOST and SMTP_USER and SMTP_PASSWORD)

app = Flask(__name__, static_folder="static", static_url_path="/static")


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

# ---- CORS (manual, no extra dependency required) ----
@app.after_request
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
    return resp


@app.route("/api/<path:_any>", methods=["OPTIONS"])
def options_handler(_any):
    return "", 204


# ---------------------------------------------------------------- #
# Database helpers — PostgreSQL (production) or SQLite (local dev).
# A tiny adapter keeps ONE code path: '?' placeholders + dict-style
# rows everywhere, translated to psycopg2's '%s' when on Postgres.
# ---------------------------------------------------------------- #
class _PgConnection:
    """Adapter that lets the sqlite3-style code run unchanged on PostgreSQL."""

    def __init__(self, conn):
        self._conn = conn

    def execute(self, sql, params=()):
        cur = self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(sql.replace("?", "%s"), tuple(params))
        return cur

    def commit(self):
        self._conn.commit()

    def close(self):
        self._conn.close()


def connect_db():
    """Open a connection to the configured database (Postgres or SQLite)."""
    if USE_POSTGRES:
        return _PgConnection(psycopg2.connect(DATABASE_URL))
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def db_insert(db, sql, params):
    """INSERT and return the new row id on both backends.

    SQLite exposes cursor.lastrowid; PostgreSQL needs RETURNING id."""
    if USE_POSTGRES:
        return db.execute(sql + " RETURNING id", params).fetchone()["id"]
    return db.execute(sql, params).lastrowid


def get_db():
    if "db" not in g:
        g.db = connect_db()
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    # Column types differ slightly between the two engines; everything else
    # (queries, data, behavior) is identical.
    id_pk = "SERIAL PRIMARY KEY" if USE_POSTGRES else "INTEGER PRIMARY KEY AUTOINCREMENT"
    real_t = "DOUBLE PRECISION" if USE_POSTGRES else "REAL"
    conn = connect_db()
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS comments (
            id {id_pk},
            text TEXT NOT NULL,
            category TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            confidence {real_t} NOT NULL,
            keywords TEXT,
            created_at TEXT NOT NULL,
            user_id INTEGER,
            user_name TEXT,
            rating INTEGER,
            status TEXT NOT NULL DEFAULT 'approved',
            likes INTEGER NOT NULL DEFAULT 0,
            source TEXT NOT NULL DEFAULT 'user',
            external_id TEXT,
            original_lang TEXT,
            text_ar TEXT,
            moderation_status TEXT NOT NULL DEFAULT 'approved',
            moderation_flags TEXT,
            moderation_reason TEXT
        )
    """)
    # ---- v12 migration: real reviews system ----
    # Older databases get the new columns added in place — existing data is
    # never touched. user_id/user_name attribute each review to its author;
    # rating is the 1..5 stars; status drives admin moderation
    # ('approved' shown to everyone / 'hidden' visible to the admin only);
    # likes is a cached counter; source tells where the comment came from
    # ('user' or an official API: 'youtube'/'reddit'/'x'); external_id
    # de-duplicates externally fetched comments.
    for coldef in ("user_id INTEGER", "user_name TEXT", "rating INTEGER",
                   "status TEXT NOT NULL DEFAULT 'approved'",
                   "likes INTEGER NOT NULL DEFAULT 0",
                   "source TEXT NOT NULL DEFAULT 'user'", "external_id TEXT",
                   # v13: language of the original text + its Arabic translation
                   # (the original in `text` is NEVER modified)
                   "original_lang TEXT", "text_ar TEXT",
                   # v14: AI moderation — 'approved' shown to everyone;
                   # 'flagged' (violation) and 'rejected' (irrelevant) are
                   # hidden from the public and visible to the admin only,
                   # who can override any decision.
                   "moderation_status TEXT NOT NULL DEFAULT 'approved'",
                   "moderation_flags TEXT", "moderation_reason TEXT",
                   # v15: Google Maps place fields (also usable by any future
                   # location-based source) — real columns for place_name/
                   # country/city so analytics can GROUP BY them directly;
                   # everything else source-specific (place_type, place_url,
                   # Reddit's community/votes/num_comments/permalink/title,
                   # …) goes in the source_meta JSON column to avoid an
                   # ever-growing list of narrow columns.
                   "place_name TEXT", "place_country TEXT", "place_city TEXT",
                   "source_meta TEXT",
                   # v15: cross-source de-duplication fingerprint (dedup.py)
                   "content_fingerprint TEXT"):
        if USE_POSTGRES:
            conn.execute(f"ALTER TABLE comments ADD COLUMN IF NOT EXISTS {coldef}")
        else:
            try:
                conn.execute(f"ALTER TABLE comments ADD COLUMN {coldef}")
            except sqlite3.OperationalError:
                pass  # column already exists
    # One like per account per comment (the Like button toggles it).
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS comment_likes (
            id {id_pk},
            comment_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    # v15.1: per-video YouTube ingestion state — lets the admin panel show
    # "when was each followed video last checked" and how many comments have
    # been collected from it so far, matching the spec's "حفظ تاريخ آخر
    # تحديث لكل فيديو" / "عرض وقت آخر عملية تحديث داخل لوحة الإدارة".
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS youtube_video_state (
            video_id TEXT PRIMARY KEY,
            channel_id TEXT,
            title TEXT,
            first_seen_at TEXT,
            last_checked_at TEXT,
            comments_fetched INTEGER NOT NULL DEFAULT 0,
            last_run_added INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.commit()
    # ---- v12: remove the FAKE seeded comments ----
    # Earlier versions seeded the training sentences as sample comments.
    # Those are deleted once here (only rows that match a training sentence
    # AND have no author and came from no external source — i.e. exactly the
    # old seed rows). Real user reviews and fetched comments always carry an
    # author/source, so they can never be removed by this.
    seed_texts = [txt for txt, _lbl in TRAIN_DATA]
    if seed_texts:
        placeholders = ",".join("?" * len(seed_texts))
        conn.execute(
            f"DELETE FROM comments WHERE user_id IS NULL AND user_name IS NULL "
            f"AND source='user' AND text IN ({placeholders})",
            seed_texts,
        )
    conn.commit()
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS users (
            id {id_pk},
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'guest',
            comment_lang TEXT NOT NULL DEFAULT 'original',
            created_at TEXT NOT NULL
        )
    """)
    # v11 migration: per-user comments display language ('original'/'ar'/'en').
    # Databases created by older versions don't have the column yet — add it
    # without touching any existing data. Each user's choice is stored on
    # their OWN row, so it never affects anyone else and survives restarts,
    # redeploys and signing in from another device.
    if USE_POSTGRES:
        conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS "
                     "comment_lang TEXT NOT NULL DEFAULT 'original'")
    else:
        try:
            conn.execute("ALTER TABLE users ADD COLUMN "
                         "comment_lang TEXT NOT NULL DEFAULT 'original'")
        except sqlite3.OperationalError:
            pass  # column already exists
    # v15: site UI language and "تعليمات الحج والعمرة" (AI assistant) answer
    # language, each independently selectable per the spec ("اختيار لغة
    # مستقلة للموقع والتعليقات والذكاء الاصطناعي"). 'auto' means "follow the
    # site's current UI language" — the default, so nothing changes for
    # existing accounts until they explicitly pick something else.
    for coldef in ("site_lang TEXT NOT NULL DEFAULT 'ar'",
                   "assistant_lang TEXT NOT NULL DEFAULT 'auto'"):
        if USE_POSTGRES:
            conn.execute(f"ALTER TABLE users ADD COLUMN IF NOT EXISTS {coldef}")
        else:
            try:
                conn.execute(f"ALTER TABLE users ADD COLUMN {coldef}")
            except sqlite3.OperationalError:
                pass  # column already exists
    # ---- Login audit log (v9) ----
    # Every sign-in attempt (and every signup, which signs the user in) is
    # recorded here PERMANENTLY: name, email, status ('success'/'failed') and
    # a full UTC timestamp (the UI splits it into date + time columns).
    # Rows live in the same database as users/comments — with DATABASE_URL
    # set that is an external PostgreSQL database, so restarts AND redeploys
    # never touch them. CREATE TABLE IF NOT EXISTS also migrates any database
    # created by an older version automatically.
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS login_logs (
            id {id_pk},
            name TEXT,
            email TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    # ---- Email OTP verification (v15.5) ----
    # One row per email with a pending verification code. Stored in the same
    # database as everything else (external PostgreSQL when DATABASE_URL is
    # set), so a code requested on one worker verifies on another and it all
    # survives restarts. The code is stored HASHED (never in plain text), has
    # an expiry, tracks attempts, and optionally carries the display name a
    # brand-new account should be created with once the code is confirmed.
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS otp_codes (
            email TEXT PRIMARY KEY,
            code_hash TEXT NOT NULL,
            pending_name TEXT,
            expires_at TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            last_sent_at TEXT NOT NULL
        )
    """)
    conn.commit()
    # Always make sure the fixed admin account exists with role=admin,
    # regardless of what else is in the users table.
    admin_row = conn.execute("SELECT id, role, password_hash FROM users WHERE lower(email)=?",
                             (ADMIN_EMAIL,)).fetchone()
    if admin_row is None:
        # Migrate an old seeded admin (if this DB predates the email change).
        legacy = None
        for old_email in LEGACY_ADMIN_EMAILS:
            legacy = conn.execute("SELECT id FROM users WHERE lower(email)=?",
                                  (old_email,)).fetchone()
            if legacy:
                break
        if legacy:
            conn.execute(
                "UPDATE users SET name=?, email=?, password_hash=?, role='admin' WHERE id=?",
                (ADMIN_NAME, ADMIN_EMAIL, generate_password_hash(ADMIN_PASSWORD), legacy["id"]),
            )
            print(f"Migrated fixed admin -> {ADMIN_EMAIL}")
        else:
            conn.execute(
                "INSERT INTO users (name, email, password_hash, role, created_at) VALUES (?,?,?,?,?)",
                (ADMIN_NAME, ADMIN_EMAIL, generate_password_hash(ADMIN_PASSWORD), "admin",
                 datetime.now(timezone.utc).isoformat()),
            )
            print(f"Seeded fixed admin -> email: {ADMIN_EMAIL}")
    else:
        if admin_row["role"] != "admin":
            conn.execute("UPDATE users SET role='admin' WHERE id=?", (admin_row["id"],))
        # Re-sync the fixed admin's password to the configured ADMIN_PASSWORD on
        # every startup — but ONLY when the stored hash doesn't already match it.
        # This makes a password change (here: "Abdullah") take effect on an
        # existing/persistent database after redeploy, without touching the email,
        # the display name, or ANY other account. It's idempotent: once the hash
        # matches, no write happens on subsequent restarts.
        try:
            current_hash = admin_row["password_hash"]
            if not current_hash or not check_password_hash(current_hash, ADMIN_PASSWORD):
                conn.execute("UPDATE users SET password_hash=? WHERE id=?",
                             (generate_password_hash(ADMIN_PASSWORD), admin_row["id"]))
                print("Re-synced fixed admin password to the configured value")
        except Exception as e:  # never let a hash check abort startup
            print(f"Admin password re-sync skipped: {e}")
    # Rename old default admin names to the configured one. Anything else the
    # admin chose later stays as-is — startup never resets existing data.
    conn.execute(
        "UPDATE users SET name=? WHERE lower(email)=? AND name IN (%s)"
        % ",".join("?" * len(LEGACY_ADMIN_NAMES)),
        (ADMIN_NAME, ADMIN_EMAIL, *LEGACY_ADMIN_NAMES),
    )
    # Enforce: the admin role belongs to ADMIN_EMAIL only — demote anyone else.
    conn.execute("UPDATE users SET role='user' WHERE role='admin' AND lower(email)<>?",
                 (ADMIN_EMAIL,))
    # Every row in the users table is a registered account -> role 'user'.
    # (The 'guest' role is only the anonymous no-account mode; legacy DBs that
    # stored registered accounts as 'guest' are migrated to 'user'.)
    conn.execute("UPDATE users SET role='user' WHERE role NOT IN ('admin','user')")
    conn.commit()

    # v12: NO seeding of sample comments anymore — the comments table holds
    # only real user reviews and comments fetched from official APIs.
    conn.close()


# Create the database tables (comments + users + login_logs), seed data, and
# ensure the fixed admin exists — at IMPORT time, not just under `python app.py`.
# Render runs the app with Gunicorn (`gunicorn app:app`), which imports this
# module and never executes the `if __name__ == "__main__"` block; without
# this call the first request would crash with a "no such table" error.
# With DATABASE_URL set this connects to the external PostgreSQL database;
# CREATE TABLE IF NOT EXISTS means existing data is NEVER touched.
init_db()


# ---------------------------------------------------------------- #
# ML model loading (trains automatically the first time it's needed)
# ---------------------------------------------------------------- #
def load_model():
    model_path = os.path.join(BASE_DIR, MODEL_PATH)
    if not os.path.exists(model_path):
        print("No trained model found — training a fresh one now...")
        return train_and_save()
    return joblib.load(model_path)


MODEL = load_model()


def _ml_predict(text: str) -> dict:
    """Probabilities from the trained TF-IDF model — used only as one signal
    inside the tier-3 fallback of the v13 engine, never on its own."""
    probs = MODEL.predict_proba([text])[0]
    return {cls: round(float(p) * 100, 1) for cls, p in zip(MODEL.classes_, probs)}


def run_sentiment_analysis(text: str, is_external: bool = False, place_type: str = None):
    """v14: the FULL unified AI pipeline for one comment (ai_pipeline.py):
    detect language -> translate (original kept, Arabic stored separately)
    -> sentiment (LLM / multilingual transformer / VADER / built-in)
    -> content moderation -> Hajj&Umrah relevance. Explanatory keywords are
    extracted from the original + Arabic translation for display only.
    v15: also classifies into the topic taxonomy and computes a
    cross-source de-duplication fingerprint (dedup.py)."""
    res = ai_pipeline.process(text, ml_predict=_ml_predict, is_external=is_external, place_type=place_type)
    pos_hits, neg_hits = find_keywords(text + " " + (res.get("text_ar") or ""))
    return {
        "label": res["sentiment"],
        "confidence": res["confidence"],
        "scores": res.get("scores", {}),
        "engine": res.get("engine"),
        "category": res.get("category") or "general",
        "original_lang": res.get("detected_language"),
        "text_ar": res.get("text_ar"),
        "moderation_status": res["moderation_status"],
        "moderation_flags": ",".join(res.get("moderation_flags") or []),
        "moderation_reason": res.get("moderation_reason") or None,
        "content_fingerprint": res.get("content_fingerprint"),
        "positive_keywords": pos_hits,
        "negative_keywords": neg_hits,
        "keywords": pos_hits + neg_hits,
    }


def check_duplicate(db, fingerprint: str, text_en: str, window: int = 400):
    """v15: cross-source de-duplication. First pass is an exact fingerprint
    match (cheap, indexed lookup); second pass is a fuzzy comparison against
    a bounded recent window of EXTERNAL comments (external reviews are where
    the same opinion can plausibly show up on more than one platform — site
    reviews are each pilgrim's own testimony and are never treated as
    duplicates of each other). Returns the id of the existing duplicate, or
    None. NEVER raises — a dedup failure must not block ingestion."""
    if not fingerprint:
        return None
    try:
        exact = db.execute(
            "SELECT id FROM comments WHERE content_fingerprint=? AND source<>'user' LIMIT 1",
            (fingerprint,)).fetchone()
        if exact:
            return exact["id"]
    except Exception as e:
        print(f"[dedup] exact-match lookup failed: {e}")
        return None
    try:
        recent = db.execute(
            "SELECT id, text, text_ar FROM comments WHERE source<>'user' "
            "ORDER BY id DESC LIMIT ?", (window,)).fetchall()
        norm_target = dedup.normalize(text_en)
        for r in recent:
            candidate = dedup.normalize(r["text_ar"] or r["text"])
            if candidate and dedup.is_near_duplicate(norm_target, [candidate]):
                return r["id"]
    except Exception as e:
        print(f"[dedup] fuzzy lookup failed: {e}")
    return None


# ---------------------------------------------------------------- #
# Auth helpers (token-based, stateless)
# ---------------------------------------------------------------- #
def make_token(user_id: int, remember: bool = True) -> str:
    """v15: "remember me" — a token issued with remember=True stays valid
    for TOKEN_MAX_AGE_REMEMBER (30 days) instead of the default 12 hours.
    The choice is embedded in the signed payload itself (not just enforced
    client-side), so unchecking "remember me" genuinely shortens the
    session even if the browser keeps the token around."""
    return _serializer.dumps({"uid": user_id, "remember": bool(remember)})


def current_user():
    """Return the users row for the Bearer token in this request, or None."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    token = auth[7:].strip()
    # Decode against the LONGEST possible lifetime first so we can read the
    # "remember" flag inside the payload, then re-check with the correct,
    # shorter max_age for non-remembered sessions.
    try:
        data = _serializer.loads(token, max_age=TOKEN_MAX_AGE_REMEMBER)
    except (BadSignature, SignatureExpired):
        return None
    if not data.get("remember", True):
        try:
            _serializer.loads(token, max_age=TOKEN_MAX_AGE)
        except SignatureExpired:
            return None
    return get_db().execute("SELECT * FROM users WHERE id=?", (data.get("uid"),)).fetchone()


def api_error(msg_en: str, msg_ar: str, status: int, code: str = None):
    body = {"error": msg_en, "error_ar": msg_ar}
    if code:
        body["code"] = code
    return jsonify(body), status


def admin_required(fn):
    """Protect admin-only endpoints (users page + destructive operations).

    Double lock: the token's account must have role='admin' AND its email must
    be the fixed ADMIN_EMAIL — so admin rights can never belong to any other
    email, even if a database row were tampered with.
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        user = current_user()
        if user is None or user["role"] != "admin" or not is_fixed_admin(user):
            return api_error("Admin access required",
                             "هذه العملية مخصصة لحساب الأدمن فقط", 403)
        return fn(*args, **kwargs)
    return wrapper


def write_required(fn):
    """Adding comments requires a registered account (user or admin token).

    Guests carry no token, so they can view comments but cannot write."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        user = current_user()
        if user is None or user["role"] not in ("admin", "user"):
            return api_error("Create an account to add comments",
                             "أنشئ حسابًا لتتمكن من إضافة التعليقات",
                             401, code="signup_required")
        return fn(*args, **kwargs)
    return wrapper


def is_fixed_admin(row) -> bool:
    return (row["email"] or "").lower() == ADMIN_EMAIL


def record_login(db, name, email, status):
    """Persist one sign-in attempt to the login_logs table (never deleted).

    status is 'success' or 'failed'. name may be empty when the email is
    unknown (a failed attempt for an address that has no account)."""
    db.execute(
        "INSERT INTO login_logs (name, email, status, created_at) VALUES (?,?,?,?)",
        (name or "", email, status, datetime.now(timezone.utc).isoformat()),
    )
    db.commit()


# ---------------------------------------------------------------- #
# Email OTP verification helpers (v15.5)
# ---------------------------------------------------------------- #
def _send_email(to_email: str, subject: str, body: str):
    """Send one plain-text email over SMTP. Raises on failure so the caller
    can report it — the real SMTP error is also printed to the Render logs."""
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = formataddr(("نظام الحج والعمرة", SMTP_FROM))
    msg["To"] = to_email
    context = ssl.create_default_context()
    if SMTP_PORT == 465:
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context, timeout=30) as server:
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_FROM, [to_email], msg.as_string())
    else:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.ehlo()
            server.starttls(context=context)
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_FROM, [to_email], msg.as_string())


def _otp_email_body(code: str) -> str:
    minutes = OTP_TTL_SECONDS // 60
    return (
        "السلام عليكم ورحمة الله،\n\n"
        f"رمز التحقق الخاص بك هو: {code}\n\n"
        f"هذا الرمز صالح لمدة {minutes} دقيقة. لا تُشاركه مع أي أحد.\n"
        "إذا لم تطلب هذا الرمز فتجاهل هذه الرسالة.\n\n"
        "نظام تحليل ملاحظات الحج والعمرة"
    )


def _issue_otp(db, email: str, pending_name: str = None):
    """Generate a 6-digit code, store it HASHED with an expiry, and email it.
    Enforces a minimum gap between sends. Returns None on success or an
    (message_en, message_ar, status) tuple describing why it couldn't send."""
    now = datetime.now(timezone.utc)
    existing = db.execute("SELECT last_sent_at FROM otp_codes WHERE email=?", (email,)).fetchone()
    if existing and existing["last_sent_at"]:
        try:
            last = datetime.fromisoformat(existing["last_sent_at"])
            if (now - last).total_seconds() < OTP_RESEND_SECONDS:
                wait = int(OTP_RESEND_SECONDS - (now - last).total_seconds())
                return ("Please wait before requesting another code",
                        f"يرجى الانتظار {wait} ثانية قبل طلب رمز جديد", 429)
        except ValueError:
            pass

    code = "{:06d}".format(secrets.randbelow(1_000_000))
    code_hash = generate_password_hash(code)
    expires_at = (now + timedelta(seconds=OTP_TTL_SECONDS)).isoformat()
    now_iso = now.isoformat()
    # Upsert (works the same on SQLite and PostgreSQL without ON CONFLICT).
    if existing:
        db.execute(
            "UPDATE otp_codes SET code_hash=?, pending_name=?, expires_at=?, "
            "attempts=0, last_sent_at=? WHERE email=?",
            (code_hash, pending_name, expires_at, now_iso, email))
    else:
        db.execute(
            "INSERT INTO otp_codes (email, code_hash, pending_name, expires_at, "
            "attempts, last_sent_at) VALUES (?,?,?,?,?,?)",
            (email, code_hash, pending_name, expires_at, 0, now_iso))
    db.commit()

    try:
        _send_email(email, "رمز التحقق - نظام الحج والعمرة", _otp_email_body(code))
    except Exception as e:
        print(f"[otp] failed to send code to {email}: {e}")
        return ("Could not send the verification email — try again later",
                "تعذّر إرسال رمز التحقق إلى بريدك — حاول لاحقًا", 502)
    print(f"[otp] verification code sent to {email}")
    return None


def _check_otp(db, email: str, code: str):
    """Validate a submitted code. Returns (ok: bool, pending_name, err_tuple).
    On success the code row is deleted (single use)."""
    row = db.execute("SELECT * FROM otp_codes WHERE email=?", (email,)).fetchone()
    if not row:
        return False, None, ("No active code for this email — request a new one",
                             "لا يوجد رمز فعّال لهذا البريد — اطلب رمزًا جديدًا", 400)
    now = datetime.now(timezone.utc)
    try:
        if now > datetime.fromisoformat(row["expires_at"]):
            db.execute("DELETE FROM otp_codes WHERE email=?", (email,))
            db.commit()
            return False, None, ("The code has expired — request a new one",
                                "انتهت صلاحية الرمز — اطلب رمزًا جديدًا", 400)
    except ValueError:
        pass
    if row["attempts"] >= OTP_MAX_ATTEMPTS:
        db.execute("DELETE FROM otp_codes WHERE email=?", (email,))
        db.commit()
        return False, None, ("Too many attempts — request a new code",
                            "تجاوزت عدد المحاولات — اطلب رمزًا جديدًا", 429)
    if not check_password_hash(row["code_hash"], (code or "").strip()):
        db.execute("UPDATE otp_codes SET attempts=attempts+1 WHERE email=?", (email,))
        db.commit()
        return False, None, ("Incorrect code", "الرمز غير صحيح", 400)
    pending_name = row["pending_name"]
    db.execute("DELETE FROM otp_codes WHERE email=?", (email,))  # single use
    db.commit()
    return True, pending_name, None


# ---------------------------------------------------------------- #
# Routes
# ---------------------------------------------------------------- #
@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model_classes": list(MODEL.classes_)})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400
    result = run_sentiment_analysis(text)
    return jsonify(result)


@app.route("/api/comments", methods=["GET"])
def list_comments():
    """List comments/reviews, newest first (v12).

    Everyone sees only status='approved'. The admin can pass
    status=all|approved|hidden to run the moderation panel.
    Filters: search, sentiment, category, rating (1..5), source
    (user/youtube/reddit/x), date_from, date_to (YYYY-MM-DD), language
    (original_lang, e.g. 'ar'/'en'), country (place_country), city (place_city).
    """
    db = get_db()
    me = current_user()
    is_admin = me is not None and me["role"] == "admin" and is_fixed_admin(me)
    search = request.args.get("search", "").strip()
    sentiment = request.args.get("sentiment", "all")
    category = request.args.get("category", "all")
    rating = request.args.get("rating", "all")
    source = request.args.get("source", "all")
    language = request.args.get("language", "all").strip()  # v15.1
    country = request.args.get("country", "all").strip()    # v15.1
    city = request.args.get("city", "all").strip()           # v15.1
    date_from = request.args.get("date_from", "").strip()
    date_to = request.args.get("date_to", "").strip()
    status = request.args.get("status", "all" if is_admin else "approved")
    sort = request.args.get("sort", "date")
    page = max(1, int(request.args.get("page", 1)))
    per_page = max(1, int(request.args.get("per_page", 10)))

    moderation = request.args.get("moderation", "all" if is_admin else "approved")
    query = "SELECT * FROM comments WHERE 1=1"
    params = []
    if not is_admin:
        status = "approved"      # non-admins can never see admin-hidden comments
        moderation = "approved"  # v14: ...nor AI-flagged / irrelevant ones
    if status != "all":
        query += " AND status = ?"
        params.append(status)
    if moderation != "all":
        query += " AND COALESCE(moderation_status,'approved') = ?"
        params.append(moderation)
    if search:
        # lower() on both sides -> case-insensitive search on SQLite AND Postgres
        query += " AND (lower(text) LIKE lower(?) OR lower(keywords) LIKE lower(?) OR lower(COALESCE(user_name,'')) LIKE lower(?))"
        params += [f"%{search}%", f"%{search}%", f"%{search}%"]
    if sentiment != "all":
        query += " AND sentiment = ?"
        params.append(sentiment)
    if language != "all":
        # A comment written originally in Arabic has no original_lang value
        # stored (only non-Arabic originals get one — see ai_pipeline.py),
        # so 'ar' also matches rows where the column is NULL/empty.
        if language == "ar":
            query += " AND (original_lang IS NULL OR original_lang = '' OR original_lang = 'ar')"
        else:
            query += " AND original_lang = ?"
            params.append(language)
    if country != "all":
        query += " AND place_country = ?"
        params.append(country)
    if city != "all":
        query += " AND place_city = ?"
        params.append(city)
    if category != "all":
        query += " AND category = ?"
        params.append(category)
    if rating != "all":
        query += " AND rating = ?"
        params.append(int(rating))
    if source != "all":
        query += " AND source = ?"
        params.append(source)
    # created_at is an ISO string, so plain string comparison sorts correctly.
    if date_from:
        query += " AND created_at >= ?"
        params.append(date_from)
    if date_to:
        query += " AND created_at <= ?"
        params.append(date_to + "T23:59:59.999999+00:00")

    if sort == "likes":
        order = "likes DESC, created_at DESC"
    elif sort == "confidence":
        order = "confidence DESC"
    else:
        order = "created_at DESC"  # default: newest first
    rows = db.execute(query + f" ORDER BY {order}", params).fetchall()
    total = len(rows)
    start = (page - 1) * per_page
    page_rows = [dict(r) for r in rows[start:start + per_page]]

    # Mark which of the returned comments the signed-in account already liked,
    # so the Like button can render its state.
    liked_ids = set()
    if me is not None and page_rows:
        ph = ",".join("?" * len(page_rows))
        liked = db.execute(
            f"SELECT comment_id FROM comment_likes WHERE user_id=? AND comment_id IN ({ph})",
            [me["id"]] + [r["id"] for r in page_rows]).fetchall()
        liked_ids = {r["comment_id"] for r in liked}
    for r in page_rows:
        r["liked_by_me"] = r["id"] in liked_ids

    return jsonify({
        "total": total,
        "page": page,
        "per_page": per_page,
        "items": page_rows,
    })


@app.route("/api/comments/filter-options")
def comments_filter_options():
    """Distinct values currently present for the language/country/city
    filters (v15.1) — lets the frontend populate those dropdowns from real
    data instead of a hardcoded list that inevitably goes stale as new
    comments arrive from new places/languages. Only considers comments a
    regular visitor could see (approved + not flagged/rejected)."""
    db = get_db()
    visible = "WHERE status='approved' AND COALESCE(moderation_status,'approved')='approved'"
    languages = sorted({
        (r["original_lang"] or "ar") for r in
        db.execute(f"SELECT DISTINCT original_lang FROM comments {visible}").fetchall()
    })
    countries = sorted({
        r["place_country"] for r in
        db.execute(f"SELECT DISTINCT place_country FROM comments {visible} "
                   "AND place_country IS NOT NULL AND place_country != ''").fetchall()
    })
    cities = sorted({
        r["place_city"] for r in
        db.execute(f"SELECT DISTINCT place_city FROM comments {visible} "
                   "AND place_city IS NOT NULL AND place_city != ''").fetchall()
    })
    return jsonify({"languages": languages, "countries": countries, "cities": cities})


@app.route("/api/comments/summary")
def comments_summary():
    """Overall rating summary for the reviews header (v12): average stars,
    review count and the 5..1 star distribution — approved comments only."""
    db = get_db()
    rows = db.execute(
        "SELECT rating FROM comments WHERE status='approved' AND COALESCE(moderation_status,'approved')='approved' AND rating IS NOT NULL").fetchall()
    ratings = [r["rating"] for r in rows]
    dist = {str(s): ratings.count(s) for s in (5, 4, 3, 2, 1)}
    total_all = db.execute(
        "SELECT COUNT(*) AS c FROM comments WHERE status='approved' AND COALESCE(moderation_status,'approved')='approved'").fetchone()["c"]
    return jsonify({
        "avg_rating": round(sum(ratings) / len(ratings), 2) if ratings else 0,
        "rated_count": len(ratings),
        "total_comments": total_all,
        "distribution": dist,
    })


@app.route("/api/insights")
def insights():
    """AI summary (v12): the most mentioned problems and strengths.

    Aggregates the extracted keywords of approved comments — keywords from
    negative comments become 'top problems', from positive ones
    'top strengths' — plus overall sentiment counts for the charts."""
    db = get_db()
    rows = db.execute(
        "SELECT sentiment, keywords FROM comments WHERE status='approved' AND COALESCE(moderation_status,'approved')='approved'").fetchall()
    problems, strengths = {}, {}
    counts = {"positive": 0, "negative": 0, "neutral": 0}
    for r in rows:
        counts[r["sentiment"]] = counts.get(r["sentiment"], 0) + 1
        bucket = strengths if r["sentiment"] == "positive" else (
            problems if r["sentiment"] == "negative" else None)
        if bucket is not None and r["keywords"]:
            for kw in r["keywords"].split(","):
                kw = kw.strip()
                if kw:
                    bucket[kw] = bucket.get(kw, 0) + 1
    top = lambda d: sorted(d.items(), key=lambda x: x[1], reverse=True)[:8]
    return jsonify({
        "top_problems": top(problems),
        "top_strengths": top(strengths),
        "counts": counts,
        "total": len(rows),
    })


@app.route("/api/comments", methods=["POST"])
@write_required
def add_comment():
    """Add a real review (v12): text + optional 1..5 star rating.

    The signed-in account is recorded as the author (user_id + user_name),
    the comment is analyzed by the AI model immediately, stored permanently
    in the database and shown newest-first. v15: if no category is chosen
    (or "auto" is sent), the AI classifies it automatically into the topic
    taxonomy — the dropdown remains available for anyone who wants to pick
    one manually."""
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get("text") or "").strip()
    category = (data.get("category") or "").strip()
    rating = data.get("rating")
    if not text:
        return jsonify({"error": "text is required"}), 400
    if rating is not None:
        try:
            rating = int(rating)
        except (TypeError, ValueError):
            return api_error("rating must be a number 1..5", "التقييم يجب أن يكون رقمًا من 1 إلى 5", 400)
        if not 1 <= rating <= 5:
            return api_error("rating must be between 1 and 5", "التقييم يجب أن يكون بين 1 و 5", 400)

    me = current_user()  # write_required guarantees a registered account
    result = run_sentiment_analysis(text)
    final_category = category if category and category.lower() != "auto" else result["category"]
    db = get_db()
    new_id = db_insert(
        db,
        "INSERT INTO comments (text, category, sentiment, confidence, keywords, created_at, "
        "user_id, user_name, rating, status, likes, source, original_lang, text_ar, "
        "moderation_status, moderation_flags, moderation_reason, content_fingerprint) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (text, final_category, result["label"], result["confidence"],
         ",".join(result["keywords"]), datetime.now(timezone.utc).isoformat(),
         me["id"], me["name"], rating, "approved", 0, "user",
         result.get("original_lang"), result.get("text_ar"),
         result["moderation_status"], result["moderation_flags"], result["moderation_reason"],
         result.get("content_fingerprint")),
    )
    db.commit()
    new_row = db.execute("SELECT * FROM comments WHERE id=?", (new_id,)).fetchone()
    return jsonify(dict(new_row)), 201


@app.route("/api/comments/<int:comment_id>/like", methods=["POST"])
@write_required
def toggle_like(comment_id):
    """Like / unlike a comment (v12). One like per account, toggled."""
    me = current_user()
    db = get_db()
    row = db.execute("SELECT id FROM comments WHERE id=?", (comment_id,)).fetchone()
    if not row:
        return jsonify({"error": "not found"}), 404
    existing = db.execute(
        "SELECT id FROM comment_likes WHERE comment_id=? AND user_id=?",
        (comment_id, me["id"])).fetchone()
    if existing:
        db.execute("DELETE FROM comment_likes WHERE id=?", (existing["id"],))
        liked = False
    else:
        db.execute("INSERT INTO comment_likes (comment_id, user_id, created_at) VALUES (?,?,?)",
                   (comment_id, me["id"], datetime.now(timezone.utc).isoformat()))
        liked = True
    # Keep the cached counter exact by recounting from the likes table.
    n = db.execute("SELECT COUNT(*) AS c FROM comment_likes WHERE comment_id=?",
                   (comment_id,)).fetchone()["c"]
    db.execute("UPDATE comments SET likes=? WHERE id=?", (n, comment_id))
    db.commit()
    return jsonify({"id": comment_id, "likes": n, "liked_by_me": liked})


@app.route("/api/comments/<int:comment_id>/moderation", methods=["PUT"])
@admin_required
def set_comment_moderation(comment_id):
    """v14: admin override of the AI moderation decision.

    'approved' publishes a comment the AI flagged/rejected (false positive);
    'flagged' / 'rejected' hides one the AI let through. The AI's flags and
    reason stay stored for the audit trail."""
    data = request.get_json(force=True, silent=True) or {}
    status = (data.get("status") or "").strip().lower()
    if status not in ("approved", "flagged", "rejected"):
        return api_error("status must be approved/flagged/rejected",
                         "الحالة يجب أن تكون approved أو flagged أو rejected", 400)
    db = get_db()
    row = db.execute("SELECT id FROM comments WHERE id=?", (comment_id,)).fetchone()
    if not row:
        return jsonify({"error": "not found"}), 404
    db.execute("UPDATE comments SET moderation_status=? WHERE id=?", (status, comment_id))
    db.commit()
    return jsonify(dict(db.execute("SELECT * FROM comments WHERE id=?", (comment_id,)).fetchone()))


@app.route("/api/comments/<int:comment_id>/status", methods=["PUT"])
@admin_required
def set_comment_status(comment_id):
    """Admin moderation (v12): approve or hide a comment.

    'hidden' keeps the comment in the database but removes it from every
    non-admin view; 'approved' publishes it again. (Deleting stays a
    separate, explicit admin action.)"""
    data = request.get_json(force=True, silent=True) or {}
    status = (data.get("status") or "").strip().lower()
    if status not in ("approved", "hidden"):
        return api_error("status must be 'approved' or 'hidden'",
                         "الحالة يجب أن تكون approved أو hidden", 400)
    db = get_db()
    row = db.execute("SELECT id FROM comments WHERE id=?", (comment_id,)).fetchone()
    if not row:
        return jsonify({"error": "not found"}), 404
    db.execute("UPDATE comments SET status=? WHERE id=?", (status, comment_id))
    db.commit()
    return jsonify(dict(db.execute("SELECT * FROM comments WHERE id=?", (comment_id,)).fetchone()))


@app.route("/api/comments/<int:comment_id>", methods=["PUT"])
@admin_required
def update_comment(comment_id):
    """Admin edit (v12): change the text (re-analyzed by the AI model) and
    optionally the rating, category or status. Author info never changes."""
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400
    result = run_sentiment_analysis(text)
    db = get_db()
    db.execute(
        "UPDATE comments SET text=?, sentiment=?, confidence=?, keywords=?, "
        "original_lang=?, text_ar=?, moderation_status=?, moderation_flags=?, "
        "moderation_reason=?, content_fingerprint=? WHERE id=?",
        (text, result["label"], result["confidence"], ",".join(result["keywords"]),
         result.get("original_lang"), result.get("text_ar"),
         result["moderation_status"], result["moderation_flags"],
         result["moderation_reason"], result.get("content_fingerprint"), comment_id),
    )
    # If the admin didn't explicitly send a category, keep the AI's fresh
    # classification of the (possibly edited) text in sync.
    if not data.get("category"):
        db.execute("UPDATE comments SET category=? WHERE id=?", (result["category"], comment_id))
    if data.get("rating") is not None:
        try:
            rating = int(data["rating"])
        except (TypeError, ValueError):
            rating = None
        if rating is not None and 1 <= rating <= 5:
            db.execute("UPDATE comments SET rating=? WHERE id=?", (rating, comment_id))
    if data.get("category"):
        db.execute("UPDATE comments SET category=? WHERE id=?", (data["category"], comment_id))
    if data.get("status") in ("approved", "hidden"):
        db.execute("UPDATE comments SET status=? WHERE id=?", (data["status"], comment_id))
    db.commit()
    row = db.execute("SELECT * FROM comments WHERE id=?", (comment_id,)).fetchone()
    if not row:
        return jsonify({"error": "not found"}), 404
    return jsonify(dict(row))


@app.route("/api/comments/<int:comment_id>", methods=["DELETE"])
@admin_required
def delete_comment(comment_id):
    db = get_db()
    db.execute("DELETE FROM comment_likes WHERE comment_id=?", (comment_id,))  # v12: clean its likes too
    db.execute("DELETE FROM comments WHERE id=?", (comment_id,))
    db.commit()
    return jsonify({"deleted": comment_id})


@app.route("/api/comments/export")
@admin_required  # v9: exporting data (CSV/JSON) is an admin-only tool now
def export_comments():
    fmt = request.args.get("format", "csv")
    db = get_db()
    rows = db.execute("SELECT * FROM comments ORDER BY created_at DESC").fetchall()

    if fmt == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["id", "text", "category", "sentiment", "confidence", "created_at",
                         "user_name", "rating", "status", "likes", "source"])  # v12 columns
        for r in rows:
            r = dict(r)
            writer.writerow([r["id"], r["text"], r["category"], r["sentiment"], r["confidence"],
                             r["created_at"], r.get("user_name") or "", r.get("rating") or "",
                             r.get("status") or "approved", r.get("likes") or 0, r.get("source") or "user"])
        return Response(
            output.getvalue(), mimetype="text/csv",
            headers={"Content-Disposition": "attachment;filename=comments.csv"},
        )
    return jsonify([dict(r) for r in rows])


def user_public(row):
    d = dict(row)
    d.pop("password_hash", None)
    d["fixed"] = is_fixed_admin(row)  # lets the UI mark the protected account without hardcoding its email
    d.setdefault("comment_lang", "original")  # v11: per-user comments display language
    d.setdefault("site_lang", "ar")           # v15: per-user site UI language
    d.setdefault("assistant_lang", "auto")    # v15: per-user AI assistant answer language
    return d


# ---- Authentication (simple, session-less: returns the user object) ---- #
# v15: full language support — Arabic, English, Turkish, Urdu, Hindi, Hebrew.
# Site UI, comments display, and the AI assistant each have an INDEPENDENT
# language choice per the spec ("اختيار لغة مستقلة للموقع والتعليقات
# والذكاء الاصطناعي"); 'auto' (assistant only) follows the site UI language.
UI_LANGS = ("ar", "en", "tr", "ur", "hi", "he")
VALID_COMMENT_LANGS = ("original",) + UI_LANGS
VALID_ASSISTANT_LANGS = ("auto",) + UI_LANGS


@app.route("/api/me/comment-lang", methods=["PUT"])
def set_comment_lang():
    """Save the signed-in account's comments display language (v11).

    Available to EVERY registered account — regular users AND the admin
    (guests carry no token). The choice is stored on the account's own row
    in the users table, so it is personal: one user's choice never affects
    anyone else. Like all data it lives in the permanent database, so it
    survives logout/login, app restarts and redeploys, and follows the
    account even from another device. Original comment text in the database
    is never modified — translation is display-only in the browser.
    """
    user = current_user()
    if user is None or user["role"] not in ("admin", "user"):
        return api_error("Sign in to save your comments language",
                         "سجّل الدخول لحفظ لغة عرض التعليقات", 401)
    data = request.get_json(force=True, silent=True) or {}
    lang = (data.get("comment_lang") or "").strip().lower()
    if lang not in VALID_COMMENT_LANGS:
        return api_error("comment_lang must be one of: original, ar, en",
                         "قيمة اللغة يجب أن تكون: original أو ar أو en", 400)
    db = get_db()
    db.execute("UPDATE users SET comment_lang=? WHERE id=?", (lang, user["id"]))
    db.commit()
    row = db.execute("SELECT * FROM users WHERE id=?", (user["id"],)).fetchone()
    return jsonify(user_public(row))


@app.route("/api/me/language-prefs", methods=["PUT"])
def set_language_prefs():
    """v15: save the site UI language and/or the AI assistant answer
    language for the signed-in account — independent from comment_lang
    above, and from each other. Body: any of {"site_lang", "assistant_lang"}.
    Guests have no account to save preferences on; the frontend keeps their
    choice in localStorage only for the current browser."""
    user = current_user()
    if user is None or user["role"] not in ("admin", "user"):
        return api_error("Sign in to save your language preferences",
                         "سجّل الدخول لحفظ تفضيلات اللغة", 401)
    data = request.get_json(force=True, silent=True) or {}
    updates = {}
    if "site_lang" in data:
        v = (data.get("site_lang") or "").strip().lower()
        if v not in UI_LANGS:
            return api_error("site_lang must be one of: " + ", ".join(UI_LANGS),
                             "قيمة لغة الموقع غير صحيحة", 400)
        updates["site_lang"] = v
    if "assistant_lang" in data:
        v = (data.get("assistant_lang") or "").strip().lower()
        if v not in VALID_ASSISTANT_LANGS:
            return api_error("assistant_lang must be one of: " + ", ".join(VALID_ASSISTANT_LANGS),
                             "قيمة لغة المساعد غير صحيحة", 400)
        updates["assistant_lang"] = v
    if not updates:
        return api_error("no valid fields provided", "لا توجد حقول صالحة", 400)
    db = get_db()
    db.execute("UPDATE users SET " + ", ".join(f"{k}=?" for k in updates) + " WHERE id=?",
              (*updates.values(), user["id"]))
    db.commit()
    row = db.execute("SELECT * FROM users WHERE id=?", (user["id"],)).fetchone()
    return jsonify(user_public(row))


@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json(force=True, silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    remember = bool(data.get("remember", True))  # v15: "remember me" checkbox (default on)
    db = get_db()
    row = db.execute("SELECT * FROM users WHERE lower(email)=?", (email,)).fetchone()
    if not row or not check_password_hash(row["password_hash"], password):
        # Audit trail: failed attempts are stored too (with the account's name
        # when the email exists, empty otherwise).
        if email:
            record_login(db, row["name"] if row else "", email, "failed")
        return api_error("Invalid email or password",
                         "البريد الإلكتروني أو كلمة المرور غير صحيحة", 401)
    record_login(db, row["name"], row["email"], "success")
    return jsonify({"user": user_public(row), "token": make_token(row["id"], remember=remember),
                    "remember": remember})


@app.route("/api/auth/signup", methods=["POST"])
def signup():
    data = request.get_json(force=True, silent=True) or {}
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    remember = bool(data.get("remember", True))
    if not name:
        return api_error("Name is required", "الاسم مطلوب", 400)
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        return api_error("Enter a valid email (e.g. name@example.com)",
                         "أدخل بريدًا إلكترونيًا صحيحًا (مثل name@example.com)", 400)
    if len(password) < 6:
        return api_error("Password must be at least 6 characters",
                         "كلمة المرور يجب أن تكون 6 أحرف على الأقل", 400)
    db = get_db()
    exists = db.execute("SELECT id FROM users WHERE lower(email)=?", (email,)).fetchone()
    if exists:
        return api_error("An account with this email already exists — sign in instead",
                         "هذا البريد مسجّل مسبقًا — سجّل الدخول بدلاً من ذلك", 409)
    # Every signup is a regular registered user; the admin role is never granted.
    new_id = db_insert(
        db,
        "INSERT INTO users (name, email, password_hash, role, created_at) VALUES (?,?,?,?,?)",
        (name, email, generate_password_hash(password), "user", datetime.now(timezone.utc).isoformat()),
    )
    db.commit()
    row = db.execute("SELECT * FROM users WHERE id=?", (new_id,)).fetchone()
    # Signing up signs the user in immediately, so it's recorded in the
    # login log as a successful sign-in as well.
    record_login(db, row["name"], row["email"], "success")
    return jsonify({"user": user_public(row), "token": make_token(row["id"], remember=remember),
                    "remember": remember}), 201


# ---- Email OTP sign-in (v15.5) — real email verification ----
# Passwordless sign-in that proves the person owns the email: a 6-digit code
# is emailed, and only the exact code creates/opens the account. This is the
# fix for "any email could sign in without verification": no code, no entry.
@app.route("/api/auth/request-otp", methods=["POST"])
def request_otp():
    """Body: {"email": "...", "name": "(optional, for a new account)"}.
    Emails a verification code to that address. Never reveals whether the
    email already has an account (same response either way)."""
    if not smtp_configured():
        return api_error("Email verification isn't configured on the server yet",
                         "خدمة التحقق بالبريد غير مُفعّلة على الخادم بعد", 501,
                         code="email_not_configured")
    data = request.get_json(force=True, silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    name = (data.get("name") or "").strip() or None
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        return api_error("Enter a valid email (e.g. name@example.com)",
                         "أدخل بريدًا إلكترونيًا صحيحًا (مثل name@example.com)", 400)
    db = get_db()
    # If the account already exists, keep its stored name; otherwise remember
    # the name the code should create the account with once it's verified.
    existing = db.execute("SELECT name FROM users WHERE lower(email)=?", (email,)).fetchone()
    pending_name = existing["name"] if existing else name
    err = _issue_otp(db, email, pending_name=pending_name)
    if err:
        return api_error(err[0], err[1], err[2])
    return jsonify({"ok": True, "sent": True,
                    "message": "تم إرسال رمز التحقق إلى بريدك الإلكتروني.",
                    "message_en": "A verification code has been sent to your email.",
                    "is_new_account": existing is None,
                    "expires_in": OTP_TTL_SECONDS})


@app.route("/api/auth/verify-otp", methods=["POST"])
def verify_otp_route():
    """Body: {"email": "...", "code": "123456", "name": "(optional)", "remember": true}.
    On the correct code: find-or-create the account by verified email and
    return a session token — identical shape to /auth/login & /auth/signup."""
    data = request.get_json(force=True, silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    code = (data.get("code") or "").strip()
    remember = bool(data.get("remember", True))
    if not email or not code:
        return api_error("Email and code are required", "البريد والرمز مطلوبان", 400)
    db = get_db()
    ok, pending_name, err = _check_otp(db, email, code)
    if not ok:
        record_login(db, "", email, "failed")
        return api_error(err[0], err[1], err[2])

    row = db.execute("SELECT * FROM users WHERE lower(email)=?", (email,)).fetchone()
    if not row:
        # First verified sign-in with this email -> create a regular account.
        # It's passwordless (email-code only), so password_hash gets a random
        # value nobody knows — same pattern as Google sign-in above.
        name = (data.get("name") or "").strip() or pending_name or email.split("@")[0]
        placeholder_pw = generate_password_hash(secrets.token_urlsafe(32))
        new_id = db_insert(
            db,
            "INSERT INTO users (name, email, password_hash, role, created_at) VALUES (?,?,?,?,?)",
            (name, email, placeholder_pw, "user", datetime.now(timezone.utc).isoformat()),
        )
        db.commit()
        row = db.execute("SELECT * FROM users WHERE id=?", (new_id,)).fetchone()
    record_login(db, row["name"], row["email"], "success")
    return jsonify({"user": user_public(row),
                    "token": make_token(row["id"], remember=remember),
                    "remember": remember})


# ---- Google sign-in — real, once GOOGLE_OAUTH_CLIENT_ID is configured ----
# The frontend uses Google Identity Services (the JS "Sign In With Google"
# button) and gets back a signed ID token straight in a JS callback — never
# a page redirect — so this endpoint verifies that token exactly the way
# every other write to this API is authenticated: a normal POST with a JSON
# body, from a page the user is already on. Verification steps, per
# Google's own docs: the token must be signed by Google (tokeninfo confirms
# this), "aud" must equal THIS app's client id (otherwise a token meant for
# a different app could be replayed here), "iss" must be Google, and the
# email must be verified — only then is the email trustworthy enough to
# find-or-create an account by.
@app.route("/api/auth/google", methods=["POST"])
def google_login():
    client_id = os.environ.get("GOOGLE_OAUTH_CLIENT_ID")
    if not client_id:
        return api_error("Sign-in with Google isn't configured yet",
                         "تسجيل الدخول عبر Google غير مُفعّل بعد — سيتم تفعيله قريبًا",
                         501, code="not_configured")
    data = request.get_json(force=True, silent=True) or {}
    id_token = (data.get("id_token") or data.get("credential") or "").strip()
    remember = bool(data.get("remember", True))
    if not id_token:
        return api_error("Missing Google credential", "بيانات اعتماد Google مفقودة", 400)

    # An unverified token is just base64 text — anyone could send a forged
    # payload claiming to be any email — so it's always checked against
    # Google before it's trusted. tokeninfo is Google's own documented
    # verification endpoint; fine at this app's scale (a few requests per
    # sign-in, not per page view). A high-traffic production app would
    # instead cache Google's public keys and verify the JWT signature
    # locally to skip the extra network round trip.
    try:
        gr = requests.get("https://oauth2.googleapis.com/tokeninfo",
                          params={"id_token": id_token}, timeout=10)
    except Exception as e:
        print(f"[auth] Google tokeninfo request failed: {e}")
        return api_error("Could not verify Google sign-in right now — try again",
                         "تعذّر التحقق من تسجيل الدخول عبر Google الآن — حاول مرة أخرى", 503)
    if not gr.ok:
        return api_error("Invalid or expired Google credential",
                         "بيانات اعتماد Google غير صالحة أو منتهية الصلاحية", 401)
    claims = gr.json()
    if claims.get("aud") != client_id:
        return api_error("This Google credential wasn't issued for this app",
                         "بيانات الاعتماد هذه غير صادرة لهذا التطبيق", 401)
    if claims.get("iss") not in ("accounts.google.com", "https://accounts.google.com"):
        return api_error("Invalid token issuer", "مُصدر الرمز غير صالح", 401)
    if str(claims.get("email_verified")).lower() != "true":
        return api_error("Google account email isn't verified",
                         "البريد الإلكتروني لحساب Google غير موثّق", 401)
    email = (claims.get("email") or "").strip().lower()
    if not email:
        return api_error("Google account has no email", "حساب Google بدون بريد إلكتروني", 401)
    name = (claims.get("name") or email.split("@")[0]).strip() or "Google User"

    db = get_db()
    # Find-or-create by verified email — an account created with a password
    # and one created via Google that share an email are the SAME identity,
    # exactly like /auth/login and /auth/signup already treat email as the
    # one unique key (see the UNIQUE constraint on users.email).
    row = db.execute("SELECT * FROM users WHERE lower(email)=?", (email,)).fetchone()
    if not row:
        # First sign-in with this email -> create a regular registered
        # account, exactly like /auth/signup. users.password_hash is
        # NOT NULL, so a random value nobody was ever told fills that
        # column — Google remains the only way into this specific account.
        placeholder_pw = generate_password_hash(secrets.token_urlsafe(32))
        new_id = db_insert(
            db,
            "INSERT INTO users (name, email, password_hash, role, created_at) VALUES (?,?,?,?,?)",
            (name, email, placeholder_pw, "user", datetime.now(timezone.utc).isoformat()),
        )
        db.commit()
        row = db.execute("SELECT * FROM users WHERE id=?", (new_id,)).fetchone()
    record_login(db, row["name"], row["email"], "success")
    return jsonify({"user": user_public(row), "token": make_token(row["id"], remember=remember),
                    "remember": remember})


@app.route("/api/auth/google-config")
def google_auth_config():
    """Public, unauthenticated — tells the frontend whether Google Sign-In
    is configured and, if so, the client id needed to initialize Google's
    JS library. An OAuth Client ID is not a secret: Google's own docs have
    it embedded directly in frontend JS on every integration guide. Trust
    still comes only from server-side verification in /api/auth/google
    above, never from the client id being present."""
    client_id = os.environ.get("GOOGLE_OAUTH_CLIENT_ID")
    return jsonify({"enabled": bool(client_id), "client_id": client_id})


@app.route("/api/auth/otp-config")
def otp_auth_config():
    """Public — tells the frontend whether email-code sign-in is available
    (i.e. whether SMTP is configured on the server)."""
    return jsonify({"enabled": smtp_configured()})


# ---- v15: Apple sign-in — PREPARED, not yet connected ----
# The spec asks the system to be READY to support this; without a real
# OAuth client id there is nothing safe to verify, so this endpoint is an
# honest stub: it tells the frontend clearly that sign-in isn't configured
# yet instead of pretending to authenticate anyone. Wiring up the real flow
# later follows the same shape as /api/auth/google above: verify the
# client's id_token against Apple's public keys using the configured
# client id, find-or-create the user by verified email, then return
# make_token(...) exactly like /auth/login.
@app.route("/api/auth/apple", methods=["POST"])
def apple_login():
    if not os.environ.get("APPLE_OAUTH_CLIENT_ID"):
        return api_error("Sign-in with Apple isn't configured yet",
                         "تسجيل الدخول عبر Apple غير مُفعّل بعد — سيتم تفعيله قريبًا",
                         501, code="not_configured")
    return api_error("Sign-in with Apple isn't configured yet",
                     "تسجيل الدخول عبر Apple غير مُفعّل بعد", 501, code="not_configured")




@app.route("/api/auth/guest", methods=["POST"])
def guest_login():
    """'Continue as Guest' — returns a guest identity with no token.

    Guests can browse the dashboard, comments, analytics and reports and can
    run the analyzer, but they carry no token, so POST /api/comments (write)
    and every admin-only endpoint reject them until they create an account.
    """
    return jsonify({
        "user": {"id": None, "name": "Guest", "email": None, "role": "guest", "guest": True},
        "token": None,
    })


@app.route("/api/auth/forgot-password", methods=["POST"])
def forgot_password():
    # NOTE: no email server is configured, so this only confirms whether the
    # flow ran — it does not actually send an email. Wire up an SMTP/email
    # provider (e.g. Flask-Mail) here for real password-reset emails.
    data = request.get_json(force=True, silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    db = get_db()
    row = db.execute("SELECT id FROM users WHERE lower(email)=?", (email,)).fetchone()
    return jsonify({
        "message": "If this email is registered, a reset link would be sent to it."
        if row else "If this email is registered, a reset link would be sent to it."
    })


# ---- Users management ---- #
@app.route("/api/users", methods=["GET"])
@admin_required
def list_users():
    db = get_db()
    rows = db.execute("SELECT * FROM users ORDER BY created_at ASC").fetchall()
    return jsonify([user_public(r) for r in rows])


@app.route("/api/users", methods=["POST"])
@admin_required
def add_user():
    data = request.get_json(force=True, silent=True) or {}
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    role = data.get("role") or "user"
    password = data.get("password") or "changeme123"
    if not name or not email:
        return api_error("name and email are required", "الاسم والبريد مطلوبان", 400)
    if role not in ASSIGNABLE_ROLES:
        return api_error("role must be 'user' or 'guest' — the admin role can never be assigned",
                         "الصلاحية يجب أن تكون 'مستخدم' أو 'ضيف' — صلاحية الأدمن لا تُمنح لأحد", 400)
    db = get_db()
    if db.execute("SELECT id FROM users WHERE lower(email)=?", (email,)).fetchone():
        return jsonify({"error": "An account with this email already exists"}), 409
    new_id = db_insert(
        db,
        "INSERT INTO users (name, email, password_hash, role, created_at) VALUES (?,?,?,?,?)",
        (name, email, generate_password_hash(password), role, datetime.now(timezone.utc).isoformat()),
    )
    db.commit()
    row = db.execute("SELECT * FROM users WHERE id=?", (new_id,)).fetchone()
    return jsonify(user_public(row)), 201


@app.route("/api/users/<int:user_id>", methods=["PUT"])
@admin_required
def update_user(user_id):
    data = request.get_json(force=True, silent=True) or {}
    db = get_db()
    row = db.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    if not row:
        return jsonify({"error": "not found"}), 404
    name = data.get("name", row["name"])
    email = (data.get("email") or row["email"]).strip().lower()
    role = data.get("role", row["role"])
    if is_fixed_admin(row):
        pass  # role checks for the fixed admin are below
    elif role not in ASSIGNABLE_ROLES:
        return api_error("role must be 'user' or 'guest' — the admin role can never be assigned",
                         "الصلاحية يجب أن تكون 'مستخدم' أو 'ضيف' — صلاحية الأدمن لا تُمنح لأحد", 400)
    if is_fixed_admin(row):
        # The fixed admin account cannot be demoted or have its email changed.
        if role != "admin":
            return jsonify({"error": "The primary admin account role cannot be changed"}), 403
        if email != ADMIN_EMAIL:
            return jsonify({"error": "The primary admin account email cannot be changed"}), 403
    db.execute("UPDATE users SET name=?, email=?, role=? WHERE id=?", (name, email, role, user_id))
    db.commit()
    row = db.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    return jsonify(user_public(row))


@app.route("/api/users/<int:user_id>", methods=["DELETE"])
@admin_required
def delete_user(user_id):
    db = get_db()
    row = db.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    if row and is_fixed_admin(row):
        return jsonify({"error": "The primary admin account cannot be deleted"}), 403
    db.execute("DELETE FROM users WHERE id=?", (user_id,))
    db.commit()
    return jsonify({"deleted": user_id})


# ---- Login logs (admin-only, v9) ---- #
@app.route("/api/login-logs")
@admin_required
def list_login_logs():
    """All sign-in attempts — searchable and sortable, admin-only.

    Query params:
        search   -> matches name, email or status
        sort     -> date_desc (default) | date_asc | name | email | status
        page / per_page -> pagination (default 15 per page)
    """
    db = get_db()
    search = request.args.get("search", "").strip()
    sort = request.args.get("sort", "date_desc")
    page = max(1, int(request.args.get("page", 1)))
    per_page = max(1, int(request.args.get("per_page", 15)))

    query = "SELECT * FROM login_logs WHERE 1=1"
    params = []
    if search:
        query += " AND (lower(name) LIKE lower(?) OR lower(email) LIKE lower(?) OR lower(status) LIKE lower(?))"
        like = f"%{search}%"
        params += [like, like, like]

    # Whitelisted ORDER BY only — never interpolate user input directly.
    order = {
        "date_desc": "created_at DESC",
        "date_asc": "created_at ASC",
        "name": "lower(name) ASC, created_at DESC",
        "email": "lower(email) ASC, created_at DESC",
        "status": "status ASC, created_at DESC",
    }.get(sort, "created_at DESC")

    rows = db.execute(query + " ORDER BY " + order, params).fetchall()
    total = len(rows)
    start = (page - 1) * per_page
    return jsonify({
        "total": total,
        "page": page,
        "per_page": per_page,
        "items": [dict(r) for r in rows[start:start + per_page]],
    })


@app.route("/api/dashboard/stats")
def dashboard_stats():
    db = get_db()
    rows = db.execute("SELECT * FROM comments").fetchall()
    total = len(rows)
    pos = sum(1 for r in rows if r["sentiment"] == "positive")
    neg = sum(1 for r in rows if r["sentiment"] == "negative")
    # "mixed" was removed as a sentiment label (v15.1). Rows analyzed before
    # this change may still carry the old "mixed" value — those rows are
    # NEVER modified or deleted, but for display they're counted as neutral
    # here so the three visible buckets (positive/negative/neutral) always
    # add up to the total.
    neu = total - pos - neg

    by_category = {}
    for r in rows:
        c = by_category.setdefault(r["category"],
                                   {"total": 0, "positive": 0, "negative": 0, "neutral": 0})
        c["total"] += 1
        label = r["sentiment"] if r["sentiment"] in ("positive", "negative", "neutral") else "neutral"
        c[label] = c.get(label, 0) + 1

    keyword_freq = {}
    for r in rows:
        if r["keywords"]:
            for kw in r["keywords"].split(","):
                kw = kw.strip()
                if kw:
                    keyword_freq[kw] = keyword_freq.get(kw, 0) + 1
    top_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:12]

    return jsonify({
        "total": total, "positive": pos, "negative": neg, "neutral": neu,
        "positive_pct": round(pos / total * 100, 1) if total else 0,
        "negative_pct": round(neg / total * 100, 1) if total else 0,
        "neutral_pct": round(neu / total * 100, 1) if total else 0,
        "by_category": by_category,
        "top_keywords": top_keywords,
    })


# ---------------------------------------------------------------- #
# v12: External comments — ingestion + hourly auto-refresh
# (fetching itself lives in external_sources.py, official APIs only)
# ---------------------------------------------------------------- #
_external_state = {"last_run": None, "last_result": None}  # status info only — data goes to the DB
_external_lock = threading.Lock()


def ingest_external_comments():
    """Fetch from every configured official API and store NEW comments only.

    De-duplication happens in two layers:
      1. Same-platform re-fetch: each platform item carries its own id
         (external_id) — a comment whose external_id already exists is
         skipped, so refreshing hourly never creates duplicates. For
         YouTube specifically, the set of already-known ids is also handed
         to the fetcher itself so it can stop paginating a video's comments
         as soon as it reaches ones it has already seen (see
         external_sources.fetch_youtube), instead of re-walking the whole
         thread on every run.
      2. v15 cross-source: the SAME opinion appearing on a different
         platform is caught by check_duplicate() (content fingerprint +
         fuzzy match) and skipped too, instead of inflating the counts.
    Every stored comment is analyzed by the AI pipeline and keeps its
    source + author + original timestamp (+ place/thread metadata when the
    source provides it). One failing source/video/channel never stops the
    others — each fetcher in external_sources.py already swallows its own
    per-item errors and returns whatever it managed to collect."""
    added = {"youtube": 0, "reddit": 0, "x": 0, "google_maps": 0}
    skipped = 0
    skipped_duplicate_cross_source = 0  # v15
    skipped_irrelevant = 0  # v14: dropped by the relevance filter
    # Own connection: this can run in the background thread, outside a request.
    conn = connect_db()
    now_iso = datetime.now(timezone.utc).isoformat()
    try:
        known_youtube_ids = {
            r["external_id"] for r in
            conn.execute("SELECT external_id FROM comments WHERE source='youtube' "
                        "AND external_id IS NOT NULL").fetchall()
        }
        youtube_videos = external_sources.fetch_youtube(known_ids=known_youtube_ids)
        # v15.1: record per-video ingestion state regardless of whether any
        # NEW comment was found this run — "checked but nothing new" still
        # updates last_checked_at, which is what the admin panel shows.
        for v in youtube_videos:
            new_count = len(v.get("comments") or [])
            existing = conn.execute(
                "SELECT comments_fetched FROM youtube_video_state WHERE video_id=?",
                (v["video_id"],)).fetchone()
            if existing:
                conn.execute(
                    "UPDATE youtube_video_state SET title=COALESCE(?, title), "
                    "channel_id=COALESCE(?, channel_id), last_checked_at=?, "
                    "comments_fetched=comments_fetched+?, last_run_added=? WHERE video_id=?",
                    (v.get("title"), v.get("channel_id"), now_iso, new_count, new_count, v["video_id"]))
            else:
                conn.execute(
                    "INSERT INTO youtube_video_state (video_id, channel_id, title, "
                    "first_seen_at, last_checked_at, comments_fetched, last_run_added) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (v["video_id"], v.get("channel_id"), v.get("title"), now_iso, now_iso,
                     new_count, new_count))
        items = (external_sources.flatten_youtube_comments(youtube_videos)
                + external_sources.fetch_reddit() + external_sources.fetch_x()
                + google_maps_source.fetch_reviews())
        for it in items:
            if not it.get("external_id") or not it.get("text"):
                continue
            dup = conn.execute("SELECT id FROM comments WHERE external_id=?",
                               (it["external_id"],)).fetchone()
            if dup:
                skipped += 1
                continue
            place_type = it.get("place_type")
            result = run_sentiment_analysis(it["text"], is_external=True, place_type=place_type)
            # v14 relevance filter: external comments that are NOT about
            # Hajj/Umrah are dropped BEFORE entering the database.
            if result["moderation_status"] == "rejected":
                skipped_irrelevant += 1
                continue
            # v15: cross-source de-duplication (same opinion, different platform).
            dup_id = check_duplicate(conn, result.get("content_fingerprint"),
                                     result.get("text_ar") or it["text"])
            if dup_id:
                skipped_duplicate_cross_source += 1
                continue
            source_meta = {k: it[k] for k in
                           ("place_type", "place_url", "title", "community",
                            "votes", "num_comments", "permalink", "language")
                           if it.get(k) is not None}
            conn.execute(
                "INSERT INTO comments (text, category, sentiment, confidence, keywords, "
                "created_at, user_id, user_name, rating, status, likes, source, external_id, "
                "original_lang, text_ar, moderation_status, moderation_flags, moderation_reason, "
                "content_fingerprint, place_name, place_country, place_city, source_meta) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (it["text"], result["category"], result["label"], result["confidence"],
                 ",".join(result["keywords"]), it["created_at"],
                 None, it["author"], it.get("rating"), "approved", 0, it["source"], it["external_id"],
                 result.get("original_lang"), result.get("text_ar"),
                 result["moderation_status"], result["moderation_flags"], result["moderation_reason"],
                 result.get("content_fingerprint"), it.get("place_name"), it.get("place_country"),
                 it.get("place_city"), json.dumps(source_meta, ensure_ascii=False) if source_meta else None),
            )
            added[it["source"]] = added.get(it["source"], 0) + 1
        conn.commit()
    finally:
        conn.close()
    info = {"added": added, "skipped_duplicates": skipped,
            "skipped_duplicate_cross_source": skipped_duplicate_cross_source,
            "skipped_irrelevant": skipped_irrelevant,
            "fetched": len(items), "at": datetime.now(timezone.utc).isoformat()}
    _external_state["last_run"] = info["at"]
    _external_state["last_result"] = info
    return info


def _auto_fetch_loop():
    """Background refresh every FETCH_INTERVAL_MINUTES (default 60 = hourly)."""
    interval = max(5, int(os.environ.get("FETCH_INTERVAL_MINUTES", "60"))) * 60
    while True:
        try:
            with _external_lock:
                ingest_external_comments()
        except Exception as e:  # the loop must survive anything
            print(f"[external] auto fetch failed: {e}")
        time.sleep(interval)


def start_auto_fetch():
    """Start the hourly refresh thread once — only when at least one source
    is configured and AUTO_FETCH_EXTERNAL isn't disabled."""
    if os.environ.get("AUTO_FETCH_EXTERNAL", "1") == "0":
        return
    if not any(external_sources.configured_sources().values()) and not google_maps_source.configured():
        return  # nothing configured — no thread needed
    t = threading.Thread(target=_auto_fetch_loop, daemon=True, name="external-fetch")
    t.start()
    print("[external] hourly auto-fetch started")


start_auto_fetch()


@app.route("/api/admin/fetch-external", methods=["POST"])
@admin_required
def admin_fetch_external():
    """Admin: fetch external comments NOW (in addition to the hourly refresh)."""
    with _external_lock:
        info = ingest_external_comments()
    return jsonify(info)


@app.route("/api/admin/reanalyze", methods=["POST"])
@admin_required
def admin_reanalyze():
    """v13: re-run the NEW translation + sentiment pipeline over existing
    comments — fixes wrong old classifications and fills in the Arabic
    translation + detected language for comments stored before v13.

    Body (optional): {"only_missing": true} -> only comments that don't have
    a detected language yet (default). {"only_missing": false} -> re-analyze
    EVERYTHING with the current engine. Batched with a per-request cap so the
    request stays fast; call repeatedly until "remaining" is 0."""
    data = request.get_json(force=True, silent=True) or {}
    only_missing = data.get("only_missing", True)
    limit = max(1, min(int(data.get("limit", 50)), 200))
    db = get_db()
    if only_missing:
        rows = db.execute(
            "SELECT id, text FROM comments WHERE original_lang IS NULL "
            "ORDER BY id LIMIT ?", (limit,)).fetchall()
        remaining_q = "SELECT COUNT(*) AS c FROM comments WHERE original_lang IS NULL"
    else:
        offset = max(0, int(data.get("offset", 0)))
        rows = db.execute("SELECT id, text FROM comments ORDER BY id LIMIT ? OFFSET ?",
                          (limit, offset)).fetchall()
        remaining_q = None
    updated = 0
    for r in rows:
        try:
            res = run_sentiment_analysis(r["text"])
            db.execute(
                "UPDATE comments SET sentiment=?, confidence=?, keywords=?, "
                "original_lang=?, text_ar=?, moderation_status=?, "
                "moderation_flags=?, moderation_reason=? WHERE id=?",
                (res["label"], res["confidence"], ",".join(res["keywords"]),
                 res.get("original_lang") or "unknown", res.get("text_ar"),
                 res["moderation_status"], res["moderation_flags"],
                 res["moderation_reason"], r["id"]),
            )
            updated += 1
        except Exception as e:  # one bad comment must not stop the batch
            print(f"[reanalyze] comment {r['id']} failed: {e}")
    db.commit()
    remaining = (db.execute(remaining_q).fetchone()["c"] if remaining_q else None)
    return jsonify({"updated": updated, "remaining": remaining,
                    "engine": sentiment.engine_status()["active_engine"]})


@app.route("/api/admin/analysis-status")
@admin_required
def admin_analysis_status():
    """v13: which sentiment tier is active + translation coverage counts."""
    db = get_db()
    total = db.execute("SELECT COUNT(*) AS c FROM comments").fetchone()["c"]
    translated = db.execute(
        "SELECT COUNT(*) AS c FROM comments WHERE text_ar IS NOT NULL").fetchone()["c"]
    pending = db.execute(
        "SELECT COUNT(*) AS c FROM comments WHERE original_lang IS NULL").fetchone()["c"]
    flagged = db.execute("SELECT COUNT(*) AS c FROM comments WHERE moderation_status='flagged'").fetchone()["c"]
    rejected = db.execute("SELECT COUNT(*) AS c FROM comments WHERE moderation_status='rejected'").fetchone()["c"]
    return jsonify({**ai_pipeline.pipeline_status(),
                    "total_comments": total,
                    "with_arabic_translation": translated,
                    "not_yet_processed": pending,
                    "flagged_comments": flagged,
                    "rejected_irrelevant_comments": rejected})


@app.route("/api/admin/external-status")
@admin_required
def admin_external_status():
    """Admin: which sources are configured, last refresh info, per-source
    counts, and per-video YouTube ingestion state (v15.1) — when each
    followed/discovered video was last checked and how many comments have
    been collected from it so far, newest-checked first."""
    db = get_db()
    counts = {}
    for row in db.execute("SELECT source, COUNT(*) AS c FROM comments GROUP BY source").fetchall():
        counts[row["source"]] = row["c"]
    configured = external_sources.configured_sources()
    configured["google_maps"] = google_maps_source.configured()  # v15
    youtube_videos = [dict(r) for r in db.execute(
        "SELECT video_id, channel_id, title, first_seen_at, last_checked_at, "
        "comments_fetched, last_run_added FROM youtube_video_state "
        "ORDER BY last_checked_at DESC LIMIT 50"
    ).fetchall()]
    return jsonify({
        "configured": configured,
        "auto_fetch_enabled": os.environ.get("AUTO_FETCH_EXTERNAL", "1") != "0",
        "interval_minutes": max(5, int(os.environ.get("FETCH_INTERVAL_MINUTES", "60"))),
        "followed_youtube_channels": external_sources.youtube_channel_ids(),
        "last_run": _external_state["last_run"],
        "last_result": _external_state["last_result"],
        "counts_by_source": counts,
        "youtube_videos": youtube_videos,
    })


# ---------------------------------------------------------------- #
# v15 — "تعليمات الحج والعمرة": the specialized Hajj/Umrah AI assistant
# ---------------------------------------------------------------- #
@app.route("/api/assistant/chat", methods=["POST"])
def assistant_chat():
    """Body: {"messages": [{"role": "user"|"assistant", "content": "..."}, ...], "lang": "ar"}
    The full conversation is sent by the client each time (stateless,
    same pattern as the Anthropic Messages API) — nothing is persisted
    server-side, matching "الاحتفاظ بسياق المحادثة أثناء الجلسة" (context
    kept for the session only). Open to guests too: the assistant is a
    reference tool, not a write action."""
    data = request.get_json(force=True, silent=True) or {}
    messages = data.get("messages") or []
    if not isinstance(messages, list) or not messages:
        return api_error("messages is required", "الرسائل مطلوبة", 400)
    lang = (data.get("lang") or "ar").strip().lower()
    if lang not in UI_LANGS:
        lang = "ar"
    result = assistant.answer(messages, lang=lang)
    return jsonify(result)


@app.route("/api/assistant/status")
def assistant_status():
    return jsonify(assistant.status())


@app.route("/api/assistant/diagnostics")
@admin_required
def assistant_diagnostics():
    """v15.6 (admin): run a LIVE test of the configured AI provider and report
    the true result (ok, or the real error — bad key, model not found, quota,
    network). Lets the admin see exactly why the assistant isn't answering
    instead of guessing. Never raises."""
    lang = (request.args.get("lang") or "ar").strip().lower()
    if lang not in UI_LANGS:
        lang = "ar"
    try:
        return jsonify(assistant.diagnostics(lang=lang))
    except Exception as e:  # defensive — diagnostics() shouldn't raise
        return jsonify({"ok": False, "detail": f"diagnostics failed: {e}"}), 200


# ---------------------------------------------------------------- #
# v15.5 — YouTube: analyze the comments of ONE specific video by URL.
# Fetches ~150 REAL comments (or all available if fewer), runs each through
# the same AI pipeline used everywhere else, returns stats + the analyzed
# comments (shown exactly like the Google Maps analysis), and also stores
# the approved/relevant ones so they feed the dashboard & analytics too.
# No comments are ever generated — only real user comments are analyzed.
# ---------------------------------------------------------------- #
@app.route("/api/youtube/analyze", methods=["POST"])
@admin_required
def youtube_analyze():
    data = request.get_json(force=True, silent=True) or {}
    url = (data.get("url") or data.get("video") or "").strip()
    if not url:
        return api_error("Enter a YouTube video link", "أدخل رابط فيديو YouTube", 400)
    try:
        max_comments = int(data.get("max_comments", 150))
    except (TypeError, ValueError):
        max_comments = 150
    max_comments = max(1, min(max_comments, 200))

    try:
        result = external_sources.fetch_video_comments(url, max_comments=max_comments)
    except external_sources.YouTubeFetchError as e:
        return api_error(e.message_en, e.message_ar, e.status)

    fetched = result["comments"]
    if not fetched:
        return api_error("No comments available for this video",
                         "لا توجد تعليقات متاحة لهذا الفيديو", 404)

    db = get_db()
    analyzed = []
    counts = {"positive": 0, "negative": 0, "neutral": 0}
    added = 0
    skipped_existing = 0
    skipped_irrelevant = 0
    skipped_duplicate = 0

    for it in fetched:
        res = run_sentiment_analysis(it["text"], is_external=True)
        label = res["label"] if res["label"] in counts else "neutral"
        counts[label] += 1
        # Return every analyzed comment as written, with its classification,
        # so the UI can show the full breakdown (positive/negative/neutral).
        analyzed.append({
            "text": it["text"],
            "author": it["author"],
            "likes": it.get("likes", 0),
            "created_at": it["created_at"],
            "sentiment": label,
            "confidence": res["confidence"],
            "category": res["category"],
            "relevant": res["moderation_status"] != "rejected",
        })

        # Persist only approved + relevant + non-duplicate ones into the
        # public comments table (same rules as the Google Maps import).
        dup = db.execute("SELECT id FROM comments WHERE external_id=?",
                         (it["external_id"],)).fetchone()
        if dup:
            skipped_existing += 1
            continue
        if res["moderation_status"] == "rejected":
            skipped_irrelevant += 1
            continue
        dup_id = check_duplicate(db, res.get("content_fingerprint"),
                                 res.get("text_ar") or it["text"])
        if dup_id:
            skipped_duplicate += 1
            continue
        source_meta = {"video_id": result["video_id"]}
        db.execute(
            "INSERT INTO comments (text, category, sentiment, confidence, keywords, "
            "created_at, user_id, user_name, rating, status, likes, source, external_id, "
            "original_lang, text_ar, moderation_status, moderation_flags, moderation_reason, "
            "content_fingerprint, place_name, place_country, place_city, source_meta) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (it["text"], res["category"], res["label"], res["confidence"],
             ",".join(res["keywords"]), it["created_at"],
             None, it["author"], None, "approved", int(it.get("likes", 0) or 0),
             "youtube", it["external_id"],
             res.get("original_lang"), res.get("text_ar"),
             res["moderation_status"], res["moderation_flags"], res["moderation_reason"],
             res.get("content_fingerprint"), None, None, None,
             json.dumps(source_meta, ensure_ascii=False)),
        )
        added += 1
    db.commit()

    total = len(analyzed)
    percentages = {k: round(v * 100.0 / total, 1) for k, v in counts.items()} if total else counts
    return jsonify({
        "video_id": result["video_id"],
        "analyzed": total,
        "counts": counts,
        "percentages": percentages,
        "comments": analyzed,
        "stored": {"added": added, "skipped_existing": skipped_existing,
                   "skipped_irrelevant": skipped_irrelevant,
                   "skipped_duplicate": skipped_duplicate},
    })


# ---------------------------------------------------------------- #
# v15.6 — X (Twitter) & Reddit analytics: search the Hajj/Umrah keywords
# on-demand, run each post/comment through the SAME AI pipeline used
# everywhere else, return stats + the analyzed items (shown exactly like
# the YouTube analysis page), and store the approved/relevant/non-duplicate
# ones so they feed the dashboard & analytics too. Real posts/comments
# only — nothing is ever generated. Both pages work like YouTube analysis
# but search by keyword instead of taking a single URL.
# ---------------------------------------------------------------- #
def _analyze_and_store_items(items, source):
    """Shared engine for the X & Reddit analysis pages. `items` is the list
    returned by external_sources.search_x_posts / search_reddit_posts. Runs
    sentiment on every item, returns the analyzed list + counts + storage
    stats, and persists approved + relevant + non-duplicate items into the
    public comments table with the given `source`. Never raises for a single
    bad item — one failure must not abort the whole batch."""
    db = get_db()
    analyzed = []
    counts = {"positive": 0, "negative": 0, "neutral": 0}
    added = skipped_existing = skipped_irrelevant = skipped_duplicate = 0

    for it in items:
        try:
            res = run_sentiment_analysis(it["text"], is_external=True)
        except Exception as e:  # keep going — never crash the page on one item
            print(f"[{source}-analyze] item failed, skipped: {e}")
            continue
        label = res["label"] if res["label"] in counts else "neutral"
        counts[label] += 1
        analyzed.append({
            "text": it["text"],
            "author": it.get("author"),
            "likes": it.get("likes", 0),
            "created_at": it.get("created_at"),
            "sentiment": label,
            "confidence": res["confidence"],
            "category": res["category"],
            "kind": it.get("kind", "post"),
            "url": it.get("url"),
            "community": it.get("community"),
            "relevant": res["moderation_status"] != "rejected",
        })

        ext_id = it.get("external_id")
        if ext_id:
            dup = db.execute("SELECT id FROM comments WHERE external_id=?", (ext_id,)).fetchone()
            if dup:
                skipped_existing += 1
                continue
        if res["moderation_status"] == "rejected":
            skipped_irrelevant += 1
            continue
        dup_id = check_duplicate(db, res.get("content_fingerprint"),
                                 res.get("text_ar") or it["text"])
        if dup_id:
            skipped_duplicate += 1
            continue
        source_meta = {}
        for k in ("url", "kind", "title", "community"):
            if it.get(k) is not None:
                source_meta[k] = it[k]
        try:
            db.execute(
                "INSERT INTO comments (text, category, sentiment, confidence, keywords, "
                "created_at, user_id, user_name, rating, status, likes, source, external_id, "
                "original_lang, text_ar, moderation_status, moderation_flags, moderation_reason, "
                "content_fingerprint, place_name, place_country, place_city, source_meta) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (it["text"], res["category"], res["label"], res["confidence"],
                 ",".join(res["keywords"]), it.get("created_at"),
                 None, it.get("author"), None, "approved", int(it.get("likes", 0) or 0),
                 source, ext_id,
                 res.get("original_lang"), res.get("text_ar"),
                 res["moderation_status"], res["moderation_flags"], res["moderation_reason"],
                 res.get("content_fingerprint"), None, None, None,
                 json.dumps(source_meta, ensure_ascii=False) if source_meta else None),
            )
            added += 1
        except Exception as e:  # a DB hiccup on one row must not lose the rest
            print(f"[{source}-analyze] insert failed, skipped: {e}")
    db.commit()

    total = len(analyzed)
    percentages = {k: round(v * 100.0 / total, 1) for k, v in counts.items()} if total else counts
    return {
        "analyzed": total,
        "counts": counts,
        "percentages": percentages,
        "comments": analyzed,
        "stored": {"added": added, "skipped_existing": skipped_existing,
                   "skipped_irrelevant": skipped_irrelevant,
                   "skipped_duplicate": skipped_duplicate},
    }


@app.route("/api/x/analyze", methods=["POST"])
@admin_required
def x_analyze():
    """Search X for Hajj/Umrah posts, analyze their sentiment, store the
    relevant ones. Body (optional): {"max_results": 50}. Clear bilingual
    error if X isn't configured or the API rejects the request — the page
    shows the reason and the site keeps working."""
    data = request.get_json(force=True, silent=True) or {}
    try:
        max_results = int(data.get("max_results", 50))
    except (TypeError, ValueError):
        max_results = 50
    try:
        result = external_sources.search_x_posts(max_results=max_results)
    except external_sources.XFetchError as e:
        return api_error(e.message_en, e.message_ar, e.status)
    except Exception as e:  # unexpected — never 500 the page
        print(f"[x-analyze] unexpected error: {e}")
        return api_error("Unexpected error fetching from X",
                         "خطأ غير متوقع أثناء الجلب من X", 502)

    items = result.get("items") or []
    if not items:
        return api_error("No X posts found for Hajj/Umrah right now",
                         "لا توجد منشورات على X عن الحج/العمرة حاليًا", 404)
    payload = _analyze_and_store_items(items, "x")
    payload["query"] = result.get("query")
    return jsonify(payload)


@app.route("/api/reddit/analyze", methods=["POST"])
@admin_required
def reddit_analyze():
    """Search Reddit for Hajj/Umrah posts + their top comments, analyze the
    sentiment, store the relevant ones. Body (optional): {"max_posts": 20,
    "max_comments_per_post": 8}. Clear bilingual error if Reddit isn't
    configured or the API rejects the request."""
    data = request.get_json(force=True, silent=True) or {}
    try:
        max_posts = int(data.get("max_posts", 20))
    except (TypeError, ValueError):
        max_posts = 20
    try:
        max_comments = int(data.get("max_comments_per_post", 8))
    except (TypeError, ValueError):
        max_comments = 8
    try:
        result = external_sources.search_reddit_posts(
            max_posts=max_posts, max_comments_per_post=max_comments)
    except external_sources.RedditFetchError as e:
        return api_error(e.message_en, e.message_ar, e.status)
    except Exception as e:
        print(f"[reddit-analyze] unexpected error: {e}")
        return api_error("Unexpected error fetching from Reddit",
                         "خطأ غير متوقع أثناء الجلب من Reddit", 502)

    items = result.get("items") or []
    if not items:
        return api_error("No Reddit posts found for Hajj/Umrah right now",
                         "لا توجد منشورات على Reddit عن الحج/العمرة حاليًا", 404)
    payload = _analyze_and_store_items(items, "reddit")
    payload["query"] = result.get("query")
    return jsonify(payload)


# ---------------------------------------------------------------- #
# v15 — Google Maps Reviews: manual/admin structured import
# (the automatic path is ingest_external_comments() -> google_maps_source)
# ---------------------------------------------------------------- #
@app.route("/api/admin/import-reviews", methods=["POST"])
@admin_required
def admin_import_reviews():
    """Admin: bulk-import reviews from a JSON list (Google Maps export,
    spreadsheet, or any tool) — normalized, relevance-filtered, analyzed
    and de-duplicated exactly like the automatic sources. Body:
    {"reviews": [{...}, ...], "default_place": {"place_name": "...", ...}}
    Field names are flexible — see google_maps_source._ALIASES."""
    data = request.get_json(force=True, silent=True) or {}
    raw_list = data.get("reviews")
    if not isinstance(raw_list, list) or not raw_list:
        return api_error("reviews must be a non-empty list", "reviews يجب أن تكون قائمة غير فارغة", 400)
    if len(raw_list) > 2000:
        return api_error("Max 2000 reviews per import — split into batches",
                         "الحد الأقصى 2000 مراجعة لكل استيراد — قسّمها إلى دفعات", 400)
    default_place = data.get("default_place") if isinstance(data.get("default_place"), dict) else None
    items = google_maps_source.normalize_batch(raw_list, default_place=default_place)

    added = 0
    skipped_existing = 0
    skipped_duplicate_cross_source = 0
    skipped_irrelevant = 0
    db = get_db()
    for it in items:
        if it.get("external_id"):
            dup = db.execute("SELECT id FROM comments WHERE external_id=?", (it["external_id"],)).fetchone()
            if dup:
                skipped_existing += 1
                continue
        result = run_sentiment_analysis(it["text"], is_external=True, place_type=it.get("place_type"))
        if result["moderation_status"] == "rejected":
            skipped_irrelevant += 1
            continue
        dup_id = check_duplicate(db, result.get("content_fingerprint"), result.get("text_ar") or it["text"])
        if dup_id:
            skipped_duplicate_cross_source += 1
            continue
        source_meta = {"place_type": it.get("place_type")} if it.get("place_type") else None
        db.execute(
            "INSERT INTO comments (text, category, sentiment, confidence, keywords, "
            "created_at, user_id, user_name, rating, status, likes, source, external_id, "
            "original_lang, text_ar, moderation_status, moderation_flags, moderation_reason, "
            "content_fingerprint, place_name, place_country, place_city, source_meta) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (it["text"], result["category"], result["label"], result["confidence"],
             ",".join(result["keywords"]), it["created_at"],
             None, it["author"], it.get("rating"), "approved", 0, it["source"], it.get("external_id"),
             result.get("original_lang"), result.get("text_ar"),
             result["moderation_status"], result["moderation_flags"], result["moderation_reason"],
             result.get("content_fingerprint"), it.get("place_name"), it.get("place_country"),
             it.get("place_city"), json.dumps(source_meta, ensure_ascii=False) if source_meta else None),
        )
        added += 1
    db.commit()
    return jsonify({"received": len(raw_list), "normalized": len(items), "added": added,
                    "skipped_existing": skipped_existing,
                    "skipped_duplicate_cross_source": skipped_duplicate_cross_source,
                    "skipped_irrelevant": skipped_irrelevant})


@app.route("/api/admin/import-reviews/schema")
@admin_required
def admin_import_reviews_schema():
    """Documents the expected JSON shape for POST /api/admin/import-reviews
    — handy for whoever wires up the real Google Maps export later."""
    return jsonify({
        "reviews_fields": ["place_name", "place_type", "country", "city", "place_url",
                          "rating", "review_date", "language", "text", "username", "review_id"],
        "default_place_fields": ["place_name", "place_type", "country", "city", "place_url"],
        "notes": "Field names are flexible (aliases like reviewText/comment/body also work "
                 "for 'text', author/user for 'username', etc.) — see google_maps_source._ALIASES.",
    })


# ---------------------------------------------------------------- #
# v15 — extended analytics: categories, sources, geography, trend
# ---------------------------------------------------------------- #
@app.route("/api/analytics/overview")
@admin_required
def analytics_overview():
    """Everything the v15 Analytics dashboard needs in one call: sentiment,
    category distribution, source comparison, country/city comparison (from
    Google Maps data), a daily sentiment trend for the last 30 days, and a
    light "suggestions" extraction (comments that read like a suggestion
    rather than a plain complaint/praise)."""
    db = get_db()
    rows = [dict(r) for r in db.execute(
        "SELECT sentiment, category, source, place_country, place_city, created_at, text, keywords "
        "FROM comments WHERE status='approved' AND COALESCE(moderation_status,'approved')='approved'"
    ).fetchall()]

    # "mixed" was removed as a sentiment label (v15.1) — new comments are
    # always positive/negative/neutral. Any older row still carrying the
    # legacy "mixed" value is never modified, but is folded into "neutral"
    # here so every breakdown below only ever shows the current 3 labels.
    def _label(r):
        return r["sentiment"] if r["sentiment"] in ("positive", "negative", "neutral") else "neutral"

    by_category = {}
    by_source = {}
    by_country = {}
    by_city = {}
    for r in rows:
        lbl = _label(r)
        for bucket, key in ((by_category, r["category"] or "general"),
                            (by_source, r["source"] or "user")):
            c = bucket.setdefault(key, {"total": 0, "positive": 0, "negative": 0, "neutral": 0})
            c["total"] += 1
            c[lbl] += 1
        if r["place_country"]:
            c = by_country.setdefault(r["place_country"], {"total": 0, "positive": 0, "negative": 0, "neutral": 0})
            c["total"] += 1
            c[lbl] += 1
        if r["place_city"]:
            c = by_city.setdefault(r["place_city"], {"total": 0, "positive": 0, "negative": 0, "neutral": 0})
            c["total"] += 1
            c[lbl] += 1

    # Daily sentiment trend, last 30 days (created_at is an ISO string, so a
    # plain string slice gives the date part on both SQLite and Postgres data).
    trend = {}
    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).date().isoformat()
    for r in rows:
        day = (r["created_at"] or "")[:10]
        if not day or day < cutoff:
            continue
        d = trend.setdefault(day, {"positive": 0, "negative": 0, "neutral": 0, "total": 0})
        d["total"] += 1
        d[_label(r)] += 1
    trend_series = [{"date": day, **counts} for day, counts in sorted(trend.items())]

    # Lightweight "suggestion" detection — comments that read like a
    # recommendation rather than a plain complaint or compliment.
    suggestion_markers = ("أقترح", "اقتراح", "أتمنى", "يفضل أن", "ياريت", "لو تم", "من الأفضل",
                          "suggest", "recommend", "it would be better", "should", "could improve",
                          "wish", "please add", "please improve")
    suggestions = []
    for r in rows:
        hay = (r["text"] or "").lower()
        if any(m in hay for m in suggestion_markers):
            suggestions.append(r["text"][:200])
    suggestions = suggestions[:20]

    total = len(rows)
    counts = {"positive": 0, "negative": 0, "neutral": 0}
    for r in rows:
        counts[_label(r)] += 1

    return jsonify({
        "total": total,
        "counts": counts,
        "by_category": by_category,
        "category_labels_ar": ai_pipeline.CATEGORY_LABELS_AR,
        "category_labels_en": ai_pipeline.CATEGORY_LABELS_EN,
        "by_source": by_source,
        "by_country": dict(sorted(by_country.items(), key=lambda kv: kv[1]["total"], reverse=True)[:15]),
        "by_city": dict(sorted(by_city.items(), key=lambda kv: kv[1]["total"], reverse=True)[:15]),
        "trend": trend_series,
        "suggestions": suggestions,
    })


if __name__ == "__main__":
    # init_db() already ran at import time above (needed for Gunicorn/Render);
    # calling it again here is harmless — everything it does is idempotent.
    init_db()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
