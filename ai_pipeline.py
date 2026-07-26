# -*- coding: utf-8 -*-
"""Unified AI processing pipeline (v14, extended in v15).

EVERY comment — site reviews and live comments fetched from YouTube / X /
Reddit / any future source — passes through process():

    fetch -> detect language -> translate -> sentiment -> moderation
          -> relevance -> save -> display approved only

The pipeline is tiered so it always works, and gets MORE accurate as the
site owner adds his own API keys on Render (plain env vars — no platform
lock-in, no paid connectors):

TIER A — LLM analysis (highest accuracy; recommended for production):
    Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY. ONE call per
    comment returns language, sentiment (positive/negative/neutral — with
    real understanding of context and sarcasm, always picking the single
    dominant sentiment), a topic category (see CATEGORIES below),
    moderation categories (profanity, insults, hate speech, harassment,
    racism, sexual content, violence, spam) and Hajj/Umrah relevance — all
    as strict JSON. Models: claude-haiku / gpt-4o-mini / gemini-flash class
    (fast + cheap, cents per thousand comments). If more than one key is
    set, priority is Anthropic -> OpenAI -> Gemini (see llm_configured()).

TIER B — specialized models / rules (no LLM key):
    * sentiment: sentiment.py (HF multilingual transformer if HF_API_TOKEN,
      else VADER / enhanced built-in on the English translation) — always
      one of positive/negative/neutral.
    * category: keyword-scored classification into the CATEGORIES taxonomy
      (v15) — same idea as the relevance heuristic below.
    * moderation: HF toxicity transformer if HF_API_TOKEN
      (unitary/multilingual-toxic-xlm-roberta) + local wordlists (ar+en)
      and spam heuristics — local checks always run as a safety net
    * relevance: topic heuristics on the original + English translation
      (shared list in knowledge_base.TOPIC_WORDS). Applied to EXTERNAL
      comments only (they come from broad searches); reviews written ON the
      site are presumed on-topic in this tier — only the LLM tier is
      precise enough to reject user reviews safely. v15: a Google Maps
      review whose place_type is clearly pilgrim-related (hotel for
      pilgrims, transport/Hajj/Umrah company or campaign, the Grand Mosque,
      the Prophet's Mosque, crowd management, government pilgrim services)
      is treated as relevant even if the free-text heuristic misses it.

v15 also adds a content fingerprint (dedup.py) so the SAME opinion posted on
multiple platforms (Google Maps / X / YouTube / Reddit) can be recognized
as a duplicate instead of inflating the counts — see fingerprint_for().

Translation always uses translation.py (free, all languages). Failures at
any stage degrade gracefully — a comment is never lost to an AI error.
"""
import json
import os
import re

import requests

import translation
import sentiment
import dedup
import knowledge_base

LLM_TIMEOUT = 25
MODERATION_FLAG_KEYS = ["profanity", "insult", "hate_speech", "harassment",
                        "racism", "sexual", "violence", "spam"]
# v15.1: "mixed" removed per product decision — every comment is classified
# into exactly one of these three labels now (see _LLM_PROMPT and
# _maybe_mixed below, which used to promote borderline cases to "mixed").
SENTIMENT_LABELS = ["positive", "negative", "neutral"]

# v15.1 fix: "claude-haiku-4-5" (no date suffix) is not a valid Anthropic
# model id — requests using it 404, get caught by the except below, and the
# pipeline silently falls back to the non-LLM tier on every single comment,
# even with a valid API key configured. Overridable via LLM_MODEL.
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
# v15.3: pinned to the stable GA model "gemini-2.5-flash". The
# "gemini-flash-latest" alias can resolve to a 3.x model that rejects
# thinkingConfig.thinkingBudget=0 (HTTP 400) — which silently knocked EVERY
# comment down to the non-LLM tier (VADER on the English translation), the
# exact cause of the Arabic mis-classifications the user reported (e.g.
# "ممتاز" landing as neutral). gemini-2.5-flash accepts the thinking-off
# config; _llm_call() also retries WITHOUT it, so a 3.x model set via
# LLM_MODEL still works. Overridable via LLM_MODEL.
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

# Moderation is the whole point of this tier, so the four adjustable Gemini
# safety categories are opened up explicitly instead of left on Google's
# default: the model has to actually SEE profanity/hate/violence/sexual text
# to classify it. A silently-blocked candidate would just look like another
# failed LLM call and fall back to Tier B, defeating the point of this tier.
GEMINI_SAFETY_SETTINGS = [
    {"category": c, "threshold": "BLOCK_NONE"} for c in (
        "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT",
    )
]

# ------------------------------------------------------------------ #
# v15 — smart classification into the platform's topic taxonomy.
# Internal codes are stable (used for filtering/analytics); display labels
# are translated in the frontend. "general" is the catch-all fallback and
# is always a valid choice, matching legacy data.
# ------------------------------------------------------------------ #
CATEGORIES = [
    "customer_service", "service_quality", "transportation", "accommodation",
    "cleanliness", "crowd_management", "accessibility", "haram_experience",
    "nabawi_experience", "hajj_experience", "umrah_experience", "general",
]
CATEGORY_LABELS_AR = {
    "customer_service": "التعامل وخدمة العملاء", "service_quality": "جودة الخدمات",
    "transportation": "النقل والمواصلات", "accommodation": "السكن والفنادق",
    "cleanliness": "النظافة", "crowd_management": "التنظيم وإدارة الحشود",
    "accessibility": "سهولة الوصول", "haram_experience": "تجربة الحرم المكي",
    "nabawi_experience": "تجربة المسجد النبوي", "hajj_experience": "تجربة الحج",
    "umrah_experience": "تجربة العمرة", "general": "عام",
}
CATEGORY_LABELS_EN = {
    "customer_service": "Customer Service", "service_quality": "Service Quality",
    "transportation": "Transportation", "accommodation": "Accommodation & Hotels",
    "cleanliness": "Cleanliness", "crowd_management": "Crowd Management",
    "accessibility": "Accessibility", "haram_experience": "Grand Mosque Experience",
    "nabawi_experience": "Prophet's Mosque Experience", "hajj_experience": "Hajj Experience",
    "umrah_experience": "Umrah Experience", "general": "General",
}
_CATEGORY_KEYWORDS = {
    "customer_service": {"staff", "employee", "employees", "customer service", "service desk",
                          "rude", "friendly", "helpful", "unprofessional", "welcoming", "hospitality",
                          "attitude", "ignored", "reception desk",
                          "موظف", "موظفين", "الموظفين", "التعامل", "أسلوب التعامل", "خدمة العملاء",
                          "استقبال", "مهذب", "متعاون", "متعاونين", "ترحيب", "ضيافة", "احترام",
                          "قلة احترام", "وقح", "وقحين", "تجاهل", "تجاهلونا", "اهتمام"},
    "service_quality": {"quality", "service", "professional", "standard", "level of service",
                         "excellent service", "poor service",
                         "جودة", "الخدمة", "الخدمات", "احترافي", "احترافية", "مستوى", "مستوى الخدمة",
                         "جودة عالية", "جودة رديئة", "إتقان", "خدمة ممتازة", "خدمة سيئة"},
    "transportation": {"bus", "transport", "transportation", "taxi", "shuttle", "traffic", "driver",
                        "vehicle", "route", "train",
                        "نقل", "مواصلات", "حافلة", "الحافلات", "باص", "تاكسي", "سائق", "قطار",
                        "خط سير", "محطة", "توصيل", "مسار", "ازدحام مروري", "زحمة مرورية",
                        "مواقف", "مواقف سيارات"},
    "accommodation": {"hotel", "room", "accommodation", "stay", "bed", "housing", "residence",
                       "check-in", "check-out", "suite",
                       "فندق", "غرفة", "غرف", "سكن", "إقامة", "سرير", "سكن الحجاج", "شقة",
                       "استراحة", "تسجيل الدخول", "فندقي"},
    "cleanliness": {"clean", "dirty", "hygiene", "hygienic", "unclean", "toilet", "trash", "garbage",
                     "smell",
                     "نظافة", "نظافة عامة", "نظيف", "متسخ", "دورات المياه", "قمامة", "زبالة",
                     "روائح", "رائحة كريهة", "صرف صحي", "تنظيف"},
    "crowd_management": {"crowd", "crowded", "packed", "overcrowding", "queue", "line", "waiting",
                          "organized", "chaos", "stampede",
                          "ازدحام", "ازدحام شديد", "زحمة", "تكدس", "تنظيم", "فوضى", "طابور",
                          "طوابير طويلة", "تدافع", "زحمة خانقة", "انتظار طويل"},
    "accessibility": {"wheelchair", "elderly", "disability", "disabled", "access", "special needs",
                       "ramp", "elevator", "senior citizens",
                       "كبار السن", "ذوي الإعاقة", "عربات", "سهولة الوصول", "إعاقة", "كرسي متحرك",
                       "منحدر", "مصعد", "ذوي الاحتياجات الخاصة", "المسنين"},
    "haram_experience": {"grand mosque", "kaaba", "tawaf", "haram", "black stone", "zamzam",
                          "المسجد الحرام", "الكعبة", "طواف", "الحرم المكي", "المطاف",
                          "الحجر الأسود", "زمزم", "ساحات الحرم"},
    "nabawi_experience": {"prophet's mosque", "nabawi", "rawdah", "green dome", "prophet's tomb",
                           "المسجد النبوي", "الروضة", "الحرم النبوي", "القبة الخضراء",
                           "الحجرة الشريفة", "باب السلام"},
    "hajj_experience": {"hajj", "arafah", "mina", "muzdalifah", "jamarat", "stoning", "pilgrims",
                         "الحج", "عرفة", "الوقوف بعرفة", "منى", "مزدلفة", "الجمرات", "رمي الجمرات",
                         "الحجاج", "طواف الإفاضة"},
    "umrah_experience": {"umrah", "umrah package", "umrah visa",
                          "العمرة", "المعتمرين", "معتمر", "عمرتي", "برنامج العمرة", "تأشيرة العمرة"},
}


def classify_category(text: str, text_en: str = "") -> str:
    """Heuristic keyword-scored classification (TIER B — no LLM key
    needed). Picks the category with the most keyword hits; "general" when
    nothing scores. The LLM tier overrides this with real understanding
    when a key is configured (see _LLM_PROMPT)."""
    hay = ((text or "") + " " + (text_en or "")).lower()
    best_cat, best_score = "general", 0
    for cat, words in _CATEGORY_KEYWORDS.items():
        score = sum(1 for w in words if w in hay)
        if score > best_score:
            best_cat, best_score = cat, score
    return best_cat

_LLM_PROMPT = """You are a strict JSON content-analysis service for a Hajj & Umrah pilgrimage feedback website. Analyze the comment below and reply with ONLY a JSON object, no other text:
{
 "language": "<ISO 639-1 code of the comment's language>",
 "sentiment": "positive"|"negative"|"neutral",
 "sentiment_confidence": <0-100>,
 "category": "<one of: """ + ",".join(CATEGORIES) + """>",
 "flags": [<zero or more of: "profanity","insult","hate_speech","harassment","racism","sexual","violence","spam">],
 "relevant": true|false,
 "reason": "<short reason in Arabic if flagged or irrelevant, else empty string>"
}
Rules:
- Understand context and sarcasm fully before deciding. Always pick exactly ONE of positive/negative/neutral — the single dominant sentiment, even when the comment has minor qualifiers on both sides. Weigh which side the comment is mainly expressing. Pure factual statements with no real opinion are "neutral".
- Brevity is not neutrality: a short comment or even a single word with clear sentiment ("excellent", "ممتاز", "bad", "سيئ") gets that same sentiment with high confidence — never default a short comment to "neutral" just because it's short.
- "category" = the single best-matching topic from the allowed list based on what the comment is mainly about; use "general" only when nothing else fits.
- "relevant" = true when the comment concerns the Hajj/Umrah journey in ANY way: rituals, the holy sites, crowds, organization, transport, hotels/accommodation for pilgrims, a Hajj/Umrah company or campaign, food, or services EXPERIENCED DURING pilgrimage. Generic content with no pilgrimage connection (random ads, unrelated products, off-topic chat) = false.
- Flag ONLY clear violations; ordinary criticism, even harsh, is NOT a violation.
Comment:
\"\"\"{TEXT}\"\"\""""


# ------------------------------------------------------------------ #
# TIER A — one LLM call analyzes everything (user's own key)
# ------------------------------------------------------------------ #
# v15.7: same API-key hygiene as assistant.py — strip whitespace/quotes and
# remove any CR/LF/TAB that would make an HTTP header value invalid
# ("Invalid header value b'Bearer ...'") and silently push every comment down
# to the non-LLM tier. Each provider keeps using its OWN key only.
def _clean_key(raw) -> str:
    if not raw:
        return ""
    key = str(raw).strip().strip('"').strip("'").strip()
    return key.replace("\r", "").replace("\n", "").replace("\t", "")


def _env_key(name: str, *fallbacks: str) -> str:
    for n in (name, *fallbacks):
        v = _clean_key(os.environ.get(n))
        if v:
            return v
    return ""


def _gemini_generate(g_key: str, prompt: str) -> str:
    """Call Gemini for the classification prompt and return the raw text.
    Model-agnostic: attempt 1 turns thinking off (cheap, correct on
    gemini-2.5-flash); attempt 2 drops thinkingConfig and widens the budget
    because newer 3.x models reject thinkingBudget=0 with an HTTP 400 — the
    exact failure that used to knock every comment down to the weaker
    non-LLM tier. Raises on total failure so _llm_call() logs + falls back.
    """
    model = os.environ.get("LLM_MODEL", DEFAULT_GEMINI_MODEL)
    url = ("https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent")
    attempts = [
        {"maxOutputTokens": 400, "responseMimeType": "application/json",
         "thinkingConfig": {"thinkingBudget": 0}},
        {"maxOutputTokens": 1200, "responseMimeType": "application/json"},
    ]
    last_detail = "no response"
    for i, gen_config in enumerate(attempts):
        r = requests.post(
            url,
            headers={"x-goog-api-key": g_key, "content-type": "application/json"},
            json={"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                  "generationConfig": gen_config,
                  "safetySettings": GEMINI_SAFETY_SETTINGS},
            timeout=LLM_TIMEOUT)
        if not r.ok:
            last_detail = f"HTTP {r.status_code}: {r.text[:300]}"
            print(f"[pipeline] Gemini API error {r.status_code} "
                  f"(attempt {i + 1}/{len(attempts)}): {r.text[:500]}")
            continue
        candidates = r.json().get("candidates") or []
        parts = (candidates[0].get("content") or {}).get("parts") or [] if candidates else []
        text = "".join(p.get("text", "") for p in parts).strip()
        if text:
            return text
        last_detail = "empty text"
        print(f"[pipeline] Gemini returned empty text (attempt {i + 1}/{len(attempts)})")
    raise RuntimeError(f"Gemini call failed: {last_detail}")


def _llm_call(text: str):
    """Returns the parsed JSON dict from Anthropic, OpenAI or Gemini, or None."""
    prompt = _LLM_PROMPT.replace("{TEXT}", text[:4000])
    a_key = _env_key("ANTHROPIC_API_KEY")
    o_key = _env_key("OPENAI_API_KEY")
    g_key = _env_key("GEMINI_API_KEY", "GOOGLE_API_KEY")
    try:
        if a_key:
            r = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": a_key, "anthropic-version": "2023-06-01",
                         "content-type": "application/json"},
                json={"model": os.environ.get("LLM_MODEL", DEFAULT_ANTHROPIC_MODEL),
                      "max_tokens": 300,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=LLM_TIMEOUT)
            if not r.ok:
                print(f"[pipeline] Anthropic API error {r.status_code}: {r.text[:500]}")
            r.raise_for_status()
            raw = "".join(b.get("text", "") for b in r.json().get("content", []))
        elif o_key:
            r = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {o_key}"},
                json={"model": os.environ.get("LLM_MODEL", DEFAULT_OPENAI_MODEL),
                      "max_tokens": 300,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=LLM_TIMEOUT)
            if not r.ok:
                print(f"[pipeline] OpenAI API error {r.status_code}: {r.text[:500]}")
            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"]
        elif g_key:
            raw = _gemini_generate(g_key, prompt)
        else:
            return None
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.M).strip()
        data = json.loads(raw)
        if data.get("sentiment") not in SENTIMENT_LABELS:
            return None
        if data.get("category") not in CATEGORIES:
            data["category"] = None  # process() falls back to the heuristic classifier
        return data
    except Exception as e:
        print(f"[pipeline] LLM analysis failed, falling back: {e}")
        return None


def llm_configured() -> str:
    if _env_key("ANTHROPIC_API_KEY"):
        return "anthropic"
    if _env_key("OPENAI_API_KEY"):
        return "openai"
    if _env_key("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        return "gemini"
    return ""


def providers_configured() -> dict:
    """v15.2: unlike llm_configured() (the single WINNING provider under the
    Anthropic -> OpenAI -> Gemini priority order), this reports each
    provider's key independently — e.g. so the admin panel can show "Gemini:
    configured" even while Anthropic is the one actually answering, the same
    way the External Sources panel shows YouTube/Reddit/X/Google Maps side
    by side regardless of which ones are actually active."""
    return {
        "anthropic": bool(_env_key("ANTHROPIC_API_KEY")),
        "openai": bool(_env_key("OPENAI_API_KEY")),
        "gemini": bool(_env_key("GEMINI_API_KEY", "GOOGLE_API_KEY")),
    }


# ------------------------------------------------------------------ #
# TIER B moderation — HF toxicity model + local wordlists / heuristics
# ------------------------------------------------------------------ #
HF_TOX_MODEL = "unitary/multilingual-toxic-xlm-roberta"


def _hf_toxicity(text: str):
    token = _env_key("HF_API_TOKEN")
    if not token:
        return None
    try:
        r = requests.post(
            f"https://api-inference.huggingface.co/models/{HF_TOX_MODEL}",
            headers={"Authorization": f"Bearer {token}"},
            json={"inputs": text[:1500], "options": {"wait_for_model": True}},
            timeout=20)
        r.raise_for_status()
        data = r.json()
        cand = data[0] if data and isinstance(data[0], list) else data
        for item in cand:
            if str(item.get("label", "")).lower() in ("toxic", "toxicity", "label_1"):
                return float(item.get("score", 0))
        return 0.0
    except Exception as e:
        print(f"[pipeline] toxicity model failed: {e}")
        return None


# Local moderation wordlists — a safety net that always runs, extensible via
# the EXTRA_BANNED_WORDS env var (comma-separated). Kept intentionally
# conservative: normal harsh criticism must never be flagged.
_PROFANITY = {
    # English
    "fuck", "fucking", "shit", "bitch", "asshole", "bastard", "dick", "cunt",
    "whore", "slut", "motherfucker", "porn", "nude",
    # Arabic (common explicit insults)
    "كلب", "حمار", "حقير", "قذر", "وسخ يا", "يلعن", "تفو", "زبالة", "خنزير",
    "حيوان يا", "غبي يا", "عاهرة", "قحبة", "زانية", "ابن الكلب", "يا خول",
}
_HATE_VIOLENCE = {
    "kill you", "i will kill", "deserve to die", "exterminate", "terrorist scum",
    "سأقتلك", "اقتلوهم", "يستاهلون الموت", "ابادة", "اذبحوهم",
}
_SPAM_PATTERNS = [
    re.compile(r"(https?://\S+.*){2,}", re.S),          # 2+ links
    re.compile(r"(whatsapp|واتساب|واتس اب).{0,30}\+?\d{8,}", re.I),
    re.compile(r"(اربح|ربح مضمون|win money|crypto|forex|promo code|discount code)", re.I),
    re.compile(r"(.)\1{9,}"),                            # aaaaaaaaaa spam
]


def _local_moderation(text: str, text_en: str):
    """Returns (flags, reason) from wordlists + heuristics."""
    flags, reason = [], ""
    hay = (text + " " + (text_en or "")).lower()
    extra = {w.strip().lower() for w in os.environ.get("EXTRA_BANNED_WORDS", "").split(",") if w.strip()}
    if any(w in hay for w in _PROFANITY | extra):
        flags.append("profanity")
        reason = "ألفاظ غير لائقة"
    if any(w in hay for w in _HATE_VIOLENCE):
        flags.append("violence")
        reason = "تهديد أو تحريض على العنف"
    for pat in _SPAM_PATTERNS:
        if pat.search(text):
            flags.append("spam")
            reason = reason or "محتوى دعائي/سبام"
            break
    return list(dict.fromkeys(flags)), reason


# ------------------------------------------------------------------ #
# TIER B relevance — topic heuristics (EXTERNAL comments only)
# ------------------------------------------------------------------ #
# v15: the topic-word list is now shared with assistant.py's scope guard,
# defined once in knowledge_base.TOPIC_WORDS.
def _heuristic_relevant(text: str, text_en: str) -> bool:
    return knowledge_base.in_scope(text) or knowledge_base.in_scope(text_en or "")


# v15: place types that make a Google Maps review relevant by definition,
# per the product spec (hajj/umrah experience, pilgrim hotel, pilgrim
# transport company, hajj/umrah company, hajj campaign, the Grand Mosque,
# the Prophet's Mosque, the sacred sites, crowd management, government
# pilgrim services) — bypasses the free-text heuristic, which can miss a
# short review like "Great stay!" that has no Hajj/Umrah keyword in it even
# though the PLACE itself is unambiguously pilgrim-related.
RELEVANT_PLACE_TYPES = {
    "hajj_experience", "umrah_experience", "pilgrim_hotel", "pilgrim_transport",
    "hajj_umrah_company", "hajj_campaign", "grand_mosque", "prophet_mosque",
    "sacred_sites", "crowd_management", "government_pilgrim_service",
}


def _place_type_relevant(place_type: str) -> bool:
    return bool(place_type) and place_type.strip().lower() in RELEVANT_PLACE_TYPES


# ------------------------------------------------------------------ #
# The pipeline
# ------------------------------------------------------------------ #
def _synthesize_scores(label: str, confidence: float) -> dict:
    """The LLM tier only returns the dominant label + a confidence number,
    not a full distribution — but the UI's percentage bars expect one for
    all four labels. Give the dominant label its confidence and split the
    remainder evenly across the other three so the bars are never blank."""
    confidence = max(0.0, min(100.0, confidence))
    remainder = round((100.0 - confidence) / 3, 1)
    scores = {lbl: remainder for lbl in SENTIMENT_LABELS}
    scores[label] = round(confidence, 1)
    return scores


def process(text: str, ml_predict=None, is_external: bool = False, place_type: str = None) -> dict:
    """Run the FULL pipeline on one comment. Never raises.

    place_type (optional): for external reviews with a known place type
    (e.g. from Google Maps — see ai_pipeline.RELEVANT_PLACE_TYPES) — lets a
    pilgrim-related place count as relevant even when the review text alone
    doesn't mention Hajj/Umrah keywords ("Great stay!" at a pilgrim hotel).

    Returns:
      detected_language, text_ar, sentiment, confidence, scores, engine,
      category, moderation_status ('approved'|'flagged'|'rejected'),
      moderation_flags (list), moderation_reason (str), relevant (bool),
      content_fingerprint (str) — for cross-source de-duplication.
    """
    text = (text or "").strip()
    # -- translation first: the display copy AND the analysis input --
    tr_ar = translation.detect_and_translate(text, target="ar")
    detected = tr_ar["detected_lang"]
    text_ar = None
    if tr_ar["ok"] and detected and detected != "ar" and tr_ar["translated"] != text:
        text_ar = tr_ar["translated"]
    text_en = translation.to_english(text)

    llm = _llm_call(text)
    if llm is not None:
        detected = llm.get("language") or detected
        flags = [f for f in (llm.get("flags") or []) if f in MODERATION_FLAG_KEYS]
        relevant = bool(llm.get("relevant", True)) or _place_type_relevant(place_type)
        confidence = round(float(llm.get("sentiment_confidence", 80)), 1)
        category = llm.get("category") or classify_category(text, text_en)
        result = {
            "sentiment": llm["sentiment"],
            "confidence": confidence,
            "scores": _synthesize_scores(llm["sentiment"], confidence),
            "engine": "llm-" + llm_configured(),
            "category": category,
            "moderation_flags": flags,
            "moderation_reason": (llm.get("reason") or "")[:300],
            "relevant": relevant,
        }
    else:
        # sentiment via the v13 tiered engine (always positive/negative/neutral)
        s = sentiment.analyze(text, ml_predict=ml_predict)
        label = s["label"]
        # moderation: local safety net + optional toxicity transformer
        flags, reason = _local_moderation(text, text_en)
        tox = _hf_toxicity(text)
        if tox is not None and tox >= 0.80 and "profanity" not in flags:
            flags.append("insult")
            reason = reason or "محتوى مسيء (نموذج كشف السمية)"
        # relevance heuristic: external comments only — site reviews are
        # presumed on-topic in this tier (only the LLM can judge them safely)
        relevant = (_heuristic_relevant(text, text_en) or _place_type_relevant(place_type)) if is_external else True
        result = {
            "sentiment": label,
            "confidence": s["confidence"],
            "scores": s.get("scores", {}),
            "engine": s.get("engine"),
            "category": classify_category(text, text_en),
            "moderation_flags": flags,
            "moderation_reason": reason,
            "relevant": relevant,
        }

    if not result["relevant"]:
        result["moderation_status"] = "rejected"
        result["moderation_reason"] = result["moderation_reason"] or "غير متعلق بالحج والعمرة"
    elif result["moderation_flags"]:
        result["moderation_status"] = "flagged"
    else:
        result["moderation_status"] = "approved"
    result["detected_language"] = detected
    result["text_ar"] = text_ar
    # v15: fingerprint on the English translation so the SAME opinion posted
    # in different languages on different platforms still matches.
    result["content_fingerprint"] = dedup.fingerprint(text_en or text)
    return result


def pipeline_status() -> dict:
    provider = llm_configured()
    default_model = {"anthropic": DEFAULT_ANTHROPIC_MODEL, "openai": DEFAULT_OPENAI_MODEL,
                      "gemini": DEFAULT_GEMINI_MODEL}.get(provider)
    return {
        "llm_provider": provider,
        "llm_model": os.environ.get("LLM_MODEL") or default_model,
        "llm_providers_configured": providers_configured(),
        "toxicity_model_enabled": bool(os.environ.get("HF_API_TOKEN")),
        "categories": CATEGORIES,
        "sentiment_labels": SENTIMENT_LABELS,
        **sentiment.engine_status(),
    }
