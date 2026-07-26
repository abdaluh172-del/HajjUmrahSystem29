# -*- coding: utf-8 -*-
""""تعليمات الحج والعمرة" — the specialized Hajj & Umrah AI assistant (v15).

A ChatGPT-style assistant, but locked to ONE domain: Hajj, Umrah, their
rituals, and every pilgrim-facing service at the two Holy Mosques. Anything
else gets a polite redirect instead of an answer.

Design goals (from the product spec):
  * Answer ONLY questions about Hajj/Umrah rituals & Haramain services;
    politely decline anything else and ask for an on-topic question.
  * Never guess: lean on a small curated knowledge base (knowledge_base.py)
    for grounding, cite the category of official source responsible
    (Ministry of Hajj & Umrah / the Grand Mosque & Prophet's Mosque
    presidency / Nusuk platform), and say so plainly when a detail isn't
    confirmed rather than inventing it.
  * Structured, professional answers: a heading, a short intro, ordered
    points/steps, important warnings, shar'i rulings when relevant (with a
    pointer to official Ifta offices for personal rulings), correct
    duas/adhkar, service locations, and a "related info" close.

Tiering mirrors ai_pipeline.py: reuses ANTHROPIC_API_KEY / OPENAI_API_KEY /
GEMINI_API_KEY (same env vars — no extra configuration) for the highest-
quality answers; without a key, falls back to a template built from
knowledge_base.py so the page is never empty/broken, just less
conversational.
"""
import json
import os
import re
import traceback

import requests

import knowledge_base

# v15.7: a build stamp so the DEPLOYED instance can be verified from the
# outside — not just "the code is correct" but "the running server is THIS
# code". It is logged at import + on every answer() call, and returned in the
# chat response, status(), and diagnostics(). If a request's response or the
# Render logs don't show this exact string, the platform is serving a stale
# build (wrong branch, cached build, old container) — redeploy/clear cache.
BUILD_VERSION = "assistant-v15.7-llm-cascade+diagnostics"

# Printed once when the module is imported by Gunicorn on Render, alongside the
# resolved file path — makes any accidental shadowing (a second assistant.py on
# the path) immediately visible in the deploy logs.
print(f"[assistant] module loaded: version={BUILD_VERSION} file={__file__}")

LLM_TIMEOUT = 30
MAX_HISTORY_MESSAGES = 12  # keep the request small & the assistant focused

# v15.1 fix: "claude-haiku-4-5" (no date suffix) is NOT a valid Anthropic
# model id — every request using it fails with 404, the exception is caught
# by answer()'s try/except, and the assistant silently falls back to the
# bare knowledge-base template forever, even with a perfectly good API key.
# This was the main reason the assistant looked "broken". Overridable via
# the LLM_MODEL env var if the site owner wants a different model/tier.
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
# v15.3: pinned to the stable GA model "gemini-2.5-flash" instead of the
# "gemini-flash-latest" alias. The alias can resolve to an experimental /
# 3.x model that (a) may carry more restrictive rate limits and (b) REJECTS
# generationConfig.thinkingConfig.thinkingBudget=0 with an HTTP 400 — which
# was making this whole tier fall back silently to the knowledge-base
# template ("لم يتم تفعيل نموذج ذكاء اصطناعي بعد") even with a perfectly
# valid GEMINI_API_KEY set. gemini-2.5-flash is fast, cheap, supported, and
# accepts the thinking-off config. _call_gemini() below ALSO retries without
# thinkingConfig, so even a 3.x model set via LLM_MODEL still answers.
# Overridable via the LLM_MODEL env var.
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

# v15.6: a resilient fallback chain of widely-available Gemini models. The
# configured model (LLM_MODEL or DEFAULT_GEMINI_MODEL) is tried FIRST; if it
# comes back 404 "model not found" / 403 "not available for your key" (which
# differs from one API key/region/tier to another and was a real cause of the
# assistant looking "broken"), the next model is tried automatically. All are
# fast + cheap generateContent models. De-duplicated at call time so the
# configured model is never tried twice.
GEMINI_FALLBACK_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-flash-latest",
    "gemini-1.5-flash",
]

# v15.6: the real reason the last generative-AI attempt failed (HTTP status +
# short body, or a network message). Surfaced to the admin via
# /api/assistant/diagnostics and appended (briefly) to the user-facing fallback
# message so "the AI can't reply" is never shown without an actual cause.
_LAST_LLM_ERROR = {"provider": None, "detail": None, "at": None}


def last_llm_error() -> dict:
    return dict(_LAST_LLM_ERROR)


def _record_llm_error(provider: str, detail: str):
    import time
    _LAST_LLM_ERROR.update({"provider": provider, "detail": (detail or "")[:500],
                            "at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())})

# Same reasoning as ai_pipeline.GEMINI_SAFETY_SETTINGS: a pilgrim can
# legitimately ask about difficult topics (crowd-crush safety, medical
# emergencies, grief) without tripping Google's default filters and coming
# back empty — the on-topic/off-topic boundary here is enforced by the
# system prompt + knowledge_base.in_scope(), not by the safety filter.
GEMINI_SAFETY_SETTINGS = [
    {"category": c, "threshold": "BLOCK_NONE"} for c in (
        "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT",
    )
]

LANG_NAMES = {
    "ar": "Arabic", "en": "English", "tr": "Turkish", "ur": "Urdu",
    "hi": "Hindi", "he": "Hebrew",
}


def _system_prompt(lang: str, kb_context: str) -> str:
    lang_name = LANG_NAMES.get(lang, "Arabic")
    prompt = f"""You are "تعليمات الحج والعمرة" (Hajj & Umrah Guidance), a specialized assistant \
inside a Hajj & Umrah pilgrim-feedback platform. You help pilgrims and prospective pilgrims with \
Hajj, Umrah, and Haramain (the two Holy Mosques) services ONLY.

IN-SCOPE topics (answer these): Hajj rituals, Umrah rituals, Ihram, Tawaf, Sa'i, standing at \
Arafah, Muzdalifah, Mina, stoning the Jamarat, the Hady (sacrifice), shaving/trimming, Tawaf \
al-Ifadah, Tawaf al-Wada', the Miqats, Ihram prohibitions, Fidyah, du'as and adhkar, shar'i \
rulings related to Hajj/Umrah, Grand Mosque services, Prophet's Mosque services, Ifta (fatwa) \
offices, guidance offices, lesson/lecture locations, Qur'an circles, restrooms, ablution areas, \
gates, prayer areas, elderly/disability carts, first aid, health centers, lost & found, crowd \
management, transport services, official Hajj/Umrah apps (e.g. Nusuk), and any other service for \
pilgrims within Makkah, Madinah, or the sacred sites (al-Masha'ir al-Muqaddasah).

OUT OF SCOPE (politely decline): anything unrelated to Hajj/Umrah/Haramain services — general \
chit-chat, coding, unrelated travel, politics, other religions' rituals, etc. When a question is \
out of scope, apologize briefly and ask the person to ask something about Hajj or Umrah instead. \
Do NOT answer the off-topic question even partially.

RELIABILITY RULES (critical):
- Never invent rulings, prices, phone numbers, exact locations, or dates. If you are not certain, \
say so plainly and suggest the person confirm with an official source.
- Ground your answers in what is well-established; official sources you may refer to by name are: \
{knowledge_base.OFFICIAL_SOURCES_AR} ({knowledge_base.OFFICIAL_SOURCES_EN}).
- For personal shar'i rulings (e.g. "did I do X correctly", fidyah for a specific situation), give \
the general rule and explicitly recommend confirming with an official Ifta/guidance office rather \
than issuing a personal fatwa yourself.
- Only give du'as/adhkar you are confident are authentic and correctly worded; if unsure of exact \
wording, describe the general content instead of inventing wording.

ANSWER FORMAT (use Markdown):
- A short bold heading line.
- A one-to-two sentence introduction.
- Organized bullet points, and numbered sequential steps when the answer describes a procedure.
- An "⚠️ تنبيه مهم" / "⚠️ Important" callout for any important warning, if relevant.
- Shar'i rulings when relevant, phrased carefully per the reliability rules above.
- Correct du'as/adhkar when relevant.
- Service locations when relevant.
- End with one short line suggesting a related follow-up topic.

LANGUAGE: Respond in {lang_name}. If the person's message is written in a different language, \
respond in the language they used instead.

{("GROUNDING CONTEXT (verified reference material — prefer this over your own memory when it " \
"applies; it may be partial or empty):\n" + kb_context) if kb_context else ""}"""
    return prompt


def _heuristic_in_scope(text: str) -> bool:
    return knowledge_base.in_scope(text)


# ------------------------------------------------------------------ #
# v15.7 — API-KEY HYGIENE (root-cause fix for the production outage)
# ------------------------------------------------------------------ #
# Render (and most dashboards) frequently store a pasted secret WITH a
# trailing newline or stray whitespace. When that raw value is dropped into
# an HTTP header — e.g. Authorization: "Bearer sk-proj-...\n" — the requests
# library rejects it with:
#     Invalid header value b'Bearer sk-proj-...'
# which was caught by answer()'s try/except and silently downgraded the whole
# assistant to the knowledge base ("LLM call failed, falling back to KB").
# Every key MUST therefore be sanitised before it touches a header. We strip
# surrounding whitespace, strip accidental surrounding quotes, and remove any
# CR/LF/TAB anywhere in the value (those are exactly the characters that make
# an HTTP header value invalid).
def _clean_key(raw) -> str:
    if not raw:
        return ""
    key = str(raw).strip().strip('"').strip("'").strip()
    return key.replace("\r", "").replace("\n", "").replace("\t", "")


def _env_key(name: str, *fallbacks: str) -> str:
    """Read an API key from the environment and sanitise it. Extra names are
    tried in order (e.g. GEMINI_API_KEY then GOOGLE_API_KEY)."""
    for n in (name, *fallbacks):
        v = _clean_key(os.environ.get(n))
        if v:
            return v
    return ""


# v15.7: provider fallback CASCADE. Each provider is tried with its OWN key;
# if it fails (bad/expired key, invalid header, model-not-found, quota, empty
# text, network) we move to the NEXT configured provider — we NEVER reuse one
# provider's key for another provider's API — and only when ALL configured
# providers fail do we fall back to the knowledge base:
#     <ordered providers> -> Knowledge Base
# The order is configurable via the LLM_PROVIDER_ORDER env var (comma
# separated). To make Gemini primary with OpenAI then Claude as fallbacks —
# i.e. Gemini -> OpenAI -> Claude -> Knowledge Base — set on Render:
#     LLM_PROVIDER_ORDER=gemini,openai,anthropic
# Default preserves the historical priority so existing deployments are
# unchanged unless they opt in.
_VALID_PROVIDERS = ("anthropic", "openai", "gemini")
_DEFAULT_PROVIDER_ORDER = ["anthropic", "openai", "gemini"]


def _provider_order() -> list:
    raw = os.environ.get("LLM_PROVIDER_ORDER", "")
    order = [p.strip().lower() for p in raw.split(",") if p.strip()]
    order = [p for p in order if p in _VALID_PROVIDERS]
    if not order:
        order = list(_DEFAULT_PROVIDER_ORDER)
    seen, out = set(), []
    for p in order:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _provider_key(provider: str) -> str:
    """Return the sanitised key that belongs to THIS provider (never another's)."""
    if provider == "anthropic":
        return _env_key("ANTHROPIC_API_KEY")
    if provider == "openai":
        return _env_key("OPENAI_API_KEY")
    if provider == "gemini":
        return _env_key("GEMINI_API_KEY", "GOOGLE_API_KEY")
    return ""


def _providers_in_order() -> list:
    """Configured providers, in the configured order, each having its own key."""
    return [p for p in _provider_order() if _provider_key(p)]


def llm_configured() -> str:
    """The PRIMARY provider (first configured one in the configured order).
    Kept for status()/diagnostics(); the actual answer path cascades over
    _providers_in_order()."""
    configured = _providers_in_order()
    return configured[0] if configured else ""


def providers_configured() -> dict:
    """v15.2: see ai_pipeline.providers_configured() — same idea, reported
    independently here since this module already duplicates llm_configured()
    rather than importing ai_pipeline. v15.7: keys are sanitised, so a value
    that is only whitespace/newline no longer counts as "configured"."""
    return {
        "anthropic": bool(_env_key("ANTHROPIC_API_KEY")),
        "openai": bool(_env_key("OPENAI_API_KEY")),
        "gemini": bool(_env_key("GEMINI_API_KEY", "GOOGLE_API_KEY")),
    }


def _call_anthropic(system_prompt: str, messages: list) -> str:
    key = _env_key("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("Anthropic call failed: ANTHROPIC_API_KEY missing/empty after cleaning")
    model = os.environ.get("LLM_MODEL", DEFAULT_ANTHROPIC_MODEL)
    print(f"[assistant] Trying Anthropic (model={model})")
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        json={
            "model": model,
            "max_tokens": 1000,
            "system": system_prompt,
            "messages": messages,
        },
        timeout=LLM_TIMEOUT,
    )
    if not r.ok:
        # Surface the real reason in the server log AND in the raised exception
        # (status + body), so /api/assistant/diagnostics and the fallback
        # message show the true cause instead of a generic "LLM call failed".
        print(f"[assistant] Anthropic API error {r.status_code}: {r.text[:500]}")
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    data = r.json()
    text = "".join(b.get("text", "") for b in data.get("content", [])).strip()
    print(f"[assistant] Anthropic returned {len(text)} chars")
    return text


def _call_openai(system_prompt: str, messages: list) -> str:
    # v15.7: sanitised so a trailing newline in OPENAI_API_KEY can no longer
    # produce `Invalid header value b'Bearer sk-proj-...'` — the exact error
    # seen in the Render logs that knocked the assistant down to the KB.
    key = _env_key("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OpenAI call failed: OPENAI_API_KEY missing/empty after cleaning")
    model = os.environ.get("LLM_MODEL", DEFAULT_OPENAI_MODEL)
    print(f"[assistant] Trying OpenAI (model={model})")
    oa_messages = [{"role": "system", "content": system_prompt}] + messages
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": model,
            "max_tokens": 1000,
            "messages": oa_messages,
        },
        timeout=LLM_TIMEOUT,
    )
    if not r.ok:
        print(f"[assistant] OpenAI API error {r.status_code}: {r.text[:500]}")
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    text = (r.json()["choices"][0]["message"]["content"] or "").strip()
    print(f"[assistant] OpenAI returned {len(text)} chars")
    return text


def _gemini_extract_text(data: dict) -> str:
    """Pull the reply text out of a Gemini response, tolerating an empty or
    filtered candidate list instead of raising a KeyError."""
    candidates = data.get("candidates") or []
    if not candidates:
        return ""
    parts = (candidates[0].get("content") or {}).get("parts") or []
    return "".join(p.get("text", "") for p in parts).strip()


def _gemini_models_to_try() -> list:
    """Configured model first (LLM_MODEL if set, else the pinned default),
    then the fallback chain — de-duplicated, order preserved."""
    ordered = [os.environ.get("LLM_MODEL") or DEFAULT_GEMINI_MODEL] + GEMINI_FALLBACK_MODELS
    seen, out = set(), []
    for m in ordered:
        m = (m or "").strip()
        if m and m not in seen:
            seen.add(m)
            out.append(m)
    return out


def _gemini_generate(model: str, key: str, system_prompt: str, contents: list):
    """One model, two generationConfig attempts (thinking-off, then no
    thinkingConfig). Returns (text, detail). text is "" if this model failed;
    detail explains why. Never raises."""
    attempts = [
        {"maxOutputTokens": 1000, "thinkingConfig": {"thinkingBudget": 0}},
        {"maxOutputTokens": 2048},
    ]
    detail = "no response"
    for i, gen_config in enumerate(attempts):
        try:
            r = requests.post(
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model}:generateContent",
                headers={"x-goog-api-key": key, "content-type": "application/json"},
                json={
                    "contents": contents,
                    "systemInstruction": {"parts": [{"text": system_prompt}]},
                    "generationConfig": gen_config,
                    "safetySettings": GEMINI_SAFETY_SETTINGS,
                },
                timeout=LLM_TIMEOUT,
            )
        except Exception as e:
            detail = f"network error contacting Gemini: {e}"
            print(f"[assistant] Gemini request failed (model={model}, "
                  f"attempt {i + 1}/{len(attempts)}): {e}")
            continue
        if not r.ok:
            detail = f"HTTP {r.status_code}: {r.text[:300]}"
            print(f"[assistant] Gemini API error {r.status_code} (model={model}, "
                  f"attempt {i + 1}/{len(attempts)}): {r.text[:500]}")
            # A 404/403 means THIS model isn't usable for this key — no point
            # retrying the second generationConfig; move on to the next model.
            if r.status_code in (403, 404):
                return "", detail
            continue
        text = _gemini_extract_text(r.json())
        if text:
            return text, "ok"
        detail = "empty text (finishReason may be MAX_TOKENS/SAFETY)"
        print(f"[assistant] Gemini returned empty text (model={model}, "
              f"attempt {i + 1}/{len(attempts)})")
    return "", detail


def _call_gemini(system_prompt: str, messages: list) -> str:
    # v15.7: Gemini uses its OWN key only (GEMINI_API_KEY, then GOOGLE_API_KEY)
    # — never OpenAI's — sanitised the same way as the others.
    key = _env_key("GEMINI_API_KEY", "GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Gemini call failed: no GEMINI_API_KEY/GOOGLE_API_KEY in environment")
    # Gemini has no "assistant" role — the model's own turns are "model".
    contents = [{"role": "model" if m["role"] == "assistant" else "user",
                 "parts": [{"text": m["content"]}]} for m in messages]

    last_detail = "no response"
    for model in _gemini_models_to_try():
        print(f"[assistant] Trying Gemini (model={model})")
        text, detail = _gemini_generate(model, key, system_prompt, contents)
        if text:
            print(f"[assistant] Gemini returned {len(text)} chars (model={model})")
            return text
        last_detail = f"model {model}: {detail}"
    # Every model failed — record the real reason and raise so answer() logs it
    # and falls back to the knowledge base with a clear message.
    _record_llm_error("gemini", last_detail)
    raise RuntimeError(f"Gemini call failed ({last_detail})")


def _human_reason_ar(detail: str) -> str:
    """Turn a raw error detail into a short, clear Arabic reason for the user."""
    d = (detail or "").lower()
    if not d:
        return "خطأ غير معروف من خدمة الذكاء الاصطناعي."
    if "no gemini_api_key" in d or "no api" in d or "environment" in d and "key" in d:
        return "لم يتم ضبط مفتاح الذكاء الاصطناعي على الخادم."
    if "network" in d or "timeout" in d or "timed out" in d or "connection" in d:
        return "تعذّر الاتصال بخدمة الذكاء الاصطناعي (مشكلة شبكة أو مهلة)."
    if "http 400" in d:
        return "طلب غير صالح لنموذج الذكاء الاصطناعي (إعداد غير مقبول)."
    if "http 401" in d or "http 403" in d or "api key not valid" in d or "permission" in d:
        return "مفتاح الذكاء الاصطناعي غير صالح أو لا يملك صلاحية الوصول."
    if "http 404" in d or "not found" in d:
        return "النموذج المطلوب غير متاح لهذا المفتاح."
    if "http 429" in d or "quota" in d or "rate" in d:
        return "تم تجاوز حصة الاستخدام المسموحة مؤقتًا — حاول لاحقًا."
    if "http 5" in d:
        return "خدمة الذكاء الاصطناعي تواجه عطلًا مؤقتًا."
    if "empty text" in d or "safety" in d or "max_tokens" in d:
        return "لم يُرجع النموذج نصًا (قد يكون بسبب مرشّح الأمان أو حد الطول)."
    return "حدث خطأ أثناء الاتصال بخدمة الذكاء الاصطناعي."


def _human_reason_en(detail: str) -> str:
    """Short, clear English reason for the user."""
    d = (detail or "").lower()
    if not d:
        return "Unknown error from the AI service."
    if "no gemini_api_key" in d or "no api" in d or ("environment" in d and "key" in d):
        return "The AI key isn't set on the server."
    if "network" in d or "timeout" in d or "timed out" in d or "connection" in d:
        return "Couldn't reach the AI service (network or timeout)."
    if "http 400" in d:
        return "Bad request to the AI model (unsupported config)."
    if "http 401" in d or "http 403" in d or "api key not valid" in d or "permission" in d:
        return "The AI key is invalid or lacks access."
    if "http 404" in d or "not found" in d:
        return "The requested model isn't available for this key."
    if "http 429" in d or "quota" in d or "rate" in d:
        return "Usage quota exceeded for now — try again later."
    if "http 5" in d:
        return "The AI service is temporarily down."
    if "empty text" in d or "safety" in d or "max_tokens" in d:
        return "The model returned no text (safety filter or length limit)."
    return "An error occurred contacting the AI service."


def _kb_fallback_answer(question: str, lang: str, provider_configured: bool = False) -> dict:
    """Fallback to the knowledge base. Reached either because no LLM key is
    configured, OR because a configured provider's call failed this turn.
    provider_configured distinguishes the two so the "empty knowledge base"
    message doesn't wrongly claim no AI is set up when one actually is (it
    just errored — the real cause is in the server logs)."""
    if not _heuristic_in_scope(question):
        msg = {
            "ar": "أستطيع الإجابة فقط عن أسئلة متعلقة بالحج والعمرة وخدمات الحرمين الشريفين. "
                  "تفضل بطرح سؤال في هذا النطاق 🙏",
            "en": "I can only answer questions about Hajj, Umrah, and services at the two Holy "
                  "Mosques. Please ask something in that scope 🙏",
        }.get(lang, None) or {
            "ar": "أستطيع الإجابة فقط عن أسئلة متعلقة بالحج والعمرة وخدمات الحرمين الشريفين.",
        }["ar"]
        return {"reply": msg, "engine": "scope-guard", "out_of_scope": True}

    hits = knowledge_base.retrieve(question, limit=2)
    if not hits:
        # Two different situations, two different honest messages:
        #   * a provider IS configured but the call failed this turn -> a
        #     temporary error, NOT "no AI configured" (that was the confusing
        #     message the user reported seeing even with Gemini enabled).
        #   * no provider configured at all -> say so plainly.
        if provider_configured:
            # v15.6: surface the REAL reason (bad key, model not found, quota,
            # rejected config, network) instead of a generic "try again" — the
            # request explicitly asked for the true cause to be shown, not hidden.
            err = last_llm_error().get("detail")
            reason_ar = _human_reason_ar(err)
            reason_en = _human_reason_en(err)
            msg = {
                "ar": f"تعذّر الحصول على رد من المساعد الذكي حاليًا. السبب: {reason_ar} "
                      f"يمكنك المحاولة مرة أخرى، أو مراجعة {knowledge_base.OFFICIAL_SOURCES_AR}.",
                "en": f"Couldn't get a reply from the AI assistant right now. Reason: {reason_en} "
                      f"You can try again, or check {knowledge_base.OFFICIAL_SOURCES_EN}.",
            }.get(lang) or (
                f"تعذّر الحصول على رد من المساعد الذكي حاليًا. السبب: {reason_ar} "
                f"يمكنك المحاولة مرة أخرى، أو مراجعة {knowledge_base.OFFICIAL_SOURCES_AR}."
            )
            return {"reply": msg, "engine": "kb-fallback-empty",
                    "out_of_scope": False, "error_detail": err}
        else:
            msg = {
                "ar": "هذا سؤال متعلق بالحج والعمرة، لكن لا تتوفر لديّ حاليًا معلومة موثوقة كافية "
                      "للإجابة عليه بدقة (لم يتم تفعيل نموذج ذكاء اصطناعي بعد). "
                      f"يُرجى مراجعة {knowledge_base.OFFICIAL_SOURCES_AR} للتأكد.",
                "en": "This is a Hajj/Umrah question, but I don't have enough verified information to "
                      "answer it precisely right now (no AI model is configured yet). Please check "
                      f"{knowledge_base.OFFICIAL_SOURCES_EN}.",
            }.get(lang) or (
                "هذا سؤال متعلق بالحج والعمرة، لكن لا تتوفر لديّ حاليًا معلومة موثوقة كافية للإجابة "
                f"عليه بدقة. يُرجى مراجعة {knowledge_base.OFFICIAL_SOURCES_AR}."
            )
        return {"reply": msg, "engine": "kb-fallback-empty", "out_of_scope": False}

    use_ar = lang != "en"
    parts = []
    for e in hits:
        title = e["title_ar"] if use_ar else e["title_en"]
        body = e["body_ar"] if use_ar else e["body_en"]
        parts.append(f"**{title}**\n\n{body}")
    footer = (
        f"\n\n_هذه معلومات عامة من قاعدة معرفة داخلية. للحصول على إجابات أكثر تفصيلًا وسياقًا، "
        f"يمكن لمسؤول الموقع تفعيل الذكاء الاصطناعي التوليدي عبر إضافة مفتاح API. للتأكد من "
        f"التفاصيل الدقيقة راجع {knowledge_base.OFFICIAL_SOURCES_AR}._"
        if use_ar else
        f"\n\n_This is general information from an internal knowledge base. For more detailed, "
        f"conversational answers, the site admin can enable the generative AI tier by adding an "
        f"API key. For precise details, check {knowledge_base.OFFICIAL_SOURCES_EN}._"
    )
    return {"reply": "\n\n---\n\n".join(parts) + footer, "engine": "kb-fallback", "out_of_scope": False}


def answer(history: list, lang: str = "ar") -> dict:
    """history: list of {"role": "user"|"assistant", "content": str}, oldest
    first, ending with the newest user message. Returns
    {"reply": str, "engine": str, "out_of_scope": bool, "assistant_version": str}.
    Never raises. The assistant_version field lets the client/browser confirm
    the DEPLOYED backend is running this build."""
    history = [h for h in (history or [])
               if h.get("role") in ("user", "assistant") and (h.get("content") or "").strip()]
    if not history or history[-1]["role"] != "user":
        return {"reply": "", "engine": "none", "out_of_scope": False,
                "assistant_version": BUILD_VERSION}
    history = history[-MAX_HISTORY_MESSAGES:]
    last_question = history[-1]["content"]

    # v15.7: try EVERY configured provider in order (each with its own key)
    # before giving up to the knowledge base — a single provider's failure
    # (bad key, invalid header, model-not-found, quota, empty text, network)
    # no longer takes the whole assistant down. Order is configurable via
    # LLM_PROVIDER_ORDER (e.g. gemini,openai,anthropic).
    providers = _providers_in_order()
    print(f"[assistant] version={BUILD_VERSION} provider order={_provider_order()} "
          f"(source={'env:LLM_PROVIDER_ORDER' if os.environ.get('LLM_PROVIDER_ORDER') else 'default'}); "
          f"configured/usable={providers}")
    if providers:
        kb_ctx = knowledge_base.context_block(last_question, lang=lang, limit=3)
        system_prompt = _system_prompt(lang, kb_ctx)
        messages = [{"role": h["role"], "content": h["content"]} for h in history]
        callers = {"anthropic": _call_anthropic, "openai": _call_openai, "gemini": _call_gemini}
        for provider in providers:
            try:
                text = callers[provider](system_prompt, messages)
                if text:
                    print(f"[assistant] SUCCESS via {provider} — returning LLM reply")
                    return {"reply": text, "engine": f"llm-{provider}", "out_of_scope": False,
                            "assistant_version": BUILD_VERSION}
                # Empty text is a failure too — record the real reason and move
                # on to the next provider instead of returning a blank reply.
                _record_llm_error(provider, "provider returned empty text")
                print(f"[assistant] {provider} returned empty text — trying next provider/KB")
            except Exception as e:
                # Record the REAL reason for every provider so the user-facing
                # fallback message and /api/assistant/diagnostics can show it,
                # and log it to Render (never hidden behind a generic message).
                _record_llm_error(provider, str(e))
                print(f"[assistant] {provider} LLM call FAILED — trying next provider/KB: {e}")
                traceback.print_exc()
        print("[assistant] all configured providers failed")
    else:
        print("[assistant] no usable AI provider (no valid key found) — "
              "check OPENAI_API_KEY / GEMINI_API_KEY / ANTHROPIC_API_KEY")
    print("[assistant] Using Knowledge Base fallback")
    result = _kb_fallback_answer(last_question, lang, provider_configured=bool(providers))
    result["assistant_version"] = BUILD_VERSION
    return result


def _mask_key(key: str) -> str:
    """Show only enough of a key to confirm WHICH key is set, never the secret.
    e.g. 'sk-proj-ABCD…WXYZ (44 chars)'."""
    if not key:
        return ""
    if len(key) <= 12:
        return f"{key[:3]}… ({len(key)} chars)"
    return f"{key[:7]}…{key[-4:]} ({len(key)} chars)"


def _default_model_for(provider: str) -> str:
    return {"anthropic": DEFAULT_ANTHROPIC_MODEL, "openai": DEFAULT_OPENAI_MODEL,
            "gemini": DEFAULT_GEMINI_MODEL}.get(provider)


def diagnostics(lang: str = "ar") -> dict:
    """v15.7 (admin): full live health check. Reports, for EACH provider,
    whether its key was found (masked) and — for every provider in the
    configured order — actually calls it with a tiny probe and records the
    real outcome (ok / HTTP status / error). Also reports the provider order
    and where it came from (env var vs default), and whether the Knowledge
    Base would be used. Directly answers "which provider was tried, what
    status, and was the KB used?". Never raises."""
    order = _provider_order()
    order_source = "env:LLM_PROVIDER_ORDER" if os.environ.get("LLM_PROVIDER_ORDER") else "default"
    usable = _providers_in_order()

    # Per-provider key visibility (masked) — answers "was the key found?"
    keys = {
        "openai": {"found": bool(_env_key("OPENAI_API_KEY")),
                   "masked": _mask_key(_env_key("OPENAI_API_KEY")),
                   "source_var": "OPENAI_API_KEY"},
        "gemini": {"found": bool(_env_key("GEMINI_API_KEY", "GOOGLE_API_KEY")),
                   "masked": _mask_key(_env_key("GEMINI_API_KEY", "GOOGLE_API_KEY")),
                   "source_var": "GEMINI_API_KEY" if _clean_key(os.environ.get("GEMINI_API_KEY"))
                   else ("GOOGLE_API_KEY" if _clean_key(os.environ.get("GOOGLE_API_KEY")) else None)},
        "anthropic": {"found": bool(_env_key("ANTHROPIC_API_KEY")),
                      "masked": _mask_key(_env_key("ANTHROPIC_API_KEY")),
                      "source_var": "ANTHROPIC_API_KEY"},
    }

    probe = [{"role": "user", "content": "قل: تم" if lang == "ar" else "Reply with: OK"}]
    sys_prompt = _system_prompt(lang, "")
    callers = {"anthropic": _call_anthropic, "openai": _call_openai, "gemini": _call_gemini}

    attempts = []          # every provider we actually tried, in order
    overall_ok = False
    answering_provider = None
    for provider in order:
        if not keys[provider]["found"]:
            attempts.append({"provider": provider, "tried": False,
                             "reason": "no API key set for this provider"})
            continue
        model = os.environ.get("LLM_MODEL") or _default_model_for(provider)
        rec = {"provider": provider, "tried": True, "model": model,
               "gemini_models": _gemini_models_to_try() if provider == "gemini" else None}
        try:
            text = callers[provider](sys_prompt, probe)
            if text:
                rec["ok"] = True
                rec["status"] = 200
                rec["sample"] = text[:80]
                attempts.append(rec)
                overall_ok = True
                answering_provider = provider
                break   # this is the one that would answer real chats
            rec["ok"] = False
            rec["status"] = None
            rec["error"] = "provider returned empty text"
            _record_llm_error(provider, "provider returned empty text")
        except Exception as e:
            rec["ok"] = False
            # pull an HTTP status out of the message if present (e.g. "HTTP 401: ...")
            m = re.search(r"HTTP (\d{3})", str(e))
            rec["status"] = int(m.group(1)) if m else None
            rec["error"] = str(e)[:400]
            _record_llm_error(provider, str(e))
        attempts.append(rec)

    return {
        "ok": overall_ok,
        "assistant_version": BUILD_VERSION,
        "module_file": __file__,
        "answering_provider": answering_provider,
        "would_use_knowledge_base": not overall_ok,
        "provider_order": order,
        "provider_order_source": order_source,
        "usable_providers": usable,
        "keys": keys,
        "providers_configured": providers_configured(),
        "attempts": attempts,
        "kb_topics": len(knowledge_base.KB),
        "last_error": last_llm_error(),
        "detail": ("Live test succeeded via " + answering_provider) if overall_ok else (
            "No provider answered — real chats will use the Knowledge Base. "
            "See 'attempts' for the status/error of each provider."),
    }


def status() -> dict:
    provider = llm_configured()
    default_model = {"anthropic": DEFAULT_ANTHROPIC_MODEL, "openai": DEFAULT_OPENAI_MODEL,
                      "gemini": DEFAULT_GEMINI_MODEL}.get(provider)
    return {"llm_provider": provider,
            "assistant_version": BUILD_VERSION,
            "llm_model": os.environ.get("LLM_MODEL") or default_model,
            "llm_providers_configured": providers_configured(),
            "kb_topics": len(knowledge_base.KB)}
