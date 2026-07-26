# -*- coding: utf-8 -*-
"""Cross-source de-duplication (v15).

Same-platform reposts are already handled by external_id (each platform's
own item id — see external_sources.py / google_maps_source.py). This module
catches the harder case the spec asks for: the SAME opinion appearing on
DIFFERENT platforms (e.g. a review posted on Google Maps and also quoted in
a Reddit thread) — merging them instead of double-counting.

Honest scope: this is text-similarity de-duplication (near-identical
wording after normalization), not semantic/paraphrase de-duplication —
doing that reliably needs an embeddings model, which is out of scope for a
free-tier deployment. It reliably catches copy-pasted/cross-posted text and
near-identical translations; it will NOT catch two people independently
describing the same event in very different words.

fingerprint(text) -> a stable hash of the text's 4-word shingle set, used as
a fast first pass (exact match = same fingerprint). is_near_duplicate() adds
a fuzzy fallback (difflib) against a small candidate pool for near-misses.
"""
import hashlib
import re
from difflib import SequenceMatcher

SIMILARITY_THRESHOLD = 0.87
SHINGLE_SIZE = 4


def normalize(text: str) -> str:
    s = (text or "").lower()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s, flags=re.UNICODE).strip()
    return s


def fingerprint(text: str) -> str:
    """Order-independent fingerprint: the sorted set of 4-word shingles,
    hashed. Two texts with the same shingle SET (e.g. identical or
    lightly-reordered copies) get the same fingerprint."""
    norm = normalize(text)
    words = norm.split()
    if len(words) < SHINGLE_SIZE:
        core = norm
    else:
        shingles = {" ".join(words[i:i + SHINGLE_SIZE]) for i in range(len(words) - SHINGLE_SIZE + 1)}
        core = "|".join(sorted(shingles))
    return hashlib.sha1(core.encode("utf-8")).hexdigest()


def is_near_duplicate(text: str, candidates) -> bool:
    """candidates: iterable of existing normalized text strings (already
    run through normalize()). Bound the candidate list before calling this
    — it's O(n) SequenceMatcher comparisons, meant for a small recent
    window, not the whole table."""
    norm = normalize(text)
    if not norm:
        return False
    for cand in candidates:
        if not cand:
            continue
        if abs(len(cand) - len(norm)) > max(20, len(norm) * 0.3):
            continue  # quick length-based skip before the expensive comparison
        if SequenceMatcher(None, norm, cand).ratio() >= SIMILARITY_THRESHOLD:
            return True
    return False
