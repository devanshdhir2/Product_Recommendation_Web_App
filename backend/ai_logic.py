# backend/ai_logic.py
# RAG core for FurniFind (Render-friendly, low RAM)
# - Embeddings via HF Inference API /models (no deprecated /pipeline)
# - Text generation via HF Inference API /models
# - Pinecone data-plane connection (host + api_key)
# - Pads text embeddings (384) to 2432 (384 + 2048 image)

import os
import re
import json
import httpx
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, Index


# -------------------------
# Environment & constants
# -------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "product-recommendations")
PINECONE_HOST = os.getenv("PINECONE_HOST", "")  # if empty, we'll resolve via control plane

HF_TOKEN = os.getenv("HF_TOKEN", "")  # required for private/gated models
HF_EMB_MODEL = os.getenv("HF_EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_TEXT_MODEL = os.getenv("HF_TEXT_MODEL", "google/gemma-2b-it")  # you can change to Qwen if you prefer

# Embedding dims: 384 text + 2048 image = 2432 index
TEXT_DIM, IMG_DIM, INDEX_DIM = 384, 2048, 2432

# HF /models endpoints (router first, fallback to api-inference)
HF_EMB_URLS = [
    f"https://router.huggingface.co/hf-inference/models/{HF_EMB_MODEL}",
    f"https://api-inference.huggingface.co/models/{HF_EMB_MODEL}",
]
HF_GEN_URLS = [
    f"https://router.huggingface.co/hf-inference/models/{HF_TEXT_MODEL}",
    f"https://api-inference.huggingface.co/models/{HF_TEXT_MODEL}",
]

# Filters to keep furniture; exclude small parts/repair bits
BAD = ("lever", "latch", "cable", "release", "hardware", "bracket", "replacement",
       "webbing", "band", "repair", "modification")
GOOD = ("sofa", "chair", "ottoman", "bench", "couch", "table", "tray", "armchair", "stool")


# -------------------------
# Pinecone index (data plane)
# -------------------------
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY missing")

_pc = Pinecone(api_key=PINECONE_API_KEY)
if not PINECONE_HOST:
    # Resolve host from control plane
    desc = _pc.describe_index(PINECONE_INDEX_NAME)
    PINECONE_HOST = desc.host

index: Index = Index(host=PINECONE_HOST, api_key=PINECONE_API_KEY)


# -------------------------
# HF helpers
# -------------------------
def _auth_headers() -> Dict[str, str]:
    # Token is optional for fully public models; required for gated/private ones (e.g., Gemma 2B IT)
    return {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}


def _hf_post(url: str, payload: Dict[str, Any], timeout_s: int = 45) -> Optional[Any]:
    headers = {"Content-Type": "application/json", **_auth_headers()}
    with httpx.Client(timeout=timeout_s) as client:
        r = client.post(url, headers=headers, json=payload)
        if r.status_code != 200:
            # Return None to allow caller to try fallback URL
            return None
        return r.json()


# -------------------------
# Embeddings (HF Inference /models)
# -------------------------
def _hf_embed(text: str) -> List[float]:
    """Calls HF Inference API /models for sentence-transformers; returns a pooled vector."""
    payload = {
        "inputs": text,
        "options": {"wait_for_model": True}
    }
    data = None
    for url in HF_EMB_URLS:
        data = _hf_post(url, payload)
        if data is not None:
            break
    if data is None:
        raise RuntimeError("HF embedding endpoint unavailable")

    # Possible shapes:
    # - [float, float, ...] pooled vector
    # - [[float,...],[float,...], ...] token vectors (need mean pool)
    if isinstance(data, list) and data and isinstance(data[0], float):
        vec = data
    elif isinstance(data, list) and data and isinstance(data[0], list):
        vec = np.mean(np.array(data, dtype="float32"), axis=0).tolist()
    else:
        raise RuntimeError("Unexpected embedding response format")

    if len(vec) != TEXT_DIM:
        # If the model wasn't the expected size, fail clearly
        raise RuntimeError(f"Unexpected embedding dim={len(vec)}, expected {TEXT_DIM}")
    return [float(x) for x in vec]


def encode_query_mm(q: str, w_text: float = 1.0) -> List[float]:
    """Encode text and pad with zeros to match the 2432-dim index."""
    v = _hf_embed(q)
    v = [w_text * x for x in v]
    return v + [0.0] * IMG_DIM


# -------------------------
# Pinecone search
# -------------------------
def search(query: str, top_k: int = 8, w_text: float = 1.0, filt: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    qvec = encode_query_mm(query, w_text=w_text)
    res = index.query(vector=qvec, top_k=top_k, include_metadata=True, filter=filt or {})
    rows = []
    for m in res.get("matches", []):
        md = m.get("metadata", {}) or {}
        rows.append({
            "id": m.get("id"),
            "score": float(m.get("score", 0.0)),
            "title": md.get("title"),
            "brand": md.get("brand"),
            "price": md.get("price"),
            "image_url": md.get("image_url"),
        })
    return pd.DataFrame(rows)


# -------------------------
# Result filtering
# -------------------------
def filter_hits(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    t = df["title"].fillna("").str.lower()
    keep = t.apply(lambda x: any(g in x for g in GOOD) and not any(b in x for b in BAD))
    df2 = df[keep].copy()
    if len(df2) >= 2:
        return df2.head(5)
    util = t.str.contains(r"(tray|table|ottoman|stool)", regex=True) & ~t.apply(lambda x: any(b in x for b in BAD))
    extra = df[util].copy()
    out = pd.concat([df2, extra]).drop_duplicates(subset=["id"])
    return out.head(5) if len(out) else df.head(5)


# -------------------------
# Text generation (HF Inference /models)
# -------------------------
def _hf_generate(prompt: str, max_new_tokens: int = 240, temperature: float = 0.2) -> str:
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "return_full_text": False,
            "repetition_penalty": 1.05
        },
        "options": {"wait_for_model": True}
    }
    data = None
    for url in HF_GEN_URLS:
        data = _hf_post(url, payload, timeout_s=60)
        if data is not None:
            break
    if data is None:
        raise RuntimeError("HF text-generation endpoint unavailable")

    # Typical shapes:
    # - [{"generated_text": "..."}]
    # - {"generated_text": "..."}
    # - "..."  (rare)
    if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
        return str(data[0]["generated_text"])
    if isinstance(data, dict) and "generated_text" in data:
        return str(data["generated_text"])
    if isinstance(data, str):
        return data
    return ""


def gemma_answer(query: str, df_hits: pd.DataFrame, max_new_tokens: int = 240) -> str:
    titles = [str(t).strip() for t in df_hits.get("title", []).fillna("").tolist() if str(t).strip()]
    ctx_lines = []
    for _, r in df_hits.head(5).iterrows():
        ctx_lines.append(f"- {r.get('title','N/A')} (Brand: {r.get('brand','N/A')}, Price: {r.get('price','N/A')})")
    ctx = "\n".join(ctx_lines) if ctx_lines else "No context."

    prompt = (
        "You are a concise, helpful product recommendation assistant.\n"
        "Rules (follow strictly):\n"
        "- Never say you cannot answer; if the exact keyword is missing, pick the closest relevant items from the context and still answer.\n"
        "- Do NOT start with 'Sure', 'Okay', or 'Here is/Here’s'. No emojis or meta-chat.\n"
        "- Write exactly ONE paragraph of 4–6 sentences. Start neutrally (not a brand).\n"
        "- Mention at least two product titles exactly as in the context. Use only the context.\n\n"
        f"Context:\n{ctx}\n\n"
        f"User need:\n{query}\n\n"
        "Write now."
    )

    raw = _hf_generate(prompt, max_new_tokens=max_new_tokens, temperature=0.0).strip()

    # sanitize openers + meta
    txt = re.sub(r"^(sure,?\s*|okay,?\s*|here'?s\s+.*?:\s*)", "", raw, flags=re.I).strip()
    for b in [r"i cannot answer", r"i can't", r"unable", r"not mention", r"no context",
              r"i'm here to assist", r"would you like", r"let me know", r"please note",
              r"i hope this helps", r"[😊😁🙂😉👍]"]:
        txt = re.sub(b, "", txt, flags=re.I)
    txt = re.sub(r"\s+", " ", txt).strip()

    # ensure ≥2 titles mentioned
    needed = []
    for t in titles[:3]:
        if t and t not in txt:
            needed.append(t)
        if len(needed) >= 2:
            break
    if needed:
        txt += " In particular, consider " + " and ".join(f'\"{n}\"' for n in needed[:2]) + "."

    # clamp to 4–6 sentences
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", txt) if s.strip()]
    while len(sents) < 4:
        sents.append("These options balance comfort, value, and everyday usability at home.")
    txt = " ".join(sents[:6])

    # refusal guard
    if re.search(r"(cannot|can't|unable|no context|not mention)", txt, re.I) or sum(1 for t in titles if t in txt) < 2:
        picks = titles[:3]
        if len(picks) >= 2:
            base = (
                "These options offer practical seating and storage for living spaces. "
                f"\"{picks[0]}\" and \"{picks[1]}\" stand out for their everyday comfort and value"
                + (f", while \"{picks[2]}\" adds a versatile accent." if len(picks) > 2 else ".")
            )
        elif len(picks) == 1:
            base = f"\"{picks[0]}\" is a practical choice for compact living spaces with solid everyday value."
        else:
            base = "These options balance comfort, value, and everyday usability at home."
        txt = base

    return txt


# -------------------------
# Small query expansion
# -------------------------
def _expand_query(q: str) -> str:
    ql = q.lower()
    if "sofa" in ql:
        return "sofa couch chair ottoman bench living room seating"
    return q


# -------------------------
# Public API
# -------------------------
def rag(query: str, top_k: int = 8) -> Dict[str, Any]:
    raw = search(_expand_query(query), top_k=top_k)
    used = filter_hits(raw)
    text = gemma_answer(query, used)
    return {"recommendations": used.to_dict(orient="records"), "generated_text": text}


def healthcheck() -> Dict[str, Any]:
    # lightweight health: test index stats and HF reachability (embed a tiny string)
    stats = index.describe_index_stats()
    ok_hf = True
    try:
        _ = _hf_embed("ping")
    except Exception:
        ok_hf = False
    return {
        "pinecone_ok": True if stats else False,
        "hf_ok": ok_hf,
        "index_total": int(stats.get("total_vector_count", 0)) if isinstance(stats, dict) else None,
        "index_dim": INDEX_DIM
    }
