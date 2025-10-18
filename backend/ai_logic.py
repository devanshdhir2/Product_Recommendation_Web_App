import os, json, re, logging
from typing import List, Dict, Any
import numpy as np
import httpx
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, Index

# ---------- logging ----------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ai")

# ---------- env ----------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_HOST = os.getenv("PINECONE_HOST", "")  # full data-plane URL (https://....pinecone.io)
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_MODEL = os.getenv("HF_MODEL", "google/gemma-2b-it")  # you can swap later without redeploy
TIMEOUT_S = float(os.getenv("HF_TIMEOUT", "18.0"))

if not (PINECONE_API_KEY and PINECONE_HOST and PINECONE_INDEX_NAME):
    log.error("Missing Pinecone envs. Check PINECONE_API_KEY / PINECONE_HOST / PINECONE_INDEX_NAME")

# ---------- clients ----------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = Index(host=PINECONE_HOST)

# MiniLM text encoder
_enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
TEXT_DIM, IMG_DIM = 384, 2048  # matches your Kaggle build (dense + zero-padded img)

def _encode_query_mm(q: str, w_text: float = 1.0) -> List[float]:
    v = _enc.encode(q, normalize_embeddings=True).tolist()
    v = [w_text * x for x in v]
    return v + [0.0] * IMG_DIM

def search(query: str, top_k: int = 8, w_text: float = 1.0, filt: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    qvec = _encode_query_mm(query, w_text=w_text)
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
    return rows

# --- filtering like your notebook ---
BAD = ("lever","latch","cable","release","hardware","bracket","replacement","webbing","band","repair","modification")
GOOD = ("sofa","chair","ottoman","bench","couch","table","tray","armchair","stool")

def filter_hits(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows: 
        return rows
    def keep_ok(t: str) -> bool:
        t = (t or "").lower()
        return any(g in t for g in GOOD) and not any(b in t for b in BAD)
    main = [r for r in rows if keep_ok(r.get("title",""))]
    if len(main) >= 2:
        return main[:5]
    util = [r for r in rows if re.search(r"(tray|table|ottoman|stool)", (r.get("title") or "").lower()) and keep_ok(r.get("title",""))]
    out = []
    seen = set()
    for r in main + util + rows:
        if r.get("id") not in seen:
            out.append(r); seen.add(r.get("id"))
        if len(out) >= 5: break
    return out

# ---------- HF Inference (no GPU) ----------
_HF_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
_HDRS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

def _hf_generate(prompt: str, max_new_tokens: int = 220) -> str:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN missing or not set on server")
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "temperature": 0.0,
            "repetition_penalty": 1.05,
            "return_full_text": False
        }
    }
    with httpx.Client(timeout=TIMEOUT_S) as cx:
        r = cx.post(_HF_URL, headers=_HDRS, json=payload)
    if r.status_code == 200:
        out = r.json()
        if isinstance(out, list) and out and "generated_text" in out[0]:
            return (out[0]["generated_text"] or "").strip()
        if isinstance(out, dict) and "generated_text" in out:
            return (out["generated_text"] or "").strip()
        return ""
    # log & raise to trigger fallback
    try:
        log.error("HF error %s: %s", r.status_code, r.text[:400])
    except Exception:
        pass
    raise RuntimeError(f"HF inference failed {r.status_code}")

def _fallback_write(query: str, titles: List[str]) -> str:
    # deterministic 4–6 sentences; no external call
    picks = [t for t in titles if t][:3]
    if len(picks) >= 2:
        base = (
            f"These options offer practical seating and storage for living spaces. "
            f"\"{picks[0]}\" and \"{picks[1]}\" stand out for everyday comfort and value"
            + (f", while \"{picks[2]}\" adds a versatile accent." if len(picks) > 2 else ".")
        )
    elif len(picks) == 1:
        base = f"\"{picks[0]}\" is a reliable pick for compact rooms with solid day-to-day usability."
    else:
        base = "These picks balance comfort, price, and everyday usability."
    return base

def _build_prompt(query: str, rows: List[Dict[str, Any]]) -> str:
    ctx_lines = [f"- {r.get('title','N/A')} (Brand: {r.get('brand','N/A')}, Price: {r.get('price','N/A')})" for r in rows[:5]]
    ctx = "\n".join(ctx_lines) if ctx_lines else "No context."
    return (
        "You are a concise, helpful product recommendation assistant.\n"
        "Rules (follow strictly):\n"
        "- Never refuse; if the exact keyword is missing, pick the closest relevant items from the context and still answer.\n"
        "- Do NOT start with 'Sure', 'Okay', or 'Here is/Here’s'. No emojis or meta-chat.\n"
        "- Write exactly ONE paragraph of 4–6 sentences. Start neutrally (not a brand).\n"
        "- Mention at least two product titles exactly as in the context. Use only the context.\n\n"
        f"Context:\n{ctx}\n\n"
        f"User need:\n{query}\n\n"
        "Write now."
    )

def gemma_like_answer(query: str, used: List[Dict[str, Any]]) -> str:
    titles = [r.get("title","") for r in used]
    prompt = _build_prompt(query, used)
    text = ""
    try:
        text = _hf_generate(prompt, max_new_tokens=220)
    except Exception as e:
        log.warning("Falling back text gen: %s", e)
        text = _fallback_write(query, titles)

    # sanitize + ensure 4–6 sentences
    text = re.sub(r"^(sure,?\s*|okay,?\s*|here'?s\s+.*?:\s*)", "", text, flags=re.I).strip()
    for b in [r"i cannot answer", r"i can't", r"unable", r"i'm here to assist", r"would you like", r"please note",
              r"i hope this helps", r"[😊😁🙂😉👍]"]:
        text = re.sub(b, "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sents) < 4:
        sents += ["These options balance comfort, value, and everyday usability at home."] * (4 - len(sents))
    return " ".join(sents[:6])

def rag(query: str, top_k: int = 8) -> Dict[str, Any]:
    # query expansion
    qx = "sofa couch chair ottoman bench living room seating" if "sofa" in query.lower() else query
    raw = search(qx, top_k=top_k)
    used = filter_hits(raw)
    generated = gemma_like_answer(query, used)
    return {"recommendations": used, "generated_text": generated}

def healthcheck() -> Dict[str, Any]:
    try:
        stats = index.describe_index_stats()
    except Exception as e:
        return {"ok": False, "pinecone": f"ERR: {e}"}
    hf_ok = bool(HF_TOKEN)
    return {"ok": True, "pinecone": stats, "hf_token": hf_ok, "model": HF_MODEL}
