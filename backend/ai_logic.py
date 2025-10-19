import os, re, json, time
from typing import List, Dict, Any
import pandas as pd
import httpx

from fastembed import TextEmbedding
from pinecone import Pinecone, Index

# ------------ ENV ------------
PINECONE_API_KEY   = os.getenv("PINECONE_API_KEY", "")
PINECONE_HOST      = os.getenv("PINECONE_HOST", "")
PINECONE_INDEX     = os.getenv("PINECONE_INDEX_NAME", "product-recommendations")

HF_TOKEN           = os.getenv("HF_TOKEN", "")
HF_MODEL_ID        = os.getenv("HF_MODEL_ID", "google/gemma-2b-it")  # you can change via Render env
HF_URL             = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"

FRONTEND_ORIGIN    = os.getenv("FRONTEND_ORIGIN", "")

# ------------ PINECONE ------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index: Index = Index(host=PINECONE_HOST)

# ------------ EMBEDDINGS (fast, low-RAM) ------------
# 384-dim model, drop-in replacement for all-MiniLM
_EMB = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")  # tiny + fast
TEXT_DIM, IMG_DIM = 384, 2048

def encode_query_mm(q: str, w_text: float = 1.0) -> List[float]:
    # fastembed returns a generator
    vec = next(_EMB.embed([q], normalize=True))
    vec = [w_text * float(x) for x in vec]
    return vec + [0.0] * IMG_DIM  # keep index dimension 2432

# ------------ SEARCH ------------
def search(query: str, top_k: int = 8, w_text: float = 1.0, filt: dict | None = None) -> pd.DataFrame:
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

# ------------ FILTER ------------
_BAD = ("lever","latch","cable","release","hardware","bracket","replacement","webbing","band","repair","modification")
_GOOD = ("sofa","chair","ottoman","bench","couch","table","tray","armchair","stool")

def filter_hits(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: 
        return df
    t = df["title"].fillna("").str.lower()
    keep = t.apply(lambda x: any(g in x for g in _GOOD) and not any(b in x for b in _BAD))
    df2 = df[keep].copy()
    if len(df2) >= 2:
        return df2.head(8)
    util = t.str.contains(r"(tray|table|ottoman|stool)", regex=True) & ~t.apply(lambda x: any(b in x for b in _BAD))
    extra = df[util].copy()
    out = pd.concat([df2, extra]).drop_duplicates(subset=["id"])
    return out.head(8) if len(out) else df.head(8)

# ------------ GENERATION (HF Inference API; no local model) ------------
def _hf_generate(prompt: str, max_new_tokens: int = 240) -> str:
    if not HF_TOKEN:
        return ""
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.2,
            "repetition_penalty": 1.05,
            "return_full_text": False
        }
    }
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    try:
        with httpx.Client(timeout=30) as client:
            r = client.post(HF_URL, headers=headers, json=payload)
            if r.status_code == 200:
                data = r.json()
                # HF response can be either list[{"generated_text": ...}] or dict
                if isinstance(data, list) and data and "generated_text" in data[0]:
                    return str(data[0]["generated_text"])
                if isinstance(data, dict) and "generated_text" in data:
                    return str(data["generated_text"])
            # fallthrough -> empty
    except Exception:
        pass
    return ""

def gemma_answer(query: str, df_hits: pd.DataFrame, max_new_tokens: int = 220) -> str:
    titles = [str(t).strip() for t in df_hits.get("title", []).fillna("").tolist() if str(t).strip()]
    ctx_lines = []
    for _, r in df_hits.head(6).iterrows():
        ctx_lines.append(f"- {r.get('title','N/A')} (Brand: {r.get('brand','N/A')}, Price: {r.get('price','N/A')})")
    ctx = "\n".join(ctx_lines) if ctx_lines else "No context."

    prompt = (
        "You are a concise, helpful product recommendation assistant.\n"
        "Rules (follow strictly):\n"
        "- Never say you cannot answer; if the exact keyword is missing, pick the closest relevant items from the context (chairs/sofas/ottomans/benches/trays) and still answer.\n"
        "- Do NOT start with 'Sure', 'Okay', or 'Here is/Here’s'. No emojis or meta-chat.\n"
        "- Write exactly ONE paragraph of 4–6 sentences. Start neutrally (not a brand).\n"
        "- Mention at least two product titles exactly as in the context. Use only the context.\n\n"
        f"Context:\n{ctx}\n\n"
        f"User need:\n{query}\n\n"
        "Write now."
    )

    out = _hf_generate(prompt, max_new_tokens=max_new_tokens)
    txt = (out or "").strip()

    # sanitize openers + meta
    txt = re.sub(r"^(sure,?\s*|okay,?\s*|here'?s\s+.*?:\s*)", "", txt, flags=re.I).strip()
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

    # refusal/fallback: deterministic summary if API cold/blocked
    if len(titles) and (len(txt) < 20 or sum(1 for t in titles if t in txt) < 2):
        picks = titles[:3]
        if len(picks) >= 2:
            txt = (
                "These options offer practical seating and storage for living spaces. "
                f"\"{picks[0]}\" and \"{picks[1]}\" stand out for their everyday comfort and value"
                + (f", while \"{picks[2]}\" adds a versatile accent." if len(picks) > 2 else ".")
            )
        else:
            txt = "These picks balance comfort, price, and everyday usability for compact spaces."
    return txt

# ------------ RAG ------------
def _expand_query(q: str) -> str:
    ql = q.lower()
    if "sofa" in ql:
        return "sofa couch chair ottoman bench living room seating"
    return q

def rag(query: str, top_k: int = 8) -> Dict[str, Any]:
    raw = search(_expand_query(query), top_k=top_k)
    used = filter_hits(raw)
    text = gemma_answer(query, used)
    return {"recommendations": used.to_dict(orient="records"), "generated_text": text}

# ------------ HEALTH ------------
def healthcheck() -> Dict[str, Any]:
    try:
        s = index.describe_index_stats()
        return {"ok": True, "index": PINECONE_INDEX, "total_vector_count": s.get("total_vector_count", 0)}
    except Exception as e:
        return {"ok": False, "error": str(e)}
