import os, math, json, time
import numpy as np
import pandas as pd
import httpx
from typing import List, Dict, Any, Optional
from pinecone import Pinecone

# ---------- env ----------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "product-recommendations")
PINECONE_HOST = os.getenv("PINECONE_HOST", "")  # data-plane host strongly recommended
HF_TOKEN = os.getenv("HF_TOKEN", "")

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GEN_MODEL = os.getenv("GEN_MODEL", "google/gemma-2b-it")

TEXT_DIM, IMG_DIM = 384, 2048
TIMEOUT = httpx.Timeout(30.0, connect=15.0)
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# ---------- pinecone client ----------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_HOST) if PINECONE_HOST else pc.Index(PINECONE_INDEX_NAME)

# ---------- HF helpers ----------
def _hf_feature_extraction(text: str) -> Optional[List[float]]:
    """Mean-pool token embeddings from HF Inference API (feature-extraction)."""
    if not text.strip():
        return [0.0] * TEXT_DIM
    url = f"https://api-inference.huggingface.co/models/{EMBED_MODEL}"
    payload = {"inputs": text, "options": {"wait_for_model": True}}
    for attempt in range(3):
        try:
            with httpx.Client(timeout=TIMEOUT) as client:
                r = client.post(url, headers=HEADERS, json=payload)
                if r.status_code == 200:
                    arr = r.json()
                    # arr can be [seq_len, hidden] or nested lists for batch
                    vec = np.array(arr, dtype=np.float32)
                    if vec.ndim == 2 and vec.shape[1] == TEXT_DIM:
                        # mean-pool over tokens
                        pooled = vec.mean(axis=0)
                        # L2 normalize (standard for cosine)
                        norm = np.linalg.norm(pooled) + 1e-12
                        return (pooled / norm).tolist()
                    # batch case: take first
                    if vec.ndim == 3 and vec.shape[-1] == TEXT_DIM:
                        pooled = vec[0].mean(axis=0)
                        norm = np.linalg.norm(pooled) + 1e-12
                        return (pooled / norm).tolist()
                elif r.status_code in (503, 429):
                    time.sleep(1.5 * (attempt + 1))
                else:
                    break
        except Exception:
            time.sleep(1.0)
    return None

def _hf_text_generate(prompt: str, max_new_tokens: int = 220) -> str:
    """Gemma 2B IT via HF Inference API (text-generation)."""
    url = f"https://api-inference.huggingface.co/models/{GEN_MODEL}"
    params = {
        "max_new_tokens": max_new_tokens,
        "temperature": 0.2,
        "repetition_penalty": 1.05,
        "return_full_text": False
    }
    payload = {"inputs": prompt, "parameters": params, "options": {"wait_for_model": True}}
    for attempt in range(3):
        try:
            with httpx.Client(timeout=TIMEOUT) as client:
                r = client.post(url, headers=HEADERS, json=payload)
                if r.status_code == 200:
                    data = r.json()
                    # HF may return list of dicts with "generated_text"
                    if isinstance(data, list) and data and "generated_text" in data[0]:
                        return str(data[0]["generated_text"]).strip()
                    # Fallback to raw string
                    if isinstance(data, str):
                        return data.strip()
                elif r.status_code in (503, 429):
                    time.sleep(1.5 * (attempt + 1))
                else:
                    break
        except Exception:
            time.sleep(1.0)
    return ""

# ---------- encode & search ----------
def encode_query_mm(q: str, w_text: float = 1.0) -> List[float]:
    emb = _hf_feature_extraction(q) or [0.0] * TEXT_DIM
    emb = [w_text * x for x in emb]
    return emb + [0.0] * IMG_DIM

def search(query: str, top_k: int = 8, w_text: float = 1.0, filt: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    vec = encode_query_mm(query, w_text=w_text)
    res = index.query(vector=vec, top_k=top_k, include_metadata=True, filter=filt or {})
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

# ---------- filter (keep furniture; drop hardware) ----------
_BAD = ("lever","latch","cable","release","hardware","bracket","replacement","webbing","band","repair","modification")
_GOOD = ("sofa","chair","ottoman","bench","couch","table","tray","armchair","stool","storage")

def filter_hits(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    t = df["title"].fillna("").str.lower()
    keep = t.apply(lambda x: any(g in x for g in _GOOD) and not any(b in x for b in _BAD))
    df2 = df[keep].copy()
    if len(df2) >= 2:
        return df2.head(5)
    util = t.str.contains(r"(tray|table|ottoman|stool|storage)", regex=True) & ~t.apply(lambda x: any(b in x for b in _BAD))
    more = df[util].copy()
    out = pd.concat([df2, more]).drop_duplicates(subset=["id"])
    return out.head(5) if len(out) else df.head(5)

# ---------- generation ----------
def _prompt_from_context(query: str, df_hits: pd.DataFrame) -> str:
    ctx_lines = []
    for _, r in df_hits.head(5).iterrows():
        ctx_lines.append(f'- {r.get("title","N/A")} (Brand: {r.get("brand","N/A")}, Price: {r.get("price","N/A")})')
    ctx = "\n".join(ctx_lines) if ctx_lines else "No context."
    return (
        "You are a concise, helpful product recommendation assistant.\n"
        "Rules:\n"
        "- Never refuse; if the exact term is missing, pick the closest items from the context (chairs/sofas/ottomans/benches/trays) and still answer.\n"
        "- Do NOT start with 'Sure', 'Okay', or 'Here is'. No emojis. One paragraph of 4–6 sentences.\n"
        "- Mention at least two product titles exactly as in the context. Use only the context.\n\n"
        f"Context:\n{ctx}\n\n"
        f"User need:\n{query}\n\n"
        "Write now."
    )

def gemma_answer(query: str, df_hits: pd.DataFrame, max_new_tokens: int = 220) -> str:
    prompt = _prompt_from_context(query, df_hits)
    txt = _hf_text_generate(prompt, max_new_tokens=max_new_tokens).strip()
    # simple sanitation
    if not txt:
        titles = [str(t).strip() for t in df_hits.get("title", []).fillna("").tolist() if str(t).strip()]
        if len(titles) >= 2:
            return (f"These options offer practical seating and storage for living spaces. "
                    f"\"{titles[0]}\" and \"{titles[1]}\" stand out for everyday comfort and value.")
        return "These options balance comfort, value, and everyday usability at home."
    return " ".join(txt.split())

# ---------- public RAG ----------
def rag(query: str, top_k: int = 8) -> Dict[str, Any]:
    raw = search(query, top_k=top_k)
    used = filter_hits(raw)
    text = gemma_answer(query, used)
    return {"recommendations": used.to_dict(orient="records"), "generated_text": text}

# ---------- analytics ----------
def analytics_from_csv(csv_path: str = "cleaned_intern_data.csv") -> Dict[str, Any]:
    df = pd.read_csv(csv_path)
    brand_counts = df["brand"].value_counts().nlargest(10).to_dict()
    avg_price = df.groupby("brand")["price"].mean().nlargest(10).round(2).to_dict()
    return {"products_per_brand": brand_counts, "avg_price_per_brand": avg_price}
