import os
import re
import math
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
import requests
from pinecone import Pinecone

load_dotenv()

# --- ENV ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "product-recommendations")
PINECONE_HOST = os.getenv("PINECONE_HOST", "").strip()
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_BASE_URL = os.getenv("HF_BASE_URL", "https://router.huggingface.co").rstrip("/")
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "google/gemma-2-2b-it:nebius")
EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
TEXT_DIM = int(os.getenv("TEXT_DIM", "384"))
IMG_DIM = int(os.getenv("IMG_DIM", "2048"))
TOP_K_DEFAULT = int(os.getenv("TOP_K_DEFAULT", "8"))

if not PINECONE_API_KEY:
    raise RuntimeError("Missing PINECONE_API_KEY")
if not HF_TOKEN:
    raise RuntimeError("Missing HF_TOKEN")

# --- CLIENTS ---
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_HOST) if PINECONE_HOST else pc.Index(PINECONE_INDEX_NAME)

http = requests.Session()
http.headers.update({
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
})

# --- HELPERS ---
BAD = ("lever","latch","cable","release","hardware","bracket","replacement","webbing","band","repair","modification")
GOOD = ("sofa","chair","ottoman","bench","couch","table","tray","armchair","stool")

def _normalize_titles(rows: List[Dict[str, Any]]) -> List[str]:
    t = []
    for r in rows:
        s = str(r.get("title", "")).strip()
        if s:
            t.append(s)
    return t

def _expand_query(q: str) -> str:
    ql = q.lower()
    if "sofa" in ql:
        return "sofa couch chair ottoman bench living room seating"
    return q

def _l2norm(vec: List[float]) -> List[float]:
    n = math.sqrt(sum(x*x for x in vec)) or 1.0
    return [x / n for x in vec]

# --- HF ROUTER CALLS ---
def _hf_embed(text: str) -> List[float]:
    """
    Uses HF Router OpenAI-compatible embeddings endpoint.
    POST {HF_BASE_URL}/v1/embeddings
    payload: {"model": EMBED_MODEL_ID, "input": text}
    """
    url = f"{HF_BASE_URL}/v1/embeddings"
    payload = {"model": EMBED_MODEL_ID, "input": text}
    r = http.post(url, data=json.dumps(payload), timeout=30)
    r.raise_for_status()
    data = r.json()
    vec = data["data"][0]["embedding"]
    # normalize to match earlier pipelines
    return _l2norm(vec)

def _hf_chat(prompt: str, max_new_tokens: int = 220) -> str:
    """
    Uses HF Router OpenAI-compatible chat completions.
    POST {HF_BASE_URL}/v1/chat/completions
    payload: {"model": HF_MODEL_ID, "messages":[{"role":"user","content": prompt}], ...}
    """
    url = f"{HF_BASE_URL}/v1/chat/completions"
    payload = {
        "model": HF_MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_new_tokens,
        "stream": False
    }
    r = http.post(url, data=json.dumps(payload), timeout=60)
    r.raise_for_status()
    out = r.json()
    return (out["choices"][0]["message"]["content"] or "").strip()

def encode_query(query: str, w_text: float = 1.0) -> List[float]:
    vec = _hf_embed(query)
    # ensure TEXT_DIM length
    if len(vec) > TEXT_DIM:
        vec = vec[:TEXT_DIM]
    elif len(vec) < TEXT_DIM:
        vec = vec + [0.0] * (TEXT_DIM - len(vec))
    # weight text + pad image slots
    vec = [w_text * x for x in vec]
    return vec + [0.0] * IMG_DIM

def search(query: str, top_k: int = TOP_K_DEFAULT, w_text: float = 1.0, filt: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    qvec = encode_query(_expand_query(query), w_text=w_text)
    res = index.query(vector=qvec, top_k=top_k, include_metadata=True, filter=filt or {})
    out = []
    for m in res.get("matches", []):
        md = m.get("metadata", {}) or {}
        out.append({
            "id": m.get("id"),
            "score": float(m.get("score", 0.0)),
            "title": md.get("title"),
            "brand": md.get("brand"),
            "price": md.get("price"),
            "image_url": md.get("image_url"),
        })
    return out

def filter_hits(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return rows
    def keep(r: Dict[str, Any]) -> bool:
        t = str(r.get("title", "")).lower()
        return any(g in t for g in GOOD) and not any(b in t for b in BAD)
    kept = [r for r in rows if keep(r)]
    if len(kept) >= 2:
        return kept[:5]
    def util(r: Dict[str, Any]) -> bool:
        t = str(r.get("title", "")).lower()
        if any(b in t for b in BAD):
            return False
        return bool(re.search(r"(tray|table|ottoman|stool)", t))
    extra = [r for r in rows if util(r)]
    seen = set()
    out = []
    for r in kept + extra:
        if r["id"] in seen:
            continue
        out.append(r); seen.add(r["id"])
        if len(out) >= 5:
            break
    return out if out else rows[:5]

def _build_prompt(user_q: str, rows: List[Dict[str, Any]]) -> str:
    ctx_lines = []
    for r in rows[:5]:
        ctx_lines.append(f'- {r.get("title","N/A")} (Brand: {r.get("brand","N/A")}, Price: {r.get("price","N/A")})')
    ctx = "\n".join(ctx_lines) if ctx_lines else "No context."
    return (
        "You are a concise, helpful product recommendation assistant.\n"
        "Rules (follow strictly):\n"
        "- Never say you cannot answer; if the exact keyword is missing, pick the closest relevant items from the context and still answer.\n"
        "- Do NOT start with 'Sure', 'Okay', or 'Here is/Here’s'. No emojis or meta-chat.\n"
        "- Write exactly ONE paragraph of 4–6 sentences. Start neutrally (not a brand).\n"
        "- Mention at least two product titles exactly as in the context. Use only the context.\n\n"
        f"Context:\n{ctx}\n\n"
        f"User need:\n{user_q}\n\n"
        "Write now."
    )

def generate_text(user_q: str, rows: List[Dict[str, Any]], max_new_tokens: int = 220) -> str:
    titles = _normalize_titles(rows)
    prompt = _build_prompt(user_q, rows)
    txt = _hf_chat(prompt, max_new_tokens=max_new_tokens)

    # cleanup and guards
    txt = re.sub(r"^(sure,?\s*|okay,?\s*|here'?s\s+.*?:\s*)", "", txt, flags=re.I).strip()
    for b in [r"i cannot answer", r"i can't", r"unable", r"not mention", r"no context",
              r"i'm here to assist", r"would you like", r"let me know", r"please note",
              r"i hope this helps", r"[😊😁🙂😉👍]"]:
        txt = re.sub(b, "", txt, flags=re.I)
    txt = re.sub(r"\s+", " ", txt).strip()

    need = [t for t in titles[:3] if t and t not in txt]
    if need:
        txt += " In particular, consider " + " and ".join(f"\"{n}\"" for n in need[:2]) + "."

    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", txt) if s.strip()]
    while len(sents) < 4:
        sents.append("These options balance comfort, value, and everyday usability at home.")
    txt = " ".join(sents[:6])

    if sum(1 for t in titles if t in txt) < 2:
        picks = titles[:3]
        if len(picks) >= 2:
            base = (
                "These options offer practical seating and storage for living spaces. "
                f"\"{picks[0]}\" and \"{picks[1]}\" stand out for everyday comfort and value"
                + (f", while \"{picks[2]}\" adds a versatile accent." if len(picks) > 2 else ".")
            )
        elif len(picks) == 1:
            base = f"\"{picks[0]}\" is a practical choice with solid everyday value."
        else:
            base = "These options balance comfort, value, and everyday usability at home."
        txt = base
    return txt

def rag(query: str, top_k: int | None = None) -> Dict[str, Any]:
    k = int(top_k or TOP_K_DEFAULT)
    raw = search(query, top_k=k)
    used = filter_hits(raw)
    text = generate_text(query, used)
    return {"recommendations": used, "generated_text": text}
