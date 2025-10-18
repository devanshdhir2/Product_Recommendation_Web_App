import os, re, torch, pandas as pd
from typing import List, Dict
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
INDEX_NAME = os.getenv("INDEX_NAME", "product-recommendations")

pc = Pinecone(api_key=PINECONE_API_KEY)
idx_desc = pc.describe_index(INDEX_NAME)
index = pc.Index(host=idx_desc.host)

TEXT_DIM, IMG_DIM = 384, 2048
enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def encode_query_mm(q: str, w_text: float = 1.0) -> List[float]:
    v = enc.encode(q, normalize_embeddings=True).tolist()
    v = [w_text * x for x in v]
    return v + [0.0] * IMG_DIM

def search(query: str, top_k: int = 5, w_text: float = 1.0, filt: Dict | None = None) -> pd.DataFrame:
    qvec = encode_query_mm(query, w_text=w_text)
    res = index.query(vector=qvec, top_k=top_k, include_metadata=True, filter=filt or {})
    rows = []
    for m in res.get("matches", []):
        md = m.get("metadata", {})
        rows.append({
            "id": m.get("id"),
            "score": float(m.get("score", 0.0)),
            "title": md.get("title"),
            "brand": md.get("brand"),
            "price": md.get("price"),
            "image_url": md.get("image_url"),
        })
    return pd.DataFrame(rows)

BAD = ("lever","latch","cable","release","hardware","bracket","replacement","webbing","band","repair","modification")
GOOD = ("sofa","chair","ottoman","bench","couch","table","tray","armchair","stool")

def filter_hits(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    t = df["title"].fillna("").str.lower()
    keep = t.apply(lambda x: any(g in x for g in GOOD) and not any(b in x for b in BAD))
    df2 = df[keep].copy()
    if len(df2) >= 2: return df2.head(5)
    util = t.str.contains(r"(tray|table|ottoman|stool)", regex=True) & ~t.apply(lambda x: any(b in x for b in BAD))
    extra = df[util].copy()
    out = pd.concat([df2, extra]).drop_duplicates(subset=["id"])
    return out.head(5) if len(out) else df.head(5)

MODEL_ID = "google/gemma-2b-it"
use_gpu = torch.cuda.is_available()
try:
    torch_dtype = torch.bfloat16 if use_gpu else torch.float32
except Exception:
    torch_dtype = torch.float16 if use_gpu else torch.float32

tok = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

gen = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    device_map="auto",
    torch_dtype=torch_dtype,
)

def gemma_answer(query: str, df_hits: pd.DataFrame, max_new_tokens: int = 240) -> str:
    titles = [str(t).strip() for t in df_hits.get("title", []).fillna("").tolist() if str(t).strip()]
    ctx_rows = []
    for _, r in df_hits.head(5).iterrows():
        ctx_rows.append(f"- {r.get('title','N/A')} (Brand: {r.get('brand','N/A')}, Price: {r.get('price','N/A')})")
    ctx = "\n".join(ctx_rows) if ctx_rows else "No context."

    prompt = (
        "You are a concise, helpful product recommendation assistant.\n"
        "Rules: never refuse; do not start with 'Sure/Okay/Here'; no emojis; no meta-chat.\n"
        "Write ONE paragraph of 4–6 sentences, start neutrally, name at least two products by exact title from context.\n\n"
        f"Context:\n{ctx}\n\nUser need:\n{query}\n\nWrite now."
    )

    ins = tok(prompt, return_tensors="pt").to(gen.device)
    with torch.no_grad():
        out = gen.generate(
            input_ids=ins["input_ids"],
            attention_mask=ins.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            min_new_tokens=min(120, max_new_tokens-20),
            do_sample=False,
            temperature=0.0,
            repetition_penalty=1.05,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
        )
    new_tokens = out[0, ins["input_ids"].shape[1]:]
    txt = tok.decode(new_tokens, skip_special_tokens=True).strip()
    txt = re.sub(r"^(sure,?\s*|okay,?\s*|here'?s\s+.*?:\s*)", "", txt, flags=re.I).strip()
    for b in [r"i cannot answer", r"i can't", r"unable", r"not mention", r"no context", r"i'm here to assist",
              r"would you like", r"let me know", r"please note", r"i hope this helps", r"[😊😁🙂😉👍]"]:
        txt = re.sub(b, "", txt, flags=re.I)
    txt = re.sub(r"\s+", " ", txt).strip()

    need = []
    for t in titles[:3]:
        if t and t not in txt:
            need.append(t)
        if len(need) >= 2:
            break
    if need:
        txt += " In particular, consider " + " and ".join(f'"{n}"' for n in need[:2]) + "."

    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", txt) if s.strip()]
    while len(sents) < 4:
        sents.append("These options balance comfort, value, and everyday usability at home.")
    return " ".join(sents[:6])

def rag(query: str, top_k: int = 5) -> dict:
    raw = search(query, top_k=top_k)
    used = filter_hits(raw)
    text = gemma_answer(query, used)
    return {"recommendations": used.to_dict(orient="records"), "generated_text": text}
