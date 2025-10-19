# FurniFind — Furniture Recommendations with RAG (Pinecone + HF + Cohere)

A production-ready Retrieval-Augmented Generation (RAG) system that recommends furniture products and summarizes the picks in natural language.

- **Search**: Pinecone serverless vector index over **multi‑modal embeddings** (text + image).
- **Embeddings**: 
  - Text: `sentence-transformers/all-MiniLM-L6-v2` (384-d) for dataset creation; **runtime query** via Cohere `embed-english-light-v2.0` (384-d).
  - Image: ResNet‑50 penultimate layer (2048-d).
  - **Final vector**: 384 + 2048 = **2432-d**, concatenated.
- **Generation**: Hugging Face Inference **Router** (OpenAI-compatible) with `google/gemma-2-2b-it:nebius`.
- **Backend**: FastAPI on Render (free).
- **Frontend**: Vercel (Next.js/React) — calls the FastAPI endpoints.

---

## 1) Repository Layout

```
.
├── backend/
│   ├── ai_logic.py          # Retrieval + prompt + generation
│   ├── main.py              # FastAPI app (CORS, routes)
│   ├── requirements.txt     # Pinned deps
│   └── render.yaml          # Render web service
├── notebooks/
│   ├── data_analytics_notebook.ipynb   # EDA & cleaning
│   └── model_training_notebook.ipynb   # Text+image embeddings + Pinecone upsert
└── frontend/                # Deployed to Vercel (optional in this repo)
```

---

## 2) Data → Indexing Pipeline (notebooks/)

**Goal:** create a single, unified embedding per product covering both **text** and **image** signals.

1. **Load dataset**: `cleaned_intern_data.csv` with title, description, categories, images, brand, price, etc.
2. **Text embedding (384-d)**  
   - Build `combined_text = title + description + categories + "Material: ..." + "Color: ..."`  
   - Encode with `sentence-transformers/all-MiniLM-L6-v2` → shape `(N, 384)`.
3. **Image embedding (2048-d)**  
   - Download first image URL per product.  
   - ResNet‑50 (ImageNet weights), take penultimate layer → `(N, 2048)`.
4. **Concatenate** → `(N, 2432)` and **upsert** to Pinecone:
   - Index name: `product-recommendations`
   - Metric: `cosine`
   - Dimension: `2432`
   - Serverless spec: `aws | us-east-1`

Each vector stores metadata: `title`, `brand`, `price`, `image_url` for UI rendering.

> See: `notebooks/model_training_notebook.ipynb` for the exact code used.

---

## 3) Runtime Retrieval + Generation (backend/ai_logic.py)

**Query embedding:** Cohere `embed-english-light-v2.0` (`input_type="search_query"`) → 384-d, padded with 2048 zeros to 2432-d to match the index.  
**Search:** Pinecone `query(top_k, include_metadata=True)` → filter + re-rank rules (ban obvious hardware parts, prioritize furniture classes).  
**Prompting:** compact, single-paragraph guidance that forces mentioning ≥2 context titles.  
**Generation:** POST to HF Router `v1/chat/completions` with model `google/gemma-2-2b-it:nebius` (OpenAI-compatible response).

---

## 4) Backend API

Base URL (Render): `https://<your-service>.onrender.com`

### Health
```
GET /healthz
200 → {"status":"ok"}
```

### Recommendations
```
POST /recommend
Content-Type: application/json

{
  "query": "sofa",
  "top_k": 8   // optional, defaults to env TOP_K_DEFAULT
}
```
**200 Response**
```json
{
  "recommendations": [
    {
      "id": "bdc9aa30-9439-50dc-8e89-213ea211d66a",
      "score": 0.023,
      "title": "Karl home Accent Chair ...",
      "brand": "Karl home Store",
      "price": 149.99,
      "image_url": "https://..."
    }
  ],
  "generated_text": "Compact 4–6 sentence summary grounded in retrieved items..."
}
```

### Analytics (demo data)
```
GET /analytics
```

**Interactive docs**: `GET /docs` (Swagger UI).

---

## 5) Environment Variables

| Key | Example / Notes |
|---|---|
| `PINECONE_API_KEY` | Pinecone project key |
| `PINECONE_INDEX_NAME` | `product-recommendations` |
| `PINECONE_HOST` | Data-plane host (from `pc.describe_index(index_name).host`) |
| `HF_TOKEN` | Hugging Face access token (authorized for Gemma 2) |
| `HF_MODEL_ID` | `google/gemma-2-2b-it:nebius` |
| `COHERE_API_KEY` | Cohere API key |
| `EMBED_MODEL_ID` | `embed-english-light-v2.0` |
| `TEXT_DIM` | `384` |
| `IMG_DIM` | `2048` |
| `TOP_K_DEFAULT` | `8` |
| `ALLOWED_ORIGINS` | Comma-separated origins, e.g. `https://your-frontend.vercel.app` |

> `main.py` consumes `ALLOWED_ORIGINS` for CORS. Set to `"*"` only for local testing.

---

## 6) Local Development

```bash
cd backend
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# .env in backend/
cat > .env << 'EOF'
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=product-recommendations
PINECONE_HOST=... # optional but recommended
HF_TOKEN=...
HF_MODEL_ID=google/gemma-2-2b-it:nebius
COHERE_API_KEY=...
EMBED_MODEL_ID=embed-english-light-v2.0
TEXT_DIM=384
IMG_DIM=2048
TOP_K_DEFAULT=8
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:3000
EOF

uvicorn main:app --reload --port 8000
# Open http://localhost:8000/docs
```

---

## 7) Deployment

### Render (Backend)
- **rootDir**: `backend`
- **Build**: `pip install -r requirements.txt`
- **Start**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Health**: `/healthz`  
Env vars: same as section 5.

### Vercel (Frontend)
- Set `NEXT_PUBLIC_API_BASE` (or equivalent) → Render URL.
- Ensure the backend’s `ALLOWED_ORIGINS` includes the deployed Vercel domain.

---

## 8) Retrieval Quality Heuristics

- **GOOD classes**: `sofa, chair, ottoman, bench, couch, table, tray, armchair, stool`
- **BAD tokens**: `lever, latch, cable, release, hardware, bracket, replacement, webbing, band, repair, modification`  
- If <2 strong hits, allow **utility** items (`tray|table|ottoman|stool`) while still blocking BAD tokens.
- Final list capped at **5 items** for clean UX.

---

## 9) Tech Stack

- **Vector DB**: Pinecone (serverless, cosine, d=2432)  
- **Embedding (dataset)**: `sentence-transformers/all-MiniLM-L6-v2`  
- **Embedding (runtime query)**: Cohere `embed-english-light-v2.0`  
- **Image model**: ResNet‑50 (torchvision)  
- **LLM**: `google/gemma-2-2b-it:nebius` via HF Router (`/v1/chat/completions`)  
- **API**: FastAPI + Uvicorn  
- **Deploy**: Render (backend), Vercel (frontend)

---

## 10) Quick cURL

```bash
curl -X POST "https://<your-service>.onrender.com/recommend"   -H "Content-Type: application/json"   -d '{"query":"ottomans","top_k":8}'
```

---

## 11) Notes

- Keep `PINECONE_HOST` to use the **data-plane** endpoint for lower latency.
- Ensure Hugging Face token has **access to the Gemma 2** gated model.
- Do not store secrets in the repo; use `.env` locally and provider env consoles in prod.

---

## 12) License

Choose a license that suits the intended use (e.g., MIT, Apache-2.0). Add a `LICENSE` file at repo root.
