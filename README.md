# FurniFind – Multimodal Product Recommendation (RAG)

_A smart, context‑aware furniture discovery engine._

> Made with ❤️ by **Devansh** for **Ikarus 3D**

---

## Overview

FurniFind is a production‑ready, multimodal Retrieval‑Augmented Generation (RAG) system for product recommendations.  
It blends **text + image embeddings** inside **Pinecone** for accurate retrieval, and uses a lightweight **LLM (Gemma‑2 2B IT via Hugging Face Inference Router)** to write concise, human‑friendly suggestions bounded strictly by retrieved context. The stack is cloud‑friendly and **free‑tier deployable**: **FastAPI on Render** (backend) and **Vite + React on Vercel** (frontend).

---

## Architecture (High Level)

```
User (Vercel app)
        │
        ▼
Frontend (Vite + React)
        │  POST /recommend { query, top_k? }
        ▼
Backend (FastAPI on Render)
   ├── Query Embedding  → Cohere (embed-english-light-v2.0)
   ├── Vector Search    → Pinecone (2432‑D multimodal vectors)
   ├── Result Filtering → product/category guardrails
   └── Generation       → Hugging Face Inference Router
                          model: google/gemma-2-2b-it:nebius (chat/completions)
        │
        ▼
JSON: { recommendations: [...], generated_text: "..." }
```

---

## Data & Indexing Pipeline

**Source:** `cleaned_intern_data.csv` (210 products).

**Cleaning & Preprocessing**
- Dropped rows with missing **price**.
- Consolidated duplicates by **uniq_id**.
- Normalized columns and extracted the **first image URL** from mixed formats (list string / list / single URL / NaN).
- Built a **rich text field** for indexing: `title + description + categories + material + color`.

**Embeddings (offline, during indexing)**
- **Text (384‑D):** `sentence-transformers/all-MiniLM-L6-v2`.
- **Image (2048‑D):** **ResNet‑50** penultimate layer on the first product image (ImageNet normalization).
- **Multimodal vector (2432‑D):** concatenation `[ text(384) | image(2048) ]`.

**Vector DB**
- **Pinecone Serverless** (`metric=cosine`, `dimension=2432`).
- Upserted vectors with metadata: `title`, `brand`, `price`, `image_url`.
- Used the **data‑plane host** when available (faster queries).

> Result: `total_vector_count = 210`, ready for low‑latency retrieval.

---

## Serving (Retrieval + Generation)

**Query Embedding (at runtime)**
- **Cohere** `embed-english-light-v2.0` for text queries.
- Output is shaped to the index schema: kept/padded to **384** for the text block, then appended **2048 zeros** for the image block → final **2432‑D** query vector. This keeps compatibility with the multimodal index without recomputing image features at query time.

**Vector Search (Pinecone)**
- `top_k` configurable (default **8**).
- Returns product candidates with metadata.

**Result Filtering (guardrails)**
- Keeps furniture‑relevant items (`sofa|chair|ottoman|bench|couch|table|tray|armchair|stool`).
- Excludes hardware/repair/parts (`lever|latch|cable|bracket|replacement|...`).
- Falls back to utility items when needed (e.g., trays, stools) to guarantee at least two relevant results.

**Constrained Generation (Hugging Face Router)**
- Endpoint: **OpenAI‑compatible** `https://router.huggingface.co/v1/chat/completions`.
- Model: **`google/gemma-2-2b-it:nebius`**.
- Prompt rules enforce: **single paragraph**, **4–6 sentences**, **no meta‑chat**, **mention ≥2 exact titles from context**.
- Final message is sanitized (no “Sure/Okay/Here’s”, no emojis, no refusal), with a deterministic fallback if titles were not referenced enough.

---

## API (Backend)

**Base URL (Render):** `https://<your-render-service>.onrender.com`

### `POST /recommend`
Request
```json
{ "query": "looking for a compact sofa under 200", "top_k": 8 }
```
Response (shape)
```json
{
  "recommendations": [
    {
      "id": "fe25ae1d-...",
      "score": 0.0245,
      "title": "Karl home Accent Chair Mid-Century Modern ...",
      "brand": "Karl home Store",
      "price": 149.99,
      "image_url": "https://...jpg"
    }
  ],
  "generated_text": "These options offer practical seating ..."
}
```

### `GET /analytics`
Returns small, static sample analytics (brand counts & average price) — useful for the UI’s charts.

### `GET /healthz`
Simple health probe for Render.

---

## Environment Variables

### Backend (`backend/.env`)
```
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=product-recommendations
PINECONE_HOST=...                         # optional (data-plane host)

COHERE_API_KEY=...                        # for runtime query embeddings
EMBED_MODEL_ID=embed-english-light-v2.0
TEXT_DIM=384
IMG_DIM=2048
TOP_K_DEFAULT=8

HF_TOKEN=...                              # Hugging Face access token
HF_MODEL_ID=google/gemma-2-2b-it:nebius   # via HF Inference Router

ALLOWED_ORIGINS=https://<your-vercel-app>.vercel.app,http://localhost:5173
```

### Frontend (`frontend/.env`)
```
VITE_API_URL=https://<your-render-service>.onrender.com
```

---

## Local Development

**Backend**
```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # or create .env as above
uvicorn main:app --reload
```

**Frontend (Node 20.x)**
```bash
cd frontend
npm i
echo "VITE_API_URL=http://127.0.0.1:8000" > .env
npm run dev
```

---

## Deployment

**Render (Backend)**
- Root: `backend/`
- Build: `pip install -r requirements.txt`
- Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Health check: `/healthz`
- Add environment variables from the table above.
- Free tier is sufficient since heavy compute is delegated to Cohere/HF/Pinecone.

**Vercel (Frontend)**
- Framework: **Vite + React** (Node 20.x).
- Environment variable: `VITE_API_URL=https://<your-render-service>.onrender.com`.
- Build: `npm run build`, Output: `dist/`.

---

## Tech Stack

- **Retrieval:** Pinecone (serverless, cosine, 2432‑D multimodal vectors)
- **Text Embedding (indexing):** Sentence‑Transformers `all-MiniLM-L6-v2` (384‑D)
- **Image Embedding (indexing):** ResNet‑50 penultimate layer (2048‑D)
- **Text Embedding (serving):** Cohere `embed-english-light-v2.0` (384‑D shaped → 2432‑D)
- **Generation:** HF Inference Router (OpenAI API‑compatible), `google/gemma-2-2b-it:nebius`
- **Backend:** FastAPI, Uvicorn
- **Frontend:** Vite + React + Tailwind
- **Infra:** Render (web service), Vercel (static hosting)

---

## Notes & Limitations

- The generator is **strictly grounded** in retrieved context; if retrieval is sparse, a deterministic fallback summary is returned.
- The free tier on Render has **~512 MiB** memory; heavy models are not hosted on the server (stateless inference calls).
- Image features are **precomputed** at indexing time for speed; runtime queries are text‑only embeddings aligned to the same 2432‑D space.

---

## Directory Layout

```
repo-root/
├─ backend/
│  ├─ ai_logic.py
│  ├─ main.py
│  ├─ requirements.txt
│  └─ render.yaml
├─ frontend/
│  ├─ index.html
│  ├─ package.json
│  ├─ src/
│  └─ vite.config.ts
└─ notebooks/  (optional)
   ├─ data_analytics_notebook.ipynb
   └─ model_training_notebook.ipynb
```

---

## License

This project is released for educational and evaluation purposes.
