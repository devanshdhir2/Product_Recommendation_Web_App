import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import orjson

from ai_logic import rag

load_dotenv()

def _split_csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]

origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS","*").split(",") if o.strip()]

app = FastAPI(title="FurniFind API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryBody(BaseModel):
    query: str
    top_k: Optional[int] = None

@app.get("/")
def root():
    return {"ok": True, "name": "FurniFind API"}

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/recommend")
def recommend(body: QueryBody):
    try:
        result = rag(body.query, top_k=body.top_k)
        return result
    except Exception as e:
        print("ERROR /recommend:", e)
        raise HTTPException(status_code=500, detail="Internal error")

# Optional: simple analytics (static or from a CSV if you add one)
@app.get("/analytics")
def analytics():
    # Minimal placeholder so the UI can render something
    # If you have backend/data/cleaned_intern_data.csv, you can expand this.
    return {
        "products_per_brand": {
            "Karl home Store": 6,
            "Nalupatio Store": 5,
            "Generic": 4,
            "Dewhut Store": 4,
            "PONTMENT": 3
        },
        "avg_price_per_brand": {
            "Karl home Store": 149.99,
            "Nalupatio Store": 72.40,
            "Generic": 18.50,
            "Dewhut Store": 219.99,
            "PONTMENT": 95.99
        }
    }
