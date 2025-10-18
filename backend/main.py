import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ai_logic import rag, analytics_from_csv

app = FastAPI(title="Furniture RAG API")

FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN] if FRONTEND_ORIGIN != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserQuery(BaseModel):
    query: str
    top_k: int | None = 8

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/recommend")
def recommend(q: UserQuery):
    try:
        res = rag(q.query, top_k=q.top_k or 8)
        return res
    except Exception as e:
        print("RAG error:", e)
        raise HTTPException(status_code=500, detail="RAG pipeline failed")

@app.get("/analytics")
def analytics():
    try:
        return analytics_from_csv(os.getenv("ANALYTICS_CSV", "cleaned_intern_data.csv"))
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="cleaned_intern_data.csv not found")
    except Exception as e:
        print("Analytics error:", e)
        raise HTTPException(status_code=500, detail="Analytics failed")
