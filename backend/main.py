from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

from ai_logic import rag, healthcheck

app = FastAPI(title="FurniFind RAG API", version="1.0.0")

FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "")
origins = [FRONTEND_ORIGIN] if FRONTEND_ORIGIN else ["*"]  # keep it simple; tighten later

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryIn(BaseModel):
    query: str
    top_k: int = 8

@app.get("/health")
def _health():
    return healthcheck()

@app.post("/recommend")
def _recommend(payload: QueryIn = Body(...)):
    data = rag(payload.query, top_k=payload.top_k)
    return data

@app.get("/")
def _root():
    return {"ok": True, "service": "FurniFind RAG API"}
