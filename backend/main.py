import os, logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from ai_logic import rag, healthcheck

log = logging.getLogger("main")

FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "")
ALLOWED = [FRONTEND_ORIGIN] if FRONTEND_ORIGIN else []
ALLOWED += ["http://localhost:5173", "http://127.0.0.1:5173"]

app = FastAPI(title="FurniFind API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RecReq(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(ge=1, le=20, default=8)

@app.get("/health")
def health():
    return healthcheck()

@app.post("/recommend")
def recommend(req: RecReq):
    try:
        out = rag(req.query, top_k=req.top_k)
        return out
    except Exception as e:
        log.exception("recommend failed: %s", e)
        raise HTTPException(status_code=500, detail="Internal error")
