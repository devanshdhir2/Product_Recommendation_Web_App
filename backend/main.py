
# === backend/main.py ===
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")  # loads backend/.env if present

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ai_logic import rag
# ...rest stays same...


app = FastAPI(title="Product Recommendation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserQuery(BaseModel):
    query: str
    top_k: int | None = 5

@app.get("/")
def root():
    return {"message": "OK"}

@app.post("/recommend")
def recommend(q: UserQuery):
    try:
        data = rag(q.query, top_k=q.top_k or 5)
        return data
    except Exception as e:
        print("/recommend error:", e)
        raise HTTPException(status_code=500, detail="Internal error")
