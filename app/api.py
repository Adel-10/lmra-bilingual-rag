"""
Phase 5 — thin API server around the RAG pipeline.

Why a server at all: the embedding model takes ~10s to load, so it must
live in a long-running process. This wraps the EXACT RagPipeline.answer()
path the eval harness scored — the UI demos the evaluated system, not a
parallel reimplementation.

    pip install fastapi uvicorn
    export ANTHROPIC_API_KEY=sk-ant-...
    python app/api.py            # -> http://localhost:8000
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from rag import RagPipeline

app = FastAPI(title="LMRA bilingual RAG")
rag: RagPipeline | None = None


class Ask(BaseModel):
    question: str


@app.on_event("startup")
def load_pipeline() -> None:
    global rag
    rag = RagPipeline()  # loads BGE-M3 once (~10s)


@app.post("/api/ask")
def ask(body: Ask) -> dict:
    out = rag.answer(body.question.strip())
    # trim the response: the UI needs the answer + sources, not raw chunks
    return {
        "answer": out["answer"],
        "abstained": out["abstained"],
        "sources": out["sources"],
        "top_score": round(out["debug"]["top_score"], 3),
        "query_language": out["debug"]["query_language"],
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "app" / "index.html")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
