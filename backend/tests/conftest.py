"""
Shared fixtures for the RAG chatbot test suite.

Provides:
- StubConfig     — minimal config object replacing the real config singleton
- mock_rag       — controllable MagicMock for RAGSystem
- app            — minimal FastAPI app mirroring app.py routes (no static file mount)
- client         — TestClient wrapping the test app
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from pydantic import BaseModel
from typing import List, Optional


# ---------------------------------------------------------------------------
# Shared config stub
# ---------------------------------------------------------------------------

class StubConfig:
    ANTHROPIC_API_KEY = "test-key"
    ANTHROPIC_MODEL = "claude-sonnet-4-6"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    MAX_RESULTS = 5
    MAX_HISTORY = 2
    CHROMA_PATH = "/tmp/test_chroma_api"


# ---------------------------------------------------------------------------
# Pydantic models (mirrors app.py — kept here so the test app stays self-contained)
# ---------------------------------------------------------------------------

class _QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class _QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    session_id: str


class _CourseStats(BaseModel):
    total_courses: int
    course_titles: List[str]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_rag():
    """RAGSystem mock with sensible defaults for all endpoints."""
    rag = MagicMock()
    rag.query.return_value = ("Test answer.", [])
    rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Course A", "Course B"],
    }
    rag.session_manager.create_session.return_value = "session_1"
    return rag


@pytest.fixture
def app(mock_rag):
    """
    Minimal FastAPI app that mirrors the routes in app.py but:
    - does NOT mount static files (frontend/ won't exist in CI / test env)
    - does NOT import the real RAGSystem or ChromaDB
    - injects mock_rag via closure so tests can manipulate it directly
    """
    _app = FastAPI(title="Test App")

    @_app.post("/api/query", response_model=_QueryResponse)
    async def query_documents(request: _QueryRequest):
        try:
            session_id = request.session_id or mock_rag.session_manager.create_session()
            answer, sources = mock_rag.query(request.query, session_id)
            return _QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @_app.get("/api/courses", response_model=_CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag.get_course_analytics()
            return _CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @_app.delete("/api/session/{session_id}")
    async def delete_session(session_id: str):
        mock_rag.session_manager.clear_session(session_id)
        return {"status": "ok"}

    return _app


@pytest.fixture
def client(app):
    """TestClient for the test FastAPI app."""
    return TestClient(app)
