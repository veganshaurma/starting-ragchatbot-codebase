"""
Tests for RAGSystem.query() handling of content-related questions.

Covers:
- Response and sources are returned correctly
- AI generator is called with tool definitions
- User query is wrapped in the RAG prompt
- Session history is retrieved and forwarded when session_id provided
- Session history is updated after a successful query
- Sources collected by tool manager are returned
- Tool manager sources are reset after each query
- Exceptions from the AI layer propagate (not silently swallowed → 500)
- Model name is a known/valid identifier
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import MagicMock, patch
from rag_system import RAGSystem

# ---------------------------------------------------------------------------
# Minimal config stub
# ---------------------------------------------------------------------------


class _Config:
    ANTHROPIC_API_KEY = "test-key"
    ANTHROPIC_MODEL = "claude-sonnet-4-5"  # expected valid model ID
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    MAX_RESULTS = 5
    MAX_HISTORY = 2
    CHROMA_PATH = "/tmp/test_chroma_rag"


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def rag():
    """RAGSystem with all heavy components replaced by mocks."""
    with (
        patch("rag_system.VectorStore"),
        patch("rag_system.AIGenerator"),
        patch("rag_system.DocumentProcessor"),
    ):
        system = RAGSystem(_Config())

    # Replace live objects with controllable mocks
    system.ai_generator = MagicMock()
    system.ai_generator.generate_response.return_value = (
        "Python is a programming language."
    )

    system.tool_manager = MagicMock()
    system.tool_manager.get_tool_definitions.return_value = [
        {
            "name": "search_course_content",
            "description": "Search",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }
    ]
    system.tool_manager.get_last_sources.return_value = []
    system.session_manager = MagicMock()
    system.session_manager.get_conversation_history.return_value = None

    return system


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRAGSystemQuery:

    def test_returns_tuple_of_response_and_sources(self, rag):
        response, sources = rag.query("What is Python?")
        assert response == "Python is a programming language."
        assert isinstance(sources, list)

    def test_ai_generator_receives_tool_definitions(self, rag):
        rag.query("Tell me about Python")
        kw = rag.ai_generator.generate_response.call_args[1]
        assert "tools" in kw
        assert len(kw["tools"]) > 0

    def test_user_query_is_embedded_in_prompt(self, rag):
        rag.query("What is Python?")
        kw = rag.ai_generator.generate_response.call_args[1]
        assert "What is Python?" in kw["query"]

    def test_tool_manager_is_passed_to_ai_generator(self, rag):
        rag.query("Content question")
        kw = rag.ai_generator.generate_response.call_args[1]
        assert kw.get("tool_manager") is rag.tool_manager

    def test_session_history_retrieved_when_session_id_given(self, rag):
        rag.session_manager.get_conversation_history.return_value = "Prior context"
        rag.query("Follow-up", session_id="session_1")
        kw = rag.ai_generator.generate_response.call_args[1]
        assert kw.get("conversation_history") == "Prior context"

    def test_no_history_when_no_session_id(self, rag):
        rag.query("Standalone question")
        kw = rag.ai_generator.generate_response.call_args[1]
        assert kw.get("conversation_history") is None

    def test_session_history_updated_after_query(self, rag):
        rag.query("Question", session_id="session_1")
        rag.session_manager.add_exchange.assert_called_once()
        call_args = rag.session_manager.add_exchange.call_args[0]
        assert call_args[0] == "session_1"
        assert "Question" in call_args[1]
        assert call_args[2] == "Python is a programming language."

    def test_sources_come_from_tool_manager(self, rag):
        expected = [{"label": "Course A - Lesson 1", "url": "https://x.com"}]
        rag.tool_manager.get_last_sources.return_value = expected
        _, sources = rag.query("Content question")
        assert sources == expected

    def test_tool_manager_sources_reset_after_query(self, rag):
        rag.query("Any question")
        rag.tool_manager.reset_sources.assert_called_once()

    def test_ai_exception_propagates_as_500_candidate(self, rag):
        """Exceptions must propagate to app.py's handler — not silently swallowed.
        A silent catch here would hide the real error from the HTTP 500 response."""
        rag.ai_generator.generate_response.side_effect = Exception(
            "API error: model not found"
        )
        with pytest.raises(Exception, match="API error"):
            rag.query("Any question")


# ---------------------------------------------------------------------------
# Config sanity check
# ---------------------------------------------------------------------------


class TestConfigModelName:
    """The configured model name must match a known Anthropic Claude model pattern.

    Background: 'claude-sonnet-4-20250514' is NOT a valid model ID for the
    Claude 4 family. Claude 4 models use the format 'claude-<variant>-4-<minor>'
    (e.g. 'claude-sonnet-4-5', 'claude-sonnet-4-6').  The date-suffix pattern
    ('...20250514') belongs to the older Claude 3.x series.  An invalid model
    name causes every API call to fail with a 404/400, which propagates to the
    frontend as 'Query failed'.
    """

    def test_model_name_matches_valid_claude_pattern(self):
        # Import the real config used in production
        import importlib
        import config as cfg_module

        importlib.reload(cfg_module)  # ensure fresh load
        model = cfg_module.config.ANTHROPIC_MODEL

        valid_prefixes = (
            "claude-opus-4",
            "claude-sonnet-4",
            "claude-haiku-4",
            "claude-3-5-sonnet",
            "claude-3-5-haiku",
            "claude-3-opus",
        )
        assert any(model.startswith(p) for p in valid_prefixes), (
            f"Model '{model}' does not match a known Claude model prefix. "
            f"Check config.py — Claude 4 models use 'claude-sonnet-4-5' or "
            f"'claude-sonnet-4-6', not date-suffixed variants like "
            f"'claude-sonnet-4-20250514'."
        )

    def test_model_name_not_date_suffixed_for_claude4(self):
        """Claude 4 model IDs must NOT end in an 8-digit date stamp.

        'claude-sonnet-4-20250514' is invalid.  Correct forms are
        'claude-sonnet-4-5' or 'claude-sonnet-4-6'.
        """
        import importlib
        import config as cfg_module

        importlib.reload(cfg_module)
        model = cfg_module.config.ANTHROPIC_MODEL

        import re

        if "claude-" in model and "-4-" in model:
            assert not re.search(r"-\d{8}$", model), (
                f"Model '{model}' looks like a Claude 4 model but uses a "
                f"date suffix (e.g. '-20250514').  Claude 4 IDs use a short "
                f"version number like '-4-5' or '-4-6'.  This invalid name "
                f"causes all API calls to fail."
            )
