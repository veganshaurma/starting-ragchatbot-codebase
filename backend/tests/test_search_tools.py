"""
Tests for CourseSearchTool.execute() and ToolManager routing.

Covers:
- Happy path: formatted results returned
- Empty results: clear message returned
- Search errors: error string passed through
- Filter parameters forwarded correctly
- Source tracking and deduplication
- ToolManager routing
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import MagicMock
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from vector_store import SearchResults


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_store(documents=None, metadata=None, distances=None, error=None, lesson_link=None):
    """Return a mock VectorStore with a preset search() response."""
    store = MagicMock()
    if error:
        store.search.return_value = SearchResults.empty(error)
    else:
        store.search.return_value = SearchResults(
            documents=documents or [],
            metadata=metadata or [],
            distances=distances or [],
        )
    store.get_lesson_link.return_value = lesson_link
    return store


# ---------------------------------------------------------------------------
# CourseSearchTool.execute()
# ---------------------------------------------------------------------------

class TestCourseSearchToolExecute:

    def test_returns_formatted_content_on_success(self):
        """Documents and their course/lesson headers appear in the output."""
        store = make_store(
            documents=["Python is interpreted.", "Lists hold ordered data."],
            metadata=[
                {"course_title": "Intro to Python", "lesson_number": 1},
                {"course_title": "Intro to Python", "lesson_number": 1},
            ],
            distances=[0.1, 0.2],
            lesson_link="https://example.com/lesson/1",
        )
        tool = CourseSearchTool(store)

        result = tool.execute(query="What is Python?")

        assert "Intro to Python" in result
        assert "Lesson 1" in result
        assert "Python is interpreted." in result

    def test_returns_no_content_message_when_results_empty(self):
        """Empty results produce a clear 'No relevant content found' message."""
        store = make_store(documents=[], metadata=[], distances=[])
        tool = CourseSearchTool(store)

        result = tool.execute(query="quantum entanglement")

        assert "No relevant content found" in result

    def test_returns_error_string_on_search_error(self):
        """SearchResults.error is returned verbatim so Claude can report it."""
        store = make_store(error="Search error: collection is empty")
        tool = CourseSearchTool(store)

        result = tool.execute(query="anything")

        assert "Search error" in result

    def test_passes_course_name_to_vector_store(self):
        """course_name kwarg is forwarded to VectorStore.search()."""
        store = make_store(
            documents=["Content"],
            metadata=[{"course_title": "ML Course", "lesson_number": 2}],
            distances=[0.1],
        )
        tool = CourseSearchTool(store)

        tool.execute(query="neural nets", course_name="ML")

        call_kwargs = store.search.call_args[1]
        assert call_kwargs.get("course_name") == "ML"

    def test_passes_lesson_number_to_vector_store(self):
        """lesson_number kwarg is forwarded to VectorStore.search()."""
        store = make_store(
            documents=["Lesson content"],
            metadata=[{"course_title": "Course A", "lesson_number": 3}],
            distances=[0.1],
        )
        tool = CourseSearchTool(store)

        tool.execute(query="content", lesson_number=3)

        call_kwargs = store.search.call_args[1]
        assert call_kwargs.get("lesson_number") == 3

    def test_empty_message_includes_course_filter_name(self):
        """When a course filter is applied and yields nothing, the filter name
        appears in the message so Claude knows the search was scoped."""
        store = make_store(documents=[], metadata=[], distances=[])
        tool = CourseSearchTool(store)

        result = tool.execute(query="something", course_name="Nonexistent Course")

        assert "Nonexistent Course" in result

    def test_tracks_sources_after_successful_search(self):
        """last_sources is populated with label and url after a search."""
        store = make_store(
            documents=["Content"],
            metadata=[{"course_title": "Python Basics", "lesson_number": 1}],
            distances=[0.1],
            lesson_link="https://example.com/l/1",
        )
        tool = CourseSearchTool(store)

        tool.execute(query="Python")

        assert len(tool.last_sources) == 1
        src = tool.last_sources[0]
        assert src["label"] == "Python Basics - Lesson 1"
        assert src["url"] == "https://example.com/l/1"

    def test_deduplicates_sources_for_same_lesson(self):
        """Two chunks from the same lesson appear as a single source entry."""
        store = make_store(
            documents=["Chunk 1", "Chunk 2"],
            metadata=[
                {"course_title": "Python", "lesson_number": 1},
                {"course_title": "Python", "lesson_number": 1},
            ],
            distances=[0.1, 0.2],
        )
        tool = CourseSearchTool(store)

        tool.execute(query="Python basics")

        assert len(tool.last_sources) == 1

    def test_sources_cleared_before_each_search(self):
        """Stale sources from a previous search are not mixed into the next."""
        store1 = make_store(
            documents=["Old content"],
            metadata=[{"course_title": "Old Course", "lesson_number": 1}],
            distances=[0.1],
        )
        tool = CourseSearchTool(store1)
        tool.execute(query="first search")

        # New search with empty results — previous sources must not survive.
        tool.store = make_store(documents=[], metadata=[], distances=[])
        tool.execute(query="second search")

        assert tool.last_sources == []

    def test_get_tool_definition_schema_is_valid(self):
        """Tool definition satisfies the minimum Anthropic schema requirements."""
        tool = CourseSearchTool(MagicMock())
        defn = tool.get_tool_definition()

        assert defn["name"] == "search_course_content"
        assert "input_schema" in defn
        assert "query" in defn["input_schema"]["properties"]
        assert "query" in defn["input_schema"]["required"]


# ---------------------------------------------------------------------------
# ToolManager
# ---------------------------------------------------------------------------

class TestToolManager:

    def _make_manager_with_search_tool(self, store=None):
        manager = ToolManager()
        s = store or make_store(documents=[], metadata=[], distances=[])
        manager.register_tool(CourseSearchTool(s))
        return manager

    def test_routes_execute_tool_to_registered_tool(self):
        """execute_tool('search_course_content') delegates to CourseSearchTool."""
        manager = self._make_manager_with_search_tool()
        result = manager.execute_tool("search_course_content", query="test")
        assert "No relevant content found" in result

    def test_execute_tool_returns_error_for_unknown_name(self):
        """Requesting an unregistered tool name returns a descriptive error."""
        manager = ToolManager()
        result = manager.execute_tool("does_not_exist", query="x")
        assert "not found" in result.lower()

    def test_get_tool_definitions_returns_all_registered_tools(self):
        """Every registered tool's definition is included in the list."""
        manager = self._make_manager_with_search_tool()
        defs = manager.get_tool_definitions()
        names = [d["name"] for d in defs]
        assert "search_course_content" in names

    def test_get_last_sources_returns_empty_before_any_search(self):
        """No sources exist before the first search is performed."""
        manager = self._make_manager_with_search_tool()
        assert manager.get_last_sources() == []

    def test_get_last_sources_returns_sources_after_search(self):
        """Sources collected by the tool are accessible via ToolManager."""
        store = make_store(
            documents=["Content"],
            metadata=[{"course_title": "Course", "lesson_number": 1}],
            distances=[0.1],
            lesson_link=None,
        )
        manager = self._make_manager_with_search_tool(store)
        manager.execute_tool("search_course_content", query="something")
        sources = manager.get_last_sources()
        assert len(sources) == 1

    def test_reset_sources_clears_all_tool_sources(self):
        """reset_sources() empties last_sources on all tools."""
        store = make_store(
            documents=["Content"],
            metadata=[{"course_title": "Course", "lesson_number": 1}],
            distances=[0.1],
        )
        manager = self._make_manager_with_search_tool(store)
        manager.execute_tool("search_course_content", query="something")
        manager.reset_sources()
        assert manager.get_last_sources() == []
