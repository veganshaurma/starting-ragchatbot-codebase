"""
Tests for FastAPI endpoints: POST /api/query, GET /api/courses, DELETE /api/session/{id}

Uses the `client` and `mock_rag` fixtures from conftest.py.
The test app mirrors app.py routes without mounting static files,
so these tests run cleanly without a frontend/ directory.

Covers:
- /api/query: success, auto session creation, provided session id, 500 on RAG error,
              422 on missing field, sources propagated
- /api/courses: success, 500 on error, empty course list
- /api/session/{id}: ok response, delegates to session_manager
"""


class TestQueryEndpoint:

    def test_success_returns_answer_and_session_id(self, client):
        response = client.post("/api/query", json={"query": "What is Python?"})
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Test answer."
        assert "session_id" in data
        assert isinstance(data["sources"], list)

    def test_auto_creates_session_when_none_provided(self, client, mock_rag):
        client.post("/api/query", json={"query": "Hello"})
        mock_rag.session_manager.create_session.assert_called_once()

    def test_skips_session_creation_when_id_provided(self, client, mock_rag):
        client.post("/api/query", json={"query": "Hello", "session_id": "session_42"})
        mock_rag.session_manager.create_session.assert_not_called()

    def test_forwards_provided_session_id_to_rag(self, client, mock_rag):
        client.post("/api/query", json={"query": "Hello", "session_id": "session_42"})
        mock_rag.query.assert_called_once_with("Hello", "session_42")

    def test_returns_500_on_rag_exception(self, client, mock_rag):
        mock_rag.query.side_effect = Exception("model not found")
        response = client.post("/api/query", json={"query": "Fail"})
        assert response.status_code == 500
        assert "model not found" in response.json()["detail"]

    def test_missing_query_field_returns_422(self, client):
        """Pydantic validation rejects a request body with no 'query' field."""
        response = client.post("/api/query", json={})
        assert response.status_code == 422

    def test_sources_from_rag_included_in_response(self, client, mock_rag):
        mock_rag.query.return_value = (
            "Here is the answer.",
            [{"label": "Course A - Lesson 1", "url": "https://example.com/l/1"}],
        )
        response = client.post("/api/query", json={"query": "What is Python?"})
        assert response.status_code == 200
        sources = response.json()["sources"]
        assert len(sources) == 1
        assert sources[0]["label"] == "Course A - Lesson 1"
        assert sources[0]["url"] == "https://example.com/l/1"

    def test_empty_sources_returned_when_rag_finds_nothing(self, client, mock_rag):
        mock_rag.query.return_value = ("I don't know.", [])
        response = client.post("/api/query", json={"query": "Unknowable question"})
        assert response.status_code == 200
        assert response.json()["sources"] == []


class TestCoursesEndpoint:

    def test_returns_course_count_and_titles(self, client):
        response = client.get("/api/courses")
        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 2
        assert data["course_titles"] == ["Course A", "Course B"]

    def test_returns_500_on_rag_exception(self, client, mock_rag):
        mock_rag.get_course_analytics.side_effect = Exception("DB connection failed")
        response = client.get("/api/courses")
        assert response.status_code == 500
        assert "DB connection failed" in response.json()["detail"]

    def test_returns_empty_list_when_no_courses_indexed(self, client, mock_rag):
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": [],
        }
        response = client.get("/api/courses")
        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    def test_total_courses_matches_titles_length(self, client, mock_rag):
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 3,
            "course_titles": ["A", "B", "C"],
        }
        response = client.get("/api/courses")
        data = response.json()
        assert data["total_courses"] == len(data["course_titles"])


class TestDeleteSessionEndpoint:

    def test_returns_ok_status(self, client):
        response = client.delete("/api/session/session_1")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_delegates_session_id_to_session_manager(self, client, mock_rag):
        client.delete("/api/session/abc123")
        mock_rag.session_manager.clear_session.assert_called_once_with("abc123")

    def test_accepts_arbitrary_session_id_format(self, client):
        """Session IDs with special characters or long strings must be accepted."""
        response = client.delete("/api/session/user-12345-xyz")
        assert response.status_code == 200
