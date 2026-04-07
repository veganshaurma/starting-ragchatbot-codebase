"""
Tests for AIGenerator.generate_response() and tool execution path.

Covers:
- Direct (non-tool) response extraction
- Tools passed to API when provided; absent when not
- Conversation history injected into system prompt
- Tool-use branch: tool is executed, result fed back, final text returned
- Synthesis call excludes tools (no infinite loop)
- Tool result content reaches the next API call
- Two sequential tool rounds (3 API calls total)
- Early exit when Claude returns text after first tool round
- Tool execution exception handled gracefully
- Round-2 API call includes tools so Claude can chain calls
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import MagicMock, patch
from ai_generator import AIGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_text_block(text: str):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _make_tool_use_block(name: str, input_data: dict, tool_id: str):
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.input = input_data
    block.id = tool_id
    return block


def _mock_response(text=None, stop_reason="end_turn", tool_calls=None):
    """Build a mock Anthropic Message response."""
    response = MagicMock()
    response.stop_reason = stop_reason
    content = []
    if text:
        content.append(_make_text_block(text))
    if tool_calls:
        for name, inp, tid in tool_calls:
            content.append(_make_tool_use_block(name, inp, tid))
    response.content = content
    return response


def _make_generator():
    """Return an AIGenerator whose Anthropic client is fully mocked."""
    with patch("ai_generator.anthropic.Anthropic"):
        gen = AIGenerator(api_key="test-key", model="claude-test")
    return gen


SAMPLE_TOOLS = [
    {
        "name": "search_course_content",
        "description": "Search course materials",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }
]


# ---------------------------------------------------------------------------
# Direct response (no tool use)
# ---------------------------------------------------------------------------

class TestDirectResponse:

    def setup_method(self):
        self.gen = _make_generator()
        self.client = self.gen.client

    def test_returns_text_from_first_content_block(self):
        self.client.messages.create.return_value = _mock_response(
            text="Here is the answer.", stop_reason="end_turn"
        )
        result = self.gen.generate_response(query="What is 2+2?")
        assert result == "Here is the answer."

    def test_query_is_placed_as_user_message(self):
        self.client.messages.create.return_value = _mock_response(
            text="Reply", stop_reason="end_turn"
        )
        self.gen.generate_response(query="My question")
        kw = self.client.messages.create.call_args[1]
        assert kw["messages"][0]["role"] == "user"
        assert kw["messages"][0]["content"] == "My question"

    def test_tools_included_in_api_call_when_provided(self):
        self.client.messages.create.return_value = _mock_response(
            text="Answer", stop_reason="end_turn"
        )
        self.gen.generate_response(query="Question", tools=SAMPLE_TOOLS)
        kw = self.client.messages.create.call_args[1]
        assert "tools" in kw
        assert kw.get("tool_choice") == {"type": "auto"}

    def test_tools_absent_from_api_call_when_not_provided(self):
        self.client.messages.create.return_value = _mock_response(
            text="Answer", stop_reason="end_turn"
        )
        self.gen.generate_response(query="General question")
        kw = self.client.messages.create.call_args[1]
        assert "tools" not in kw

    def test_conversation_history_appended_to_system_prompt(self):
        self.client.messages.create.return_value = _mock_response(
            text="Answer", stop_reason="end_turn"
        )
        self.gen.generate_response(
            query="Follow-up",
            conversation_history="User: Hi\nAI: Hello",
        )
        kw = self.client.messages.create.call_args[1]
        assert "Previous conversation" in kw["system"]
        assert "User: Hi" in kw["system"]

    def test_system_prompt_used_directly_when_no_history(self):
        self.client.messages.create.return_value = _mock_response(
            text="Answer", stop_reason="end_turn"
        )
        self.gen.generate_response(query="Question")
        kw = self.client.messages.create.call_args[1]
        # No history injected — system should be the static SYSTEM_PROMPT only
        assert "Previous conversation" not in kw["system"]


# ---------------------------------------------------------------------------
# Tool-use branch
# ---------------------------------------------------------------------------

class TestToolUseBranch:

    def setup_method(self):
        self.gen = _make_generator()
        self.client = self.gen.client

    def _mock_tool_manager(self, return_value="Tool output"):
        mgr = MagicMock()
        mgr.execute_tool.return_value = return_value
        return mgr

    def test_final_text_returned_after_tool_execution(self):
        self.client.messages.create.side_effect = [
            _mock_response(
                stop_reason="tool_use",
                tool_calls=[("search_course_content", {"query": "Python"}, "tid1")],
            ),
            _mock_response(text="Python is a language.", stop_reason="end_turn"),
        ]
        mgr = self._mock_tool_manager("Python content here")

        result = self.gen.generate_response(
            query="Tell me about Python", tools=SAMPLE_TOOLS, tool_manager=mgr
        )

        assert result == "Python is a language."

    def test_tool_manager_called_with_correct_name_and_input(self):
        self.client.messages.create.side_effect = [
            _mock_response(
                stop_reason="tool_use",
                tool_calls=[
                    ("search_course_content", {"query": "neural nets", "course_name": "ML"}, "tid2")
                ],
            ),
            _mock_response(text="Neural networks are...", stop_reason="end_turn"),
        ]
        mgr = self._mock_tool_manager("Some content")

        self.gen.generate_response(
            query="What are neural networks?", tools=SAMPLE_TOOLS, tool_manager=mgr
        )

        mgr.execute_tool.assert_called_once_with(
            "search_course_content", query="neural nets", course_name="ML"
        )

    def test_tool_result_appears_in_second_api_call_messages(self):
        self.client.messages.create.side_effect = [
            _mock_response(
                stop_reason="tool_use",
                tool_calls=[("search_course_content", {"query": "Python"}, "tid3")],
            ),
            _mock_response(text="Done.", stop_reason="end_turn"),
        ]
        mgr = self._mock_tool_manager("Returned content from tool")

        self.gen.generate_response(
            query="Python?", tools=SAMPLE_TOOLS, tool_manager=mgr
        )

        assert self.client.messages.create.call_count == 2
        second_kw = self.client.messages.create.call_args_list[1][1]
        messages = second_kw["messages"]

        found = False
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "tool_result":
                        found = True
                        assert item["content"] == "Returned content from tool"
                        assert item["tool_use_id"] == "tid3"
        assert found, "tool_result block not found in second API call messages"

    def test_synthesis_call_does_not_include_tools(self):
        """The final synthesis call (after 2 tool rounds) never includes tools."""
        self.client.messages.create.side_effect = [
            _mock_response(
                stop_reason="tool_use",
                tool_calls=[("search_course_content", {"query": "q1"}, "tid4a")],
            ),
            _mock_response(
                stop_reason="tool_use",
                tool_calls=[("search_course_content", {"query": "q2"}, "tid4b")],
            ),
            _mock_response(text="Final.", stop_reason="end_turn"),
        ]
        mgr = self._mock_tool_manager()

        self.gen.generate_response(
            query="Question", tools=SAMPLE_TOOLS, tool_manager=mgr
        )

        third_kw = self.client.messages.create.call_args_list[2][1]
        assert "tools" not in third_kw

    def test_api_called_exactly_twice_for_tool_use_flow(self):
        self.client.messages.create.side_effect = [
            _mock_response(
                stop_reason="tool_use",
                tool_calls=[("search_course_content", {"query": "q"}, "tid5")],
            ),
            _mock_response(text="Answer.", stop_reason="end_turn"),
        ]
        mgr = self._mock_tool_manager()

        self.gen.generate_response(
            query="Question", tools=SAMPLE_TOOLS, tool_manager=mgr
        )

        assert self.client.messages.create.call_count == 2

    def test_tool_use_without_tool_manager_returns_direct_response(self):
        """If stop_reason is tool_use but no tool_manager provided, returns
        the text content of the initial response (graceful degradation)."""
        self.client.messages.create.return_value = _mock_response(
            text="Fallback text.",
            stop_reason="tool_use",
            tool_calls=[("search_course_content", {"query": "q"}, "tid6")],
        )

        result = self.gen.generate_response(
            query="Question", tools=SAMPLE_TOOLS, tool_manager=None
        )

        # No second call should happen
        assert self.client.messages.create.call_count == 1
        assert result == "Fallback text."


# ---------------------------------------------------------------------------
# Multi-round tool use (up to 2 sequential rounds)
# ---------------------------------------------------------------------------

class TestMultiRoundToolUse:

    def setup_method(self):
        self.gen = _make_generator()
        self.client = self.gen.client

    def _mock_tool_manager(self, side_effects=None, return_value="Tool output"):
        mgr = MagicMock()
        if side_effects is not None:
            mgr.execute_tool.side_effect = side_effects
        else:
            mgr.execute_tool.return_value = return_value
        return mgr

    def test_two_sequential_tool_rounds_return_synthesis_text(self):
        """3 API calls total: round-1 tool_use → round-2 tool_use → synthesis end_turn."""
        self.client.messages.create.side_effect = [
            _mock_response(
                stop_reason="tool_use",
                tool_calls=[("get_course_outline", {"course_name": "ML"}, "tid-r1")],
            ),
            _mock_response(
                stop_reason="tool_use",
                tool_calls=[("search_course_content", {"query": "neural nets"}, "tid-r2")],
            ),
            _mock_response(text="Neural networks are...", stop_reason="end_turn"),
        ]
        mgr = self._mock_tool_manager(side_effects=["Outline result", "Content result"])

        result = self.gen.generate_response(
            query="Compare neural nets across courses", tools=SAMPLE_TOOLS, tool_manager=mgr
        )

        assert result == "Neural networks are..."
        assert self.client.messages.create.call_count == 3

    def test_tool_manager_called_once_per_round(self):
        """execute_tool is called exactly twice for a two-round flow."""
        self.client.messages.create.side_effect = [
            _mock_response(
                stop_reason="tool_use",
                tool_calls=[("get_course_outline", {"course_name": "ML"}, "tid-a1")],
            ),
            _mock_response(
                stop_reason="tool_use",
                tool_calls=[("search_course_content", {"query": "backprop"}, "tid-a2")],
            ),
            _mock_response(text="Done.", stop_reason="end_turn"),
        ]
        mgr = self._mock_tool_manager(side_effects=["outline data", "search data"])

        self.gen.generate_response(
            query="Question", tools=SAMPLE_TOOLS, tool_manager=mgr
        )

        assert mgr.execute_tool.call_count == 2
        first_call, second_call = mgr.execute_tool.call_args_list
        assert first_call[0][0] == "get_course_outline"
        assert second_call[0][0] == "search_course_content"

    def test_round2_api_call_includes_tools(self):
        """The intermediate call (between round 1 and round 2) must include tools."""
        self.client.messages.create.side_effect = [
            _mock_response(
                stop_reason="tool_use",
                tool_calls=[("get_course_outline", {"course_name": "ML"}, "tid-b1")],
            ),
            _mock_response(
                stop_reason="tool_use",
                tool_calls=[("search_course_content", {"query": "q"}, "tid-b2")],
            ),
            _mock_response(text="Answer.", stop_reason="end_turn"),
        ]
        mgr = self._mock_tool_manager()

        self.gen.generate_response(
            query="Question", tools=SAMPLE_TOOLS, tool_manager=mgr
        )

        second_kw = self.client.messages.create.call_args_list[1][1]
        assert "tools" in second_kw

    def test_round1_result_appears_in_round2_api_call_messages(self):
        """Tool output from round 1 must be in the messages sent for round 2."""
        self.client.messages.create.side_effect = [
            _mock_response(
                stop_reason="tool_use",
                tool_calls=[("get_course_outline", {"course_name": "ML"}, "tid-c1")],
            ),
            _mock_response(
                stop_reason="tool_use",
                tool_calls=[("search_course_content", {"query": "q"}, "tid-c2")],
            ),
            _mock_response(text="Done.", stop_reason="end_turn"),
        ]
        mgr = self._mock_tool_manager(side_effects=["Lesson 4 covers recursion", "search data"])

        self.gen.generate_response(
            query="Question", tools=SAMPLE_TOOLS, tool_manager=mgr
        )

        second_kw = self.client.messages.create.call_args_list[1][1]
        messages = second_kw["messages"]
        found = any(
            isinstance(msg.get("content"), list) and
            any(
                isinstance(item, dict) and
                item.get("type") == "tool_result" and
                item.get("content") == "Lesson 4 covers recursion"
                for item in msg["content"]
            )
            for msg in messages
        )
        assert found, "Round-1 tool result not found in round-2 API call messages"

    def test_early_exit_when_claude_returns_text_after_round1(self):
        """If Claude answers with end_turn after round 1 (tools still offered), return immediately — no synthesis call."""
        self.client.messages.create.side_effect = [
            _mock_response(
                stop_reason="tool_use",
                tool_calls=[("search_course_content", {"query": "Python"}, "tid-d1")],
            ),
            _mock_response(text="Python is a language.", stop_reason="end_turn"),
        ]
        mgr = self._mock_tool_manager()

        result = self.gen.generate_response(
            query="What is Python?", tools=SAMPLE_TOOLS, tool_manager=mgr
        )

        assert result == "Python is a language."
        assert self.client.messages.create.call_count == 2

    def test_tool_exception_handled_gracefully(self):
        """If execute_tool raises, error is sent to Claude and synthesis still returns text."""
        self.client.messages.create.side_effect = [
            _mock_response(
                stop_reason="tool_use",
                tool_calls=[("search_course_content", {"query": "q"}, "tid-e1")],
            ),
            _mock_response(text="Sorry, an error occurred.", stop_reason="end_turn"),
        ]
        mgr = self._mock_tool_manager()
        mgr.execute_tool.side_effect = Exception("db connection failed")

        result = self.gen.generate_response(
            query="Question", tools=SAMPLE_TOOLS, tool_manager=mgr
        )

        assert result == "Sorry, an error occurred."
        # Synthesis call must have received a tool_result with the error message
        second_kw = self.client.messages.create.call_args_list[1][1]
        messages = second_kw["messages"]
        found = any(
            isinstance(msg.get("content"), list) and
            any(
                isinstance(item, dict) and
                item.get("type") == "tool_result" and
                "db connection failed" in item.get("content", "")
                for item in msg["content"]
            )
            for msg in messages
        )
        assert found, "Error tool_result not found in synthesis call messages"

    def test_tool_exception_skips_round2_goes_straight_to_synthesis(self):
        """On exception in round 1, round 2 is skipped — exactly 2 API calls total."""
        self.client.messages.create.side_effect = [
            _mock_response(
                stop_reason="tool_use",
                tool_calls=[("search_course_content", {"query": "q"}, "tid-f1")],
            ),
            _mock_response(text="Partial answer.", stop_reason="end_turn"),
        ]
        mgr = self._mock_tool_manager()
        mgr.execute_tool.side_effect = Exception("timeout")

        self.gen.generate_response(
            query="Question", tools=SAMPLE_TOOLS, tool_manager=mgr
        )

        assert self.client.messages.create.call_count == 2

    def test_message_list_structure_after_two_rounds(self):
        """Messages sent to synthesis call: user, assistant, tool_result, assistant, tool_result (5 items)."""
        self.client.messages.create.side_effect = [
            _mock_response(
                stop_reason="tool_use",
                tool_calls=[("get_course_outline", {"course_name": "ML"}, "tid-g1")],
            ),
            _mock_response(
                stop_reason="tool_use",
                tool_calls=[("search_course_content", {"query": "q"}, "tid-g2")],
            ),
            _mock_response(text="Final.", stop_reason="end_turn"),
        ]
        mgr = self._mock_tool_manager(side_effects=["result1", "result2"])

        self.gen.generate_response(
            query="Original question", tools=SAMPLE_TOOLS, tool_manager=mgr
        )

        third_kw = self.client.messages.create.call_args_list[2][1]
        messages = third_kw["messages"]
        assert len(messages) == 5
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Original question"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        assert messages[3]["role"] == "assistant"
        assert messages[4]["role"] == "user"
