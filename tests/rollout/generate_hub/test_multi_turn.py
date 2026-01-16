import json

import pytest
from pydantic import TypeAdapter
from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser

from miles.utils.test_utils.mock_tools import (
    MULTI_TURN_FIRST_PROMPT,
    MULTI_TURN_FIRST_RESPONSE,
    MULTI_TURN_SECOND_RESPONSE,
    SAMPLE_TOOLS,
    execute_tool_call,
    multi_turn_tool_call_process_fn,
)


class TestToolCallParsing:
    @pytest.fixture
    def parser(self):
        tools = TypeAdapter(list[Tool]).validate_python(SAMPLE_TOOLS)
        return FunctionCallParser(tools=tools, tool_call_parser="qwen25")

    def test_parse_multi_tool_calls(self, parser):
        response = multi_turn_tool_call_process_fn(MULTI_TURN_FIRST_PROMPT).text
        normal_text, calls = parser.parse_non_stream(response)

        assert normal_text == "Let me get the year and temperature first."
        assert len(calls) == 2
        assert calls[0].name == "get_year"
        assert calls[0].parameters == "{}"
        assert calls[1].name == "get_temperature"
        assert json.loads(calls[1].parameters) == {"location": "Mars"}

    def test_parse_no_tool_calls(self, parser):
        normal_text, calls = parser.parse_non_stream(MULTI_TURN_SECOND_RESPONSE)
        assert len(calls) == 0
        assert "The answer is: 42 + 2026 + -60 = 2008" in normal_text


class TestMultiTurnProcessFn:
    def test_first_turn_returns_tool_calls(self):
        result = multi_turn_tool_call_process_fn(MULTI_TURN_FIRST_PROMPT)

        assert result.finish_reason == "stop"
        assert result.text == MULTI_TURN_FIRST_RESPONSE

    def test_second_turn_returns_answer(self):
        result = multi_turn_tool_call_process_fn('{"year": 2026}')

        assert result.finish_reason == "stop"
        assert result.text == MULTI_TURN_SECOND_RESPONSE

    def test_unexpected_prompt_raises(self):
        with pytest.raises(ValueError, match="Unexpected prompt"):
            multi_turn_tool_call_process_fn("some random input")


class TestEndToEndToolFlow:
    @pytest.fixture
    def parser(self):
        tools = TypeAdapter(list[Tool]).validate_python(SAMPLE_TOOLS)
        return FunctionCallParser(tools=tools, tool_call_parser="qwen25")

    def test_full_multi_turn_flow(self, parser):
        first_response = multi_turn_tool_call_process_fn(MULTI_TURN_FIRST_PROMPT)
        normal_text, calls = parser.parse_non_stream(first_response.text)

        assert len(calls) == 2
        tool_results = []
        for call in calls:
            params = json.loads(call.parameters) if call.parameters else {}
            result = execute_tool_call(call.name, params)
            tool_results.append({"name": call.name, "result": result})

        assert tool_results[0] == {"name": "get_year", "result": {"year": 2026}}
        assert tool_results[1] == {"name": "get_temperature", "result": {"temperature": -60}}

        tool_response_str = "\n".join(json.dumps(r["result"]) for r in tool_results)
        second_response = multi_turn_tool_call_process_fn(tool_response_str)

        assert second_response.text == MULTI_TURN_SECOND_RESPONSE
