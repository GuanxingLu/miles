import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.deepseekv3_detector import DeepSeekV3Detector
from sglang.srt.function_call.function_call_parser import FunctionCallParser


class TestSGLangToolCallParser(unittest.TestCase):
    """
    Demonstrates sglang's tool call parser usage
    """

    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get current weather for a city",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                        },
                        "required": ["city"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="search",
                    description="Search for information",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                        },
                        "required": ["query"],
                    },
                ),
            ),
        ]

    def test_detect_and_parse_single_tool_call(self):
        """Test parsing a single tool call in DeepSeek V3 format (non-streaming)."""
        detector = DeepSeekV3Detector()

        model_output = (
            "<｜tool▁calls▁begin｜>"
            "<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
            '```json\n{"city": "Beijing", "unit": "celsius"}\n```'
            "<｜tool▁call▁end｜>"
            "<｜tool▁calls▁end｜>"
        )

        assert detector.has_tool_call(model_output), "Should detect tool call markers"

        result = detector.detect_and_parse(model_output, self.tools)

        assert len(result.calls) == 1, "Should parse exactly one tool call"
        assert result.calls[0].name == "get_weather"
        params = json.loads(result.calls[0].parameters)
        assert params["city"] == "Beijing"
        assert params["unit"] == "celsius"

    def test_detect_and_parse_multiple_tool_calls(self):
        """Test parsing multiple parallel tool calls in DeepSeek V3 format."""
        detector = DeepSeekV3Detector()

        model_output = (
            "<｜tool▁calls▁begin｜>"
            "<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
            '```json\n{"city": "Shanghai"}\n```'
            "<｜tool▁call▁end｜>\n"
            "<｜tool▁call▁begin｜>function<｜tool▁sep｜>search\n"
            '```json\n{"query": "restaurants in Shanghai"}\n```'
            "<｜tool▁call▁end｜>"
            "<｜tool▁calls▁end｜>"
        )

        result = detector.detect_and_parse(model_output, self.tools)

        assert len(result.calls) == 2, "Should parse two tool calls"

        assert result.calls[0].name == "get_weather"
        params0 = json.loads(result.calls[0].parameters)
        assert params0["city"] == "Shanghai"

        assert result.calls[1].name == "search"
        params1 = json.loads(result.calls[1].parameters)
        assert params1["query"] == "restaurants in Shanghai"

    def test_text_before_tool_call(self):
        """Test that normal text before tool calls is preserved as normal_text."""
        detector = DeepSeekV3Detector()

        model_output = (
            "Let me check the weather for you.\n"
            "<｜tool▁calls▁begin｜>"
            "<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
            '```json\n{"city": "Tokyo"}\n```'
            "<｜tool▁call▁end｜>"
            "<｜tool▁calls▁end｜>"
        )

        result = detector.detect_and_parse(model_output, self.tools)

        assert result.normal_text == "Let me check the weather for you."
        assert len(result.calls) == 1
        assert result.calls[0].name == "get_weather"

    def test_no_tool_call_returns_original_text(self):
        """Test that text without tool calls is returned as normal_text."""
        detector = DeepSeekV3Detector()

        model_output = "The weather in Beijing is sunny today with a high of 25°C."

        assert not detector.has_tool_call(model_output)

        result = detector.detect_and_parse(model_output, self.tools)

        assert result.normal_text == model_output
        assert len(result.calls) == 0

    def test_using_function_call_parser_wrapper(self):
        """
        Test using FunctionCallParser as a high-level wrapper.

        FunctionCallParser provides a unified interface for different model formats.
        Supported parsers: deepseekv3, qwen25, llama3, mistral, pythonic, etc.
        """
        parser = FunctionCallParser(tools=self.tools, tool_call_parser="deepseekv3")

        model_output = (
            "<｜tool▁calls▁begin｜>"
            "<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
            '```json\n{"city": "Paris"}\n```'
            "<｜tool▁call▁end｜>"
            "<｜tool▁calls▁end｜>"
        )

        assert parser.has_tool_call(model_output)

        normal_text, tool_calls = parser.parse_non_stream(model_output)

        assert normal_text == ""
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "get_weather"
        params = json.loads(tool_calls[0].parameters)
        assert params["city"] == "Paris"


if __name__ == "__main__":
    unittest.main()
