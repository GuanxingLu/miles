from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.core_types import ToolCallItem
from sglang.srt.function_call.function_call_parser import FunctionCallParser


class TestSGLangFunctionCallParser:
    """FunctionCallParser supports: deepseekv3, qwen25, llama3, mistral, pythonic, etc."""

    SAMPLE_TOOLS = [
        Tool(
            type="function",
            function=Function(
                name="get_weather",
                description="Get current weather for a city",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
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
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            ),
        ),
    ]

    QWEN3_SINGLE_TOOL_CALL = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'

    QWEN3_MULTI_TOOL_CALLS = (
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Shanghai"}}\n</tool_call>\n'
        '<tool_call>\n{"name": "search", "arguments": {"query": "restaurants"}}\n</tool_call>'
    )

    def test_single_tool_call(self):
        parser = FunctionCallParser(tools=self.SAMPLE_TOOLS, tool_call_parser="qwen25")

        assert parser.has_tool_call(self.QWEN3_SINGLE_TOOL_CALL)

        normal_text, tool_calls = parser.parse_non_stream(self.QWEN3_SINGLE_TOOL_CALL)

        assert (normal_text, tool_calls) == (
            "",
            [ToolCallItem(tool_index=0, name="get_weather", parameters='{"city": "Paris"}')],
        )

    def test_multi_tool_calls(self):
        parser = FunctionCallParser(tools=self.SAMPLE_TOOLS, tool_call_parser="qwen25")

        normal_text, tool_calls = parser.parse_non_stream(self.QWEN3_MULTI_TOOL_CALLS)

        assert (normal_text, tool_calls) == (
            "",
            [
                ToolCallItem(tool_index=0, name="get_weather", parameters='{"city": "Shanghai"}'),
                ToolCallItem(tool_index=1, name="search", parameters='{"query": "restaurants"}'),
            ],
        )

    def test_no_tool_call(self):
        parser = FunctionCallParser(tools=self.SAMPLE_TOOLS, tool_call_parser="qwen25")
        model_output = "The weather is sunny today."

        assert not parser.has_tool_call(model_output)

        normal_text, tool_calls = parser.parse_non_stream(model_output)

        assert (normal_text, tool_calls) == ("The weather is sunny today.", [])
