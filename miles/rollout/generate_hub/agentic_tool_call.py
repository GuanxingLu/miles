"""
Simple agentic demo with tool calling.
"""

import argparse
from copy import deepcopy

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_hub.generate_endpoint_wrapper import (
    compute_prompt_ids_from_sample,
    compute_request_payload,
    update_sample_from_response,
)
from miles.rollout.generate_hub.tool_call_utils import (
    create_tool_call_parser,
    execute_tool_calls,
    update_sample_with_tool_responses,
)
from miles.utils.http_utils import post
from miles.utils.misc import load_function


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    TODO


def _add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--generate-max-turns", type=int, default=16)
    parser.add_argument("--generate-tool-specs-path", type=str)
    parser.add_argument("--generate-tool-call-parser", type=str)
    parser.add_argument("--generate-execute-tool-function-path", type=str)
    parser.add_argument("--generate-multi-samples", action="store_true")


generate.add_arguments = _add_arguments


class _ToolCallAgent:
    """Imagine this is a black-box agent that does arbitrarily complex work."""
    async def run(self):
        TODO
