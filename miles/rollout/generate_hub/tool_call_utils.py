from typing import Any


DUMMY_USER = {"role": "user", "content": "dummy"}
DUMMY_ASSISTANT = {
    "role": "assistant",
    "content": None,
    "tool_calls": [
        {
            "id": "call_dummy",
            "type": "function",
            "function": {
                "name": "dummy_func",
                "arguments": "{}",
            },
        }
    ],
}


def tokenize_tool_response(
    message: dict[str, Any],
    tokenizer,
) -> list[int]:
    messages_with_tool = [DUMMY_USER, DUMMY_ASSISTANT, message]
    messages_without_tool = [DUMMY_USER, DUMMY_ASSISTANT]

    tokens_with_tool = tokenizer.apply_chat_template(
        messages_with_tool, tokenize=True, add_generation_prompt=False
    )
    tokens_without_tool = tokenizer.apply_chat_template(
        messages_without_tool, tokenize=True, add_generation_prompt=False
    )

    assert tokens_with_tool[: len(tokens_without_tool)] == tokens_without_tool, (
        "Token prefix mismatch: the tokens without tool should be a prefix of tokens with tool"
    )

    return tokens_with_tool[len(tokens_without_tool) :]
