"""Smoke test: does the configured GWDG model emit OpenAI-style tool calls via LiteLLM,
and does it correctly consume the tool result in a follow-up turn?

Run with:
    uv run python scripts/smoke_tool_calling.py
    uv run python scripts/smoke_tool_calling.py --model qwen2.5-72b-instruct
"""

import argparse
import json
import os

import litellm
from dotenv import load_dotenv

from machine_learning_engineering.shared_libraries.search_util import web_search

load_dotenv()

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for recent information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (1-10).",
                    },
                },
                "required": ["query"],
            },
        },
    }
]

MESSAGES = [
    {
        "role": "system",
        "content": (
            "You are a helpful assistant. If the user asks about recent "
            "events or information you don't know, call the web_search tool. "
            "After receiving search results, use them to write a concise answer."
        ),
    },
    {
        "role": "user",
        "content": "What are the most effective tabular ML models in 2025? Search the web.",
    },
]


def _print_tool_calls(tool_calls) -> None:
    for tc in tool_calls:
        print(f"  name: {tc.function.name}")
        try:
            args_obj = json.loads(tc.function.arguments)
            print(f"  args: {args_obj}")
        except Exception:
            print(f"  args (raw): {tc.function.arguments!r}")


def _execute_tool_call(tc) -> str:
    if tc.function.name != "web_search":
        return f"error: unknown tool {tc.function.name!r}"
    try:
        args_obj = json.loads(tc.function.arguments)
    except Exception as exc:
        return f"error: invalid JSON arguments: {exc!r}"
    return web_search(
        query=args_obj.get("query", ""),
        num_results=args_obj.get("num_results", 5),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=os.environ.get("ROOT_AGENT_MODEL", "llama-3.3-70b-instruct"),
    )
    args = parser.parse_args()

    api_base = os.environ["OPENAI_API_BASE"]
    api_key = os.environ["OPENAI_API_KEY"]
    model = args.model if args.model.startswith("openai/") else f"openai/{args.model}"

    print(f"Endpoint: {api_base}")
    print(f"Model:    {model}")
    print()

    # --- Turn 1: prompt → expect a tool call ---
    resp = litellm.completion(
        model=model,
        api_base=api_base,
        api_key=api_key,
        messages=MESSAGES,
        tools=TOOLS,
        tool_choice="auto",
        temperature=0.0,
    )
    msg = resp.choices[0].message
    tool_calls = getattr(msg, "tool_calls", None)
    content = getattr(msg, "content", None)

    print("=== Turn 1: assistant response ===")
    print(f"content:    {content!r}")
    print(f"tool_calls: {tool_calls}")
    print()

    if not tool_calls:
        print("FAIL: model produced text instead of a tool call on turn 1.")
        return 1

    print("Turn 1 PASS: model emitted a tool call.")
    _print_tool_calls(tool_calls)
    print()

    # --- Execute the tool(s) ---
    print("=== Executing web_search locally ===")
    tool_messages = []
    for tc in tool_calls:
        result = _execute_tool_call(tc)
        preview = result if len(result) < 400 else result[:400] + " …[truncated for log]"
        print(f"[{tc.function.name}] →\n{preview}\n")
        tool_messages.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.function.name,
                "content": result,
            }
        )

    # --- Turn 2: feed tool result back, expect a grounded text answer ---
    followup_messages = list(MESSAGES) + [msg.model_dump()] + tool_messages
    resp2 = litellm.completion(
        model=model,
        api_base=api_base,
        api_key=api_key,
        messages=followup_messages,
        tools=TOOLS,
        tool_choice="auto",
        temperature=0.0,
    )
    msg2 = resp2.choices[0].message
    final_content = getattr(msg2, "content", None) or ""
    further_tool_calls = getattr(msg2, "tool_calls", None)

    print("=== Turn 2: assistant final answer ===")
    if further_tool_calls:
        print("(model issued ANOTHER tool call instead of answering)")
        _print_tool_calls(further_tool_calls)
        print()
    print(final_content)
    print()

    if not final_content.strip() and not further_tool_calls:
        print("FAIL: turn 2 produced empty content and no tool call.")
        return 1

    print("Round-trip PASS: model received the tool result and responded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
