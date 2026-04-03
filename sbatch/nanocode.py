#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "openai>=2.21.0",
#   "prompt-toolkit>=3.0.52",
#   "python-dotenv>=1.0.1",
#   "rich>=13.0.0",
# ]
# ///
"""nanocode - minimal claude code alternative"""

from __future__ import annotations

import concurrent.futures
import glob as globlib
import json
import os
import re
import subprocess
import sys
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts import print_formatted_text
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown

load_dotenv()


def _ensure_windows_utf8() -> None:
    if os.name != "nt":
        return
    try:
        subprocess.run("chcp 65001 >NUL", shell=True, check=False)
    except Exception:
        pass
    for stream_name in ("stdout", "stderr", "stdin"):
        stream = getattr(sys, stream_name, None)
        if stream and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


_ensure_windows_utf8()

OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
MODEL = "gpt-5.4"
REASONING_EFFORT = "high"
MODEL_CONTEXT_TOKENS = 128000
MAX_PARALLEL_TOOLS = 8
MAX_TELEMETRY_OUTPUT_LINES = 10
READ_ONLY_TOOLS = {"read", "glob", "grep"}
CONSOLE = Console()
LAST_USAGE = ""
STYLE = Style.from_dict(
    {
        "hdr": "bold",
        "dim": "fg:#6c6c6c",
        "prompt": "bold fg:#4aa3ff",
        "ok": "fg:#34c759",
        "err": "fg:#ff5f56",
        "bullet": "fg:#5ac8fa",
    }
)


def pft(parts: Any, **kwargs: Any) -> None:
    formatted = FormattedText(parts) if isinstance(parts, list) else parts
    print_formatted_text(formatted, style=STYLE, **kwargs)


# --- Tool implementations ---


def read(args: dict[str, Any]) -> str:
    lines = open(args["path"]).readlines()
    offset = args.get("offset", 0)
    limit = args.get("limit", len(lines))
    selected = lines[offset : offset + limit]
    return "".join(
        f"{offset + idx + 1:4}| {line}" for idx, line in enumerate(selected)
    )


def write(args: dict[str, Any]) -> str:
    with open(args["path"], "w") as f:
        f.write(args["content"])
    return "ok"


def edit(args: dict[str, Any]) -> str:
    text = open(args["path"]).read()
    old, new = args["old"], args["new"]
    if old not in text:
        return "error: old_string not found"
    count = text.count(old)
    if not args.get("all") and count > 1:
        return f"error: old_string appears {count} times, must be unique (use all=true)"
    replacement = text.replace(old, new) if args.get("all") else text.replace(old, new, 1)
    with open(args["path"], "w") as f:
        f.write(replacement)
    return "ok"


def glob(args: dict[str, Any]) -> str:
    pattern = (args.get("path", ".") + "/" + args["pat"]).replace("//", "/")
    files = globlib.glob(pattern, recursive=True)
    files = sorted(
        files,
        key=lambda f: os.path.getmtime(f) if os.path.isfile(f) else 0,
        reverse=True,
    )
    return "\n".join(files) or "none"


def grep(args: dict[str, Any]) -> str:
    pattern = re.compile(args["pat"])
    hits = []
    for filepath in globlib.glob(args.get("path", ".") + "/**", recursive=True):
        try:
            for line_num, line in enumerate(open(filepath), 1):
                if pattern.search(line):
                    hits.append(f"{filepath}:{line_num}:{line.rstrip()}")
        except Exception:
            pass
    return "\n".join(hits[:50]) or "none"


def bash(args: dict[str, Any]) -> str:
    command = args["cmd"]
    try:
        completed = subprocess.run(
            command,
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
        )
        output = completed.stdout or ""
    except subprocess.TimeoutExpired as err:
        output = (err.stdout or "") + "\n(timed out after 30s)"
    lines_out = output.splitlines()
    for line in lines_out[:MAX_TELEMETRY_OUTPUT_LINES]:
        pft([("class:dim", f"  | {line.rstrip()}")])
    hidden = len(lines_out) - MAX_TELEMETRY_OUTPUT_LINES
    if hidden > 0:
        pft([("class:dim", f"  | ... ({hidden} more lines hidden)")])
    return output.strip() or "(empty)"


# --- Tool definitions: (description, schema, function) ---

TOOLS = {
    "read": (
        "Read file with line numbers (file path, not directory)",
        {"path": "string", "offset": "number?", "limit": "number?"},
        read,
    ),
    "write": (
        "Write content to file",
        {"path": "string", "content": "string"},
        write,
    ),
    "edit": (
        "Replace old with new in file (old must be unique unless all=true)",
        {"path": "string", "old": "string", "new": "string", "all": "boolean?"},
        edit,
    ),
    "glob": (
        "Find files by pattern, sorted by mtime",
        {"pat": "string", "path": "string?"},
        glob,
    ),
    "grep": (
        "Search files for regex pattern",
        {"pat": "string", "path": "string?"},
        grep,
    ),
    "bash": (
        "Run shell command",
        {"cmd": "string"},
        bash,
    ),
}


def run_tool(name: str, args: dict[str, Any]) -> str:
    try:
        return TOOLS[name][2](args)
    except Exception as err:
        return f"error: {err}"


def _tool_preview(result: str) -> str:
    result_lines = result.splitlines() or ["(empty)"]
    preview = result_lines[0][:60]
    if len(result_lines) > 1:
        preview += f" ... +{len(result_lines) - 1} lines"
    elif len(result_lines[0]) > 60:
        preview += "..."
    return preview


def _announce_tool(block: dict[str, Any]) -> None:
    arg_preview = str(next(iter(block["input"].values()), ""))[:50]
    pft(
        [
            ("class:ok", "\n* "),
            ("class:ok", block["name"].capitalize()),
            ("", "("),
            ("class:dim", arg_preview),
            ("", ")"),
        ]
    )
    if block["name"] == "bash":
        command = str(block["input"].get("cmd", "")).strip()
        command_preview = " ".join(command.split())[:140] if command else "(empty)"
        pft([("class:dim", f"  $ {command_preview}")])


def execute_tool_calls(tool_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tool_results = []
    read_only_batch = []

    def flush_read_only_batch() -> None:
        nonlocal read_only_batch
        if not read_only_batch:
            return
        for block in read_only_batch:
            _announce_tool(block)
        results_by_id = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(MAX_PARALLEL_TOOLS, len(read_only_batch))
        ) as executor:
            futures = {
                executor.submit(run_tool, block["name"], block["input"]): block
                for block in read_only_batch
            }
            for future in concurrent.futures.as_completed(futures):
                block = futures[future]
                try:
                    results_by_id[block["id"]] = future.result()
                except Exception as err:
                    results_by_id[block["id"]] = f"error: {err}"
        for block in read_only_batch:
            result = results_by_id.get(block["id"], "error: missing tool result")
            pft([("class:dim", f"  -> {_tool_preview(result)}")])
            tool_results.append(
                {
                    "type": "function_call_output",
                    "call_id": block["id"],
                    "output": result,
                }
            )
        read_only_batch = []

    for block in tool_blocks:
        if block["name"] in READ_ONLY_TOOLS:
            read_only_batch.append(block)
            continue

        flush_read_only_batch()
        _announce_tool(block)
        result = run_tool(block["name"], block["input"])
        pft([("class:dim", f"  -> {_tool_preview(result)}")])
        tool_results.append(
            {"type": "function_call_output", "call_id": block["id"], "output": result}
        )

    flush_read_only_batch()

    return tool_results


def make_schema() -> list[dict[str, Any]]:
    result = []
    for name, (description, params, _fn) in TOOLS.items():
        properties = {}
        required = []
        for param_name, param_type in params.items():
            is_optional = param_type.endswith("?")
            base_type = param_type.rstrip("?")
            properties[param_name] = {
                "type": "integer" if base_type == "number" else base_type
            }
            if not is_optional:
                required.append(param_name)
        input_schema = {
            "type": "object",
            "properties": properties,
            "required": required,
        }
        result.append(
            {
                "type": "function",
                "name": name,
                "description": description,
                "parameters": input_schema,
            }
        )
    return result


def call_api(
    client: OpenAI,
    input_items: list[dict[str, Any]],
    system_prompt: str,
    previous_response_id: str | None = None,
) -> dict[str, Any]:
    global LAST_USAGE
    tools = make_schema()
    tools.append({"type": "web_search"})
    payload = {
        "model": MODEL,
        "instructions": system_prompt,
        "reasoning": {"effort": REASONING_EFFORT},
        "input": input_items,
        "tools": tools,
        "tool_choice": "auto",
    }
    if previous_response_id:
        payload["previous_response_id"] = previous_response_id
    parsed = client.responses.create(**payload).model_dump()

    usage = parsed.get("usage") or {}
    total_tokens = usage.get("total_tokens") or (usage.get("input_tokens") or 0) + (
        usage.get("output_tokens") or 0
    )
    LAST_USAGE = (
        f"{total_tokens}/{MODEL_CONTEXT_TOKENS} ({total_tokens / MODEL_CONTEXT_TOKENS * 100:.1f}%)"
        if total_tokens
        else ""
    )

    content_blocks = []
    for item in parsed.get("output", []):
        item_type = item.get("type")
        if item_type == "message":
            text = ""
            for part in item.get("content", []):
                if part.get("type") in ("output_text", "text"):
                    text += part.get("text", "")
            if text.strip():
                content_blocks.append({"type": "text", "text": text.strip()})
        elif item_type == "function_call":
            arguments = item.get("arguments", "{}")
            try:
                tool_input = json.loads(arguments or "{}")
            except json.JSONDecodeError:
                tool_input = {}
            content_blocks.append(
                {
                    "type": "tool_use",
                    "id": item.get("call_id") or item.get("id"),
                    "name": item["name"],
                    "input": tool_input,
                }
            )

    return {"id": parsed.get("id"), "content": content_blocks}


def main() -> None:
    if not OPENAI_KEY:
        pft([("class:err", "* Missing API key. Set OPENAI_API_KEY in .env or environment.")])
        return

    client = OpenAI(api_key=OPENAI_KEY)
    pft(
        [
            ("class:hdr", "nanocode"),
            (
                "class:dim",
                f" | {MODEL} (reasoning={REASONING_EFFORT}) | {os.getcwd()}",
            ),
        ]
    )
    pft(
        [
            (
                "class:dim",
                "Enter submits. Ctrl+J or Esc+Enter inserts newline. /c clears, /q exits.",
            )
        ]
    )
    bindings = KeyBindings()

    @bindings.add("enter")
    def submit(event: Any) -> None:
        event.current_buffer.validate_and_handle()

    @bindings.add("c-j")
    @bindings.add("escape", "enter")
    def newline(event: Any) -> None:
        event.current_buffer.insert_text("\n")

    prompt_session = PromptSession(
        multiline=True,
        history=InMemoryHistory(),
        key_bindings=bindings,
        style=STYLE,
        prompt_continuation=lambda _w, _line_no, _soft_wrap: "... ",
        bottom_toolbar=lambda: [
            ("class:dim", f"Tokens: {LAST_USAGE}")
        ]
        if LAST_USAGE
        else "",
    )
    previous_response_id = None
    system_prompt = f"Concise coding assistant. cwd: {os.getcwd()}"

    while True:
        try:
            user_input = prompt_session.prompt([("class:prompt", "> ")])
            command = user_input.strip()
            if not command:
                continue
            if command in ("/q", "exit"):
                return
            if command == "/c":
                previous_response_id = None
                pft([("class:ok", "* Cleared conversation")])
                continue

            input_items = [{"role": "user", "content": user_input}]

            while True:
                response = call_api(client, input_items, system_prompt, previous_response_id)
                previous_response_id = response.get("id") or previous_response_id
                content_blocks = response.get("content", [])

                for block in content_blocks:
                    if block["type"] == "text":
                        pft([("class:bullet", "\n* ")], end="")
                        CONSOLE.print(Markdown(block["text"]))
                tool_blocks = [b for b in content_blocks if b["type"] == "tool_use"]
                tool_results = execute_tool_calls(tool_blocks)

                if not tool_results:
                    break
                input_items = tool_results

            print()

        except (KeyboardInterrupt, EOFError):
            return
        except Exception as err:
            pft([("class:err", f"* Error: {err}")])


if __name__ == "__main__":
    main()
