"""
handler.py — The inner agentic loop.

Flow:
  1. Receive task from Codex (via server.py)
  2. Build initial message with system prompt
  3. If a reference_image is provided, attach it to the first user message
     so qwen2.5-vl-72b can do visual comparison before any tool calls.
  4. Loop:
     a. Call model (force_answer=True on final iteration)
     b. Parse response for tool calls
     c. Duplicate-call guard: skip tool if same (name, args) seen this run
     d. If tool call found: truncate content field in result, append, continue
     e. If final answer found: extract and return with metadata
     f. If model returns empty/blank: nudge once then continue
     g. If no tool call and no answer: nudge once then continue (not immediate exit)
     h. If max iterations reached: return what we have
"""

import re
import json
import time
import hashlib
import logging
import asyncio
from pathlib import Path
from model_client import ModelClient, build_vision_message
from prompt import build_system_prompt
from mcp_client import MCPClientPool

log = logging.getLogger("handler")

MAX_TOOL_RESULT_CHARS = 3000
RECENT_FULL_TOOL_RESULTS = 2
SUMMARY_PREVIEW_CHARS = 240

_task_cache: dict[str, tuple[float, dict]] = {}


# ── Parsing helpers ───────────────────────────────────────────────────────────

def _parse_tool_call(text: str) -> dict | None:
    match = re.search(r"```tool_call\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError as e:
            log.warning("Failed to parse tool call JSON: %s", e)
    return None


def _parse_answer(text: str) -> str | None:
    match = re.search(r"```answer\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


# ── Result formatting & compaction ───────────────────────────────────────────

def _truncate_result_content(result: dict) -> dict:
    if not isinstance(result, dict):
        return result
    if "content" in result and isinstance(result["content"], str):
        if len(result["content"]) > MAX_TOOL_RESULT_CHARS:
            result = dict(result)
            result["content"] = result["content"][:MAX_TOOL_RESULT_CHARS] + "\n...[truncated]"
            result["truncated"] = True
    return result


def _format_tool_result(tool_name: str, result: dict) -> str:
    return f"TOOL_RESULT [{tool_name}]:\n{json.dumps(result, indent=2)}"


def _estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4) if text else 0


def _preview(text: str, limit: int = 500) -> str:
    return " ".join((text or "").split())[:limit]


def _content_chars(content) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(len(b.get("text") or "") for b in content if isinstance(b, dict))
    return 0


def _message_stats(messages: list[dict]) -> dict:
    chars = sum(_content_chars(m.get("content")) for m in messages)
    return {"messages": len(messages), "chars": chars, "est_tokens": _estimate_tokens("x" * chars)}


def _summarize_tool_result_text(text: str) -> str:
    first_line, _, body = text.partition("\n")
    match = re.match(r"TOOL_RESULT \[(.*?)\]:", first_line)
    tool_name = match.group(1) if match else "unknown"

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return (
            f"TOOL_RESULT_SUMMARY [{tool_name}]: "
            f"chars={len(text)} preview={_preview(text, SUMMARY_PREVIEW_CHARS)!r}"
        )

    keys = list(parsed.keys()) if isinstance(parsed, dict) else []
    ok = parsed.get("ok") if isinstance(parsed, dict) else None
    error = parsed.get("error") if isinstance(parsed, dict) else None
    preview_value = ""
    if isinstance(parsed, dict):
        for key in ("summary", "content", "diff", "page_preview", "page_structure",
                    "text", "matches", "body", "stdout"):
            if key in parsed:
                preview_value = str(parsed[key])
                break

    parts = [
        f"TOOL_RESULT_SUMMARY [{tool_name}]:",
        f"ok={ok}", f"keys={keys[:10]}", f"original_chars={len(text)}",
    ]
    if error:
        parts.append(f"error={_preview(str(error), SUMMARY_PREVIEW_CHARS)!r}")
    if preview_value:
        parts.append(f"preview={_preview(preview_value, SUMMARY_PREVIEW_CHARS)!r}")
    return " ".join(parts)


def _compact_tool_history(messages: list[dict]) -> int:
    tool_result_indexes = [
        i for i, m in enumerate(messages)
        if m.get("role") == "user"
        and isinstance(m.get("content"), str)
        and m["content"].startswith("TOOL_RESULT [")
    ]
    compacted = 0
    for index in tool_result_indexes[:-RECENT_FULL_TOOL_RESULTS]:
        original = messages[index]["content"]
        summary = _summarize_tool_result_text(original)
        if summary != original:
            messages[index]["content"] = summary
            compacted += 1
            log.debug("Compacted tool result message[%s]: %s → %s chars", index, len(original), len(summary))
    return compacted


# ── Duplicate-call detection ─────────────────────────────────────────────────

def _call_key(tool_name: str, tool_args: dict) -> str:
    return hashlib.md5(
        json.dumps({"tool": tool_name, "args": tool_args}, sort_keys=True).encode()
    ).hexdigest()


# ── Task cache ────────────────────────────────────────────────────────────────

def _cache_key(scoped_task: str) -> str:
    return hashlib.md5(scoped_task.strip().encode()).hexdigest()


def _cache_get(scoped_task: str, ttl_seconds: int) -> dict | None:
    if ttl_seconds <= 0:
        return None
    key = _cache_key(scoped_task)
    if key in _task_cache:
        ts, result = _task_cache[key]
        if time.monotonic() - ts < ttl_seconds:
            return result
        del _task_cache[key]
    return None


def _cache_set(scoped_task: str, result: dict) -> None:
    _task_cache[_cache_key(scoped_task)] = (time.monotonic(), result)


# ── Format nudge messages ─────────────────────────────────────────────────────

_NUDGE_EMPTY = (
    "Your last response was empty. You MUST respond with exactly one of:\n"
    "1. A tool_call block to invoke a tool:\n"
    "```tool_call\n"
    '{{"tool": "tool_name", "args": {{}}}}\n'
    "```\n"
    "2. An answer block with your final answer:\n"
    "```answer\n"
    "Your answer here.\n"
    "```\n"
    "Output ONLY that block — no other text."
)

_NUDGE_BAD_FORMAT = (
    "Your last response did not use the required format. Do NOT write prose or pseudocode.\n"
    "You MUST output exactly one of these two blocks:\n\n"
    "Option A — call a tool:\n"
    "```tool_call\n"
    '{{"tool": "playwright__browser_navigate", "args": {{"url": "http://localhost:5173"}}}}\n'
    "```\n\n"
    "Option B — write your final answer:\n"
    "```answer\n"
    "Your answer here.\n"
    "```\n\n"
    "Output ONLY that block. Nothing else."
)


# ── Main handler ──────────────────────────────────────────────────────────────

class TaskHandler:

    def __init__(self, config: dict):
        self.config = config
        self.model = ModelClient(config)
        self.max_iterations = config["agent"]["max_iterations"]
        self.force_answer_after = config["agent"]["force_answer_after"]
        self.cache_ttl_seconds = config.get("cache", {}).get("ttl_seconds", 30)
        self.mcp_pool: MCPClientPool | None = None
        self._mcp_loop = None
        self._mcp_tool_names: set[str] = set()

    async def _initialize_mcp_servers(self, workspace_root: Path | None = None) -> None:
        mcp_configs = self.config.get("mcp_servers", [])
        if not mcp_configs:
            log.warning("No MCP servers configured in config.yaml")
            return

        self.mcp_pool = MCPClientPool(mcp_configs, workspace_root=workspace_root)
        self._mcp_tool_names.clear()
        log.info(
            "Starting %s MCP servers (workspace_root=%s)...",
            len(mcp_configs), self.mcp_pool.workspace_root,
        )
        await self.mcp_pool.start_all()
        self._mcp_loop = asyncio.get_running_loop()

        try:
            all_schemas = await self.mcp_pool.get_all_tool_schemas()
            for schema in all_schemas:
                self._mcp_tool_names.add(schema["function"]["name"])
            log.info("Loaded %s tools from MCP servers", len(all_schemas))
        except Exception as e:
            log.error("Failed to load MCP tool schemas: %s", e)

    async def run(
        self,
        task: str,
        reference_image=None,
        workspace_root: str | Path | None = None,
    ) -> dict:
        current_loop = asyncio.get_running_loop()

        requested_root = (
            Path(workspace_root).expanduser().resolve()
            if workspace_root is not None
            else None
        )

        pool_missing = self.mcp_pool is None
        loop_changed = self._mcp_loop is not current_loop
        root_changed = (
            requested_root is not None
            and self.mcp_pool is not None
            and self.mcp_pool.workspace_root != requested_root
        )

        if self.config.get("mcp_servers") and (pool_missing or loop_changed or root_changed):
            if root_changed and self.mcp_pool is not None:
                log.info(
                    "workspace_root changed (%s → %s), restarting MCP pool",
                    self.mcp_pool.workspace_root, requested_root,
                )
                await self.mcp_pool.close_all()
                self.mcp_pool = None
            await self._initialize_mcp_servers(workspace_root=requested_root)

        cache_scope = str(self.mcp_pool.workspace_root) if self.mcp_pool else str(requested_root or "")
        scoped_task = f"{cache_scope}\0{task}"

        if reference_image is None:
            cached = _cache_get(scoped_task, self.cache_ttl_seconds)
            if cached is not None:
                log.info("Cache hit for task (chars=%s)", len(task))
                return {**cached, "cached": True}

        tool_schemas = []
        if self.mcp_pool:
            try:
                tool_schemas = await self.mcp_pool.get_all_tool_schemas()
                self._mcp_tool_names = {s["function"]["name"] for s in tool_schemas}
            except Exception as e:
                log.error("Failed to get MCP tool schemas: %s", e)

        if not tool_schemas:
            return {
                "answer": (
                    "No MCP tools are available. Check the configured MCP servers "
                    "and server logs before retrying."
                ),
                "tools_used": [], "iterations": 0,
                "confidence": "low", "files_read": [], "error": True,
            }

        system_prompt = build_system_prompt(tool_schemas)

        if reference_image is not None:
            log.info("Vision task: reference_image provided (type=%s)", type(reference_image).__name__)
            try:
                first_user_msg = build_vision_message(f"Task: {task}", reference_image)
            except Exception as e:
                log.error("Failed to build vision message: %s", e)
                return {
                    "answer": f"Failed to attach reference image: {e}",
                    "tools_used": [], "iterations": 0,
                    "confidence": "low", "files_read": [], "error": True,
                }
        else:
            first_user_msg = {"role": "user", "content": f"Task: {task}"}

        messages = [
            {"role": "system", "content": system_prompt},
            first_user_msg,
        ]

        tools_used: list[str] = []
        files_read: list[str] = []
        seen_calls: set[str] = set()
        iterations = 0
        consecutive_bad_format = 0  # how many turns in a row the model missed the format

        log.info(
            "Run started: task_chars=%s vision=%s workspace_root=%s system_prompt_chars=%s",
            len(task), reference_image is not None, cache_scope, len(system_prompt),
        )

        while iterations < self.max_iterations:
            iterations += 1
            log.info("Iteration %s/%s", iterations, self.max_iterations)

            is_force_answer_turn = (iterations == self.force_answer_after)
            if is_force_answer_turn:
                messages.append({
                    "role": "user",
                    "content": (
                        "You have gathered enough information. "
                        "Stop calling tools. Write your final answer now using ONLY the answer block:\n"
                        "```answer\n"
                        "Your answer here.\n"
                        "```"
                    ),
                })

            compacted = _compact_tool_history(messages)
            stats = _message_stats(messages)
            log.info(
                "Before model call: iteration=%s compacted=%s messages=%s chars=%s est_tokens=%s",
                iterations, compacted, stats["messages"], stats["chars"], stats["est_tokens"],
            )

            try:
                response = await self.model.chat(
                    messages,
                    tool_schemas=tool_schemas,
                    force_answer=is_force_answer_turn,
                )
            except Exception as e:
                log.error("Model call failed: %s", e)
                return {
                    "answer": f"The inner model failed: {e}. Check server logs.",
                    "tools_used": tools_used, "iterations": iterations,
                    "error": True, "confidence": "low",
                }

            log.info("After model call: iteration=%s response_chars=%s preview=%r",
                     iterations, len(response), response[:200])
            messages.append({"role": "assistant", "content": response})

            # ── Check for final answer ──────────────────────────────────────
            answer = _parse_answer(response)
            if answer:
                consecutive_bad_format = 0
                log.info("Got final answer after %s iterations", iterations)
                result = {
                    "answer": answer,
                    "tools_used": tools_used,
                    "iterations": iterations,
                    "confidence": "high" if iterations <= 3 else "medium",
                    "files_read": list(set(files_read)),
                    "vision_used": reference_image is not None,
                }
                if reference_image is None:
                    _cache_set(scoped_task, result)
                return result

            # ── Check for tool call ─────────────────────────────────────────
            tool_call = _parse_tool_call(response)
            if tool_call:
                consecutive_bad_format = 0
                tool_name = tool_call.get("tool")
                tool_args = tool_call.get("args", {})

                log.info(
                    "Parsed tool call: iteration=%s tool=%s arg_keys=%s",
                    iterations, tool_name,
                    list(tool_args.keys()) if isinstance(tool_args, dict) else [],
                )

                call_key = _call_key(tool_name, tool_args)
                if call_key in seen_calls:
                    log.warning("Duplicate tool call detected: tool=%s — skipping", tool_name)
                    messages.append({
                        "role": "user",
                        "content": (
                            f"TOOL_RESULT [{tool_name}]: "
                            "SKIPPED — identical call already made. "
                            "Use the previous result or call a different tool."
                        ),
                    })
                    continue
                seen_calls.add(call_key)

                if tool_name in self._mcp_tool_names and self.mcp_pool:
                    log.info("Calling MCP tool: %s", tool_name)
                    tools_used.append(tool_name)
                    if tool_name.startswith("filesystem__"):
                        files_read.append(str(
                            tool_args.get("path") or tool_args.get("directory")
                            or tool_args.get("root") or ""
                        ))
                    try:
                        tool_result = await self.mcp_pool.call_tool(tool_name, tool_args)
                    except Exception as e:
                        tool_result = {"ok": False, "error": str(e)}
                        log.error("MCP tool call failed: %s", e)
                else:
                    tool_result = {"ok": False, "error": f"Unknown tool: {tool_name}"}

                screenshot_bytes = None
                if isinstance(tool_result, dict):
                    screenshot_bytes = tool_result.pop("screenshot_bytes", None)

                tool_result = _truncate_result_content(tool_result)
                result_text = _format_tool_result(tool_name, tool_result)

                log.info(
                    "Tool result: iteration=%s tool=%s ok=%s raw_chars=%s",
                    iterations, tool_name,
                    tool_result.get("ok") if isinstance(tool_result, dict) else None,
                    len(result_text),
                )

                if screenshot_bytes:
                    log.info("Attaching screenshot from tool result: %s bytes", len(screenshot_bytes))
                    messages.append(build_vision_message(result_text, screenshot_bytes))
                else:
                    messages.append({"role": "user", "content": result_text})

                continue

            # ── Model returned empty or wrong format — nudge, don't exit ───
            consecutive_bad_format += 1
            log.warning(
                "Bad format response: iteration=%s consecutive=%s response_chars=%s preview=%r",
                iterations, consecutive_bad_format, len(response), response[:300],
            )

            # Hard exit after 2 consecutive bad format responses
            if consecutive_bad_format >= 2:
                log.error("Model failed format twice in a row — giving up")
                # If the model wrote something useful (just not in a block), surface it
                fallback_answer = response.strip() if response.strip() else (
                    "The model did not produce a usable response after multiple attempts."
                )
                return {
                    "answer": fallback_answer,
                    "tools_used": tools_used,
                    "iterations": iterations,
                    "confidence": "low",
                    "files_read": list(set(files_read)),
                    "vision_used": reference_image is not None,
                    "warning": "Model failed format compliance after nudge",
                }

            # First bad response — inject the appropriate nudge and retry
            nudge = _NUDGE_EMPTY if not response.strip() else _NUDGE_BAD_FORMAT
            messages.append({"role": "user", "content": nudge})
            log.info("Injected format nudge, retrying...")
            continue

        log.warning("Max iterations (%s) reached without answer", self.max_iterations)
        return {
            "answer": (
                "Reached the maximum number of steps without completing the task. "
                "Try rephrasing or breaking it into smaller parts."
            ),
            "tools_used": tools_used, "iterations": iterations,
            "confidence": "low", "files_read": list(set(files_read)),
            "vision_used": reference_image is not None, "error": True,
        }
