"""
model_client.py — Calls the qwen2.5-vl-72b streaming endpoint.

Tool schemas are not built from a static registry. chat() accepts the live
schema list discovered from real MCP servers.
- force_answer=True still works: sends tools=[], tool_choice="none" to save tokens.
- finish_reason "tool_calls" handled (stream break).
"""

import base64
import json
import httpx
import logging
from pathlib import Path

log = logging.getLogger("model_client")


def build_vision_message(text: str, image_source: str | bytes | Path) -> dict:
    """Build a user message containing text and an image."""
    if isinstance(image_source, bytes):
        encoded = base64.b64encode(image_source).decode()
        url = f"data:image/png;base64,{encoded}"
    elif isinstance(image_source, (str, Path)):
        path = Path(image_source)
        if path.exists():
            raw = path.read_bytes()
            suffix = path.suffix.lower().lstrip(".")
            mime = f"image/{'jpeg' if suffix in ('jpg', 'jpeg') else suffix or 'png'}"
            url = f"data:{mime};base64,{base64.b64encode(raw).decode()}"
        elif isinstance(image_source, str) and image_source.startswith(
            ("data:", "http://", "https://")
        ):
            url = image_source
        else:
            raise ValueError(f"Invalid image path or URL: {image_source!r}")
    else:
        raise TypeError(f"Unsupported image source type: {type(image_source)}")

    return {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": text},
        ],
    }


def _normalize_messages(messages: list[dict]) -> list[dict]:
    return [
        {**message, "content": ""}
        if message.get("content") is None
        else message
        for message in messages
    ]


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            block.get("text") or ""
            for block in content
            if isinstance(block, dict)
        )
    return ""


class ModelClient:

    def __init__(self, config: dict):
        self.base_url = config["model"]["base_url"].rstrip("/")
        self.model_name = config["model"]["model_name"]
        self.max_tokens = config["model"]["max_tokens"]
        self.temperature = config["model"]["temperature"]
        self.timeout = config["model"]["timeout_seconds"]

    def _headers(self) -> dict:
        return {"Content-Type": "application/json"}

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, (len(text) + 3) // 4) if text else 0

    @staticmethod
    def _preview(text: str, limit: int = 500) -> str:
        return " ".join((text or "").split())[:limit]

    async def chat(
        self,
        messages: list[dict],
        tool_schemas: list[dict],
        force_answer: bool = False,
        _retry_reasoning_only: bool = True,
    ) -> str:
        """
        Send messages to the streaming endpoint.

        tool_schemas: current live MCP schema list. Passed in per-call so the
                      handler controls the tool surface.
        force_answer: if True, sends tools=[] and tool_choice="none" to skip
                      tool calling on the final wrap-up turn (saves schema tokens).
        """
        messages = _normalize_messages(messages)
        message_texts = [_content_text(message.get("content")) for message in messages]
        message_chars = sum(len(text) for text in message_texts)

        if force_answer:
            schema_to_send: list[dict] = []
            tool_choice = "none"
            schema_chars = 0
        else:
            schema_to_send = tool_schemas
            tool_choice = "auto"
            schema_chars = len(json.dumps(tool_schemas, separators=(",", ":")))

        log.info(
            "Model request: model=%s messages=%s message_chars=%s est_message_tokens=%s "
            "tool_schemas=%s schema_chars=%s est_schema_tokens=%s max_tokens=%s force_answer=%s",
            self.model_name, len(messages), message_chars,
            self._estimate_tokens("".join(message_texts)),
            len(schema_to_send), schema_chars,
            self._estimate_tokens(" " * schema_chars),
            self.max_tokens, force_answer,
        )
        for i, m in enumerate(messages):
            content = _content_text(m.get("content"))
            log.debug(
                "Input message[%s]: role=%s chars=%s est_tokens=%s preview=%r",
                i, m.get("role"), len(content),
                self._estimate_tokens(content), self._preview(content),
            )

        payload = {
            "model": self.model_name,
            "stream": True,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "tools": schema_to_send,
            "tool_choice": tool_choice,
        }

        content_parts: list[str] = []
        reasoning_chars = 0
        tool_call_parts: dict[int, dict] = {}
        chunk_count = 0
        finish_reason = None
        usage = None

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    headers=self._headers(),
                ) as response:

                    # ── Error status: must read body inside the stream context ──
                    # httpx streaming responses buffer nothing until iterated.
                    # Calling response.raise_for_status() then accessing
                    # e.response.text in the outer except block raises
                    # "ResponseNotRead". Read the body here while the connection
                    # is still open, then raise with the text in hand.
                    if response.status_code >= 400:
                        error_body = await response.aread()
                        error_text = error_body.decode(errors="replace")
                        raise RuntimeError(
                            f"Model API {response.status_code}: {error_text}"
                        )

                    async for raw_line in response.aiter_lines():
                        line = raw_line.strip()
                        if not line.startswith("data:"):
                            continue
                        data_str = line[len("data:"):].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                        except json.JSONDecodeError:
                            log.warning("Could not parse chunk: %s", data_str[:80])
                            continue

                        chunk_count += 1
                        if chunk.get("usage"):
                            usage = chunk["usage"]

                        choices = chunk.get("choices", [])
                        if not choices:
                            continue

                        delta = choices[0].get("delta", {})
                        finish_reason = choices[0].get("finish_reason")

                        if delta.get("content"):
                            content_parts.append(delta["content"])
                        if delta.get("reasoning_content"):
                            reasoning_chars += len(delta["reasoning_content"])

                        # Assemble tool call deltas across chunks
                        for tc in delta.get("tool_calls", []):
                            idx = tc.get("index", 0)
                            if idx not in tool_call_parts:
                                tool_call_parts[idx] = {"id": "", "name": "", "arguments": ""}
                            if tc.get("id"):
                                tool_call_parts[idx]["id"] += tc["id"]
                            fn = tc.get("function", {})
                            if fn.get("name"):
                                tool_call_parts[idx]["name"] += fn["name"]
                            if fn.get("arguments"):
                                tool_call_parts[idx]["arguments"] += fn["arguments"]

                        if finish_reason in ("stop", "tool_calls"):
                            break

        except RuntimeError:
            raise  # already formatted above — pass through
        except httpx.ConnectError:
            raise RuntimeError(f"Could not connect to model at {self.base_url}")
        except httpx.TimeoutException:
            raise RuntimeError(f"Model timed out after {self.timeout}s")
        except httpx.HTTPStatusError as e:
            # Fallback: raised outside the stream context (shouldn't happen now,
            # but keeps the handler from seeing a raw httpx exception).
            try:
                body = e.response.text
            except Exception:
                body = "(body unavailable)"
            raise RuntimeError(f"Model API {e.response.status_code}: {body}")

        # Convert native tool calls → ```tool_call blocks for handler.py
        if tool_call_parts:
            blocks = []
            for idx in sorted(tool_call_parts.keys()):
                tc = tool_call_parts[idx]
                try:
                    args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                except json.JSONDecodeError:
                    args = {"raw": tc["arguments"]}
                block = json.dumps({"tool": tc["name"], "args": args}, indent=2)
                blocks.append(f"```tool_call\n{block}\n```")

            prefix = "".join(content_parts).strip()
            final_text = (prefix + "\n" + "\n".join(blocks)).strip()
            log.info(
                "Model response: chunks=%s finish=%s content_chars=%s tool_calls=%s "
                "returned_chars=%s est_tokens=%s usage=%s",
                chunk_count, finish_reason, len(prefix), len(tool_call_parts),
                len(final_text), self._estimate_tokens(final_text), usage,
            )
            return final_text

        final_text = "".join(content_parts).strip()
        if not final_text and reasoning_chars:
            log.warning(
                "Model produced reasoning-only stream: chunks=%s finish=%s "
                "reasoning_chars=%s content_chars=0 tool_calls=0 usage=%s",
                chunk_count, finish_reason, reasoning_chars, usage,
            )

            if _retry_reasoning_only:
                retry_messages = [
                    *messages,
                    {
                        "role": "user",
                        "content": (
                            "Your last response contained only hidden reasoning. "
                            "Retry now with a visible response only: either call one MCP tool "
                            "or write the final answer in the required ```answer block. "
                            "Do not emit reasoning-only output."
                        ),
                    },
                ]
                log.info(
                    "Retrying reasoning-only response once: original_messages=%s retry_messages=%s",
                    len(messages), len(retry_messages),
                )
                return await self.chat(
                    retry_messages,
                    tool_schemas=tool_schemas,
                    force_answer=force_answer,
                    _retry_reasoning_only=False,
                )

            raise RuntimeError(
                "Model produced only reasoning_content and no final content or tool call. "
                "The endpoint responded, but not in a usable chat/tool-call format."
            )

        log.info(
            "Model response: chunks=%s finish=%s content_chars=%s tool_calls=0 "
            "reasoning_chars=%s returned_chars=%s est_tokens=%s usage=%s",
            chunk_count, finish_reason, len(final_text),
            reasoning_chars,
            len(final_text), self._estimate_tokens(final_text), usage,
        )
        return final_text
