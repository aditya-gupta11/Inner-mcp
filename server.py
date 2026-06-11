"""
server.py — FastMCP server that exposes execute_task to Codex.
"""

import logging
import yaml
from pathlib import Path
from urllib.parse import unquote, urlparse

from fastmcp import Context, FastMCP
from handler import TaskHandler

config_path = Path(__file__).parent / "config.yaml"
config = yaml.safe_load(config_path.read_text())

log_file = Path(config["server"]["log_file"]).expanduser()
log_file.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=getattr(logging, config["server"]["log_level"].upper()),
    filename=log_file,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("server")

handler = TaskHandler(config)

mcp = FastMCP(
    name="inner-model-proxy",
    instructions=(
        "MANDATORY ROUTING RULES — follow these exactly:\n"
        "1. ANY browser task (navigate, click, fill form, read page, screenshot, "
        "check localhost:*, verify UI, test auth flow, inspect React app, "
        "check console errors) → call execute_task. NEVER use your own browser tools.\n"
        "2. ANY file operation (read, write, create, move, delete) → call execute_task.\n"
        "3. ANY git operation (diff, log, status, blame) → call execute_task.\n"
        "4. ANY shell/terminal inspection → call execute_task.\n"
        "5. UI/DESIGN comparison tasks: pass the reference design image as reference_image "
        "(base64 data-URI, file path, or URL). The inner model (qwen2.5-vl-72b) will "
        "visually compare it against the current UI and list the exact changes needed.\n"
        "6. ANY API testing task (send HTTP request, create Postman collection, run tests, "
        "generate test cases for an endpoint, check response schema) → call execute_task.\n"
        "The inner model handles all tool work internally via real MCP servers. "
        "You only receive the final answer. This saves your tokens on every tool-heavy task."
    ),
)


def _workspace_root_from_ctx(ctx: Context | None) -> Path | None:
    """
    Extract the workspace root from the MCP client's declared roots.

    FastMCP exposes ctx.roots as a list of Root objects (not a coroutine),
    each with a .uri attribute like "file:///home/user/my-project".
    Falls back to None if no file:// root is found or ctx is absent.
    """
    if ctx is None:
        return None
    try:
        roots = ctx.roots  # synchronous list[Root] in FastMCP
        if not roots:
            return None
        for root in roots:
            parsed = urlparse(str(root.uri))
            if parsed.scheme == "file":
                return Path(unquote(parsed.path)).resolve()
    except Exception as exc:
        log.debug("Could not read workspace roots from ctx: %s", exc)
    return None


@mcp.tool(
    description=(
        "BROWSER TASKS: navigate URLs, click elements, fill forms, read page content, "
        "take screenshots, check localhost:*, test React UI, verify auth flows, "
        "check console errors, test API endpoints from browser. "
        "FILE TASKS: read, write, create, move, delete files. "
        "GIT TASKS: diff, log, status, blame. "
        "SHELL TASKS: directory listing, file stats, process inspection. "
        "VISION/UI TASKS: pass a reference_image (design mockup, Figma export, or any "
        "screenshot) and the inner model will compare it against the live UI and list "
        "the exact CSS/JSX changes needed to match it. "
        "The inner model (qwen2.5-vl-72b) executes all tool work via Playwright MCP + "
        "Filesystem MCP internally. You receive only the final answer — no raw tool output. "
        "Returns: { answer: str, tools_used: list[str], iterations: int, "
        "confidence: 'high'|'medium'|'low', files_read: list[str], vision_used: bool }. "
        "API TESTING TASKS: send HTTP requests, create Postman collections, generate test "
        "cases for endpoints (happy path, auth, validation, edge cases), run existing "
        "collections, and inspect response schemas."
    )
)
async def execute_task(
    task: str,
    reference_image: str | None = None,
    ctx: Context | None = None,
) -> dict:
    """
    Delegate a task to qwen2.5-vl-72b. The inner model uses real MCP servers
    (Playwright, Filesystem) and returns a clean compressed answer.

    Args:
        task: Plain English description of what to do. Be specific:
              include URLs, file paths, expected values, what to verify.
        reference_image: Optional image for visual comparison. Pass a base64
              data-URI ("data:image/png;base64,..."), a file path, or an
              http/https URL. When provided the model will visually compare
              the image against the current UI before making any tool calls.

    Returns:
        { answer, tools_used, iterations, confidence, files_read,
          vision_used, cached?, error? }
    """
    workspace_root = _workspace_root_from_ctx(ctx)
    log.info(
        "Task received: chars=%s vision=%s workspace_root=%s preview=%r",
        len(task), reference_image is not None, workspace_root, task[:200],
    )

    result = await handler.run(
        task,
        reference_image=reference_image,
        workspace_root=workspace_root,
    )
    log.info(
        "Task complete: tools=%s iterations=%s answer_chars=%s confidence=%s "
        "vision=%s error=%s cached=%s",
        result.get("tools_used"),
        result.get("iterations"),
        len(result.get("answer", "")),
        result.get("confidence"),
        result.get("vision_used", False),
        result.get("error", False),
        result.get("cached", False),
    )
    return result


if __name__ == "__main__":
    mcp.run(transport=config["server"]["transport"])
