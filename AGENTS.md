# Agent Instructions for mcp-inner

## Tool Usage Policy

### ALWAYS use `execute_task` for:
- **Any browser task** — navigation, clicking, form filling, reading page content, screenshots, checking URLs, testing UI flows, verifying auth, inspecting React components, checking console errors
- **Any file operation** — reading, writing, creating, moving, deleting files
- **Any localhost testing** — checking http://localhost:*, verifying API responses, testing UI state
- **Any web scraping or data extraction** from a URL
- **Any git operation** — diff, log, status, blame
- **Any shell command** for inspection (file stats, directory listing, process checks)

### NEVER use your own tools when `execute_task` is available:
- Do NOT use your built-in browser/Playwright tools directly
- Do NOT use your built-in filesystem tools directly  
- Do NOT use your built-in shell tools directly
- ALWAYS delegate these to `execute_task` instead

### Why:
`execute_task` routes tasks to gpt-oss-120b (inner model) which uses real MCP servers
(Playwright MCP, Filesystem MCP) internally. You only see the final answer.
This saves your tokens significantly — you never see raw tool output.

## Task Delegation Format

When calling `execute_task`, describe the task in plain English with enough context:

**Good:**
```
execute_task("Navigate to http://localhost:5173, check if the login form is present, 
fill email=test@test.com and password=test123, click the login button, 
and tell me what happens after submission")
```

**Bad (too vague):**
```
execute_task("check the login")
```

**Bad (doing it yourself):**
```
playwright__browser_navigate(url="http://localhost:5173")  ← DON'T DO THIS
```

## When to NOT use execute_task

- Pure code generation (writing new code, refactoring) — do this yourself
- Answering questions from code already in context — answer directly  
- Simple string/math operations — do this yourself

## Project Context

This is the mcp-inner project itself — a FastMCP proxy server that delegates tool work
to qwen2.5-vl-72b. The inner model uses Playwright MCP and Filesystem MCP.
