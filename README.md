# Anvil

Multimodal web chat for local OpenAI-compatible model servers (vLLM and
friends). One UI, many models: streaming chat, image upload, tool calling via
MCP, per-message thinking-level control, and automatic context compaction.

## Features
- **Model dropdown** — any number of OpenAI-compatible endpoints in
  `config.json` (per-model context window, input modalities, tool gate).
- **Thinking control** — `off / low / medium / high` per message (maps to the
  hybrid thinking template contract: `enable_thinking=false`,
  `reasoning_effort low|medium`, or omitted for the server default). Thinking
  streams live as a collapsible block, collapses to `💭 thinking (N tok)`
  when done.
- **Media upload** — images go into the prompt natively (any multimodal
  model); other files are attached by path so file-reading models can read
  them.
- **MCP tools** — built-in two-transport MCP client:
  - streamable-HTTP (JSON-RPC 2.0 over POST)
  - stdio subprocess
  Toggle "MCP tools" in the header; per-model `tools` flag; per-server status
  dots; `/api/mcp/refresh` to reconnect.
- **Context compaction** — token estimate per message (chars/4 + 1600/image);
  auto-compacts at 80% of the context window and on context-limit errors
  (summarizes older history via the model, keeps the 6 most recent turns).
  Manual 🗜 Compact button; live context % meter.
- **Reset** — clears session history server-side and in the UI.
- Streaming over SSE, tool-call chips, no frontend build step
  (Flask + vanilla JS in one template).

## Run
```bash
./run-anvil.sh                # port 8590 by default
ANVIL_PORT=9000 ./run-anvil.sh
```
`run-anvil.sh` creates a venv and installs `flask` + `httpx` if missing.

Point `config.json` `models[].base_url` at your model servers. Models that
speak the hybrid thinking template contract get the thinking dropdown; others
simply ignore the extra params.

## Endpoints
| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | chat UI |
| `/health` | GET | liveness |
| `/api/models` | GET | model list |
| `/api/status` | GET | MCP server status |
| `/api/chat` | POST | `{model, message, media[], tools, thinking}` → SSE stream |
| `/api/upload` | POST | multipart `file` → media record |
| `/api/reset` | POST | clear session |
| `/api/compact` | POST | force compaction now |
| `/api/mcp/refresh` | POST | reconnect MCP servers |

SSE event types: `token`, `thinking`, `tool_call`, `tool_result`,
`compacted`, `status`, `done` (carries token estimate + thinking size),
`error`.

## Layout
```
app.py               # Flask server: sessions, chat loop, compaction, media
mcp_client.py        # self-contained MCP client (streamable-HTTP + stdio)
config.json          # models, MCP servers, compaction knobs
templates/chat.html  # the entire frontend (vanilla JS)
run-anvil.sh         # launcher (venv + deps + port)
```

## Notes
- Sessions are in-memory (per model id, locked); the reset button is the
  lifecycle. Uploaded media live in `media/` until process restart.
- Assistant turns store `reasoning_content` so multi-turn replay matches what
  vLLM expects for thinking models.
- Dev server is fine for single-user LAN use; put gunicorn behind it for
  anything else.
