#!/usr/bin/env python3
"""Anvil — multimodal web chat for local OpenAI-compatible models.

Features:
- model dropdown (config.json, OpenAI-compatible endpoints)
- SSE streaming chat, agentic tool loop via MCP servers
- media upload (images inline; other files referenced by path)
- per-session context compaction (auto on limit, manual button)
- reset (clears session history)
"""
import base64
import json
import mimetypes
import os
import re
import threading
import time
import uuid

import httpx
from flask import Flask, jsonify, render_template, request

from mcp_client import McpManager, McpError

import vcc_compact as V

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MEDIA_DIR = os.path.join(APP_DIR, "media")
CONFIG_PATH = os.path.join(APP_DIR, "config.json")
os.makedirs(MEDIA_DIR, exist_ok=True)

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

MAX_TOOL_ROUNDS = 12  # fallback if config.json has no tools.max_rounds


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


CONFIG = load_config()
MCP = McpManager()
MCP.load(CONFIG)

MEDIA = {}  # id -> record
MEDIA_LOCK = threading.Lock()


def get_session(sid="main"):
    with SESSIONS_LOCK:
        if sid not in SESSIONS:
            SESSIONS[sid] = Session()
        return SESSIONS[sid]


class Session:
    def __init__(self):
        self.history = []
        self.compactions = 0
        self.lock = threading.Lock()
        # vcc: full transcript log (survives compactions) + recall
        self.log = []
        self.log_n = 0
        self.log_offset = 0
        self.vcc_sections = None
        self.mode = "summary"


SESSIONS = {"main": Session()}
SESSIONS_LOCK = threading.Lock()


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

def estimate_tokens(messages, image_tokens):
    n = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            n += len(c) // 4
        elif isinstance(c, list):
            for part in c:
                if part.get("type") == "text":
                    n += len(part.get("text", "")) // 4
                elif part.get("type") == "image_url":
                    n += image_tokens
        for tc in m.get("tool_calls") or []:
            f = tc.get("function", {})
            n += (len(f.get("name", "")) + len(f.get("arguments", ""))) // 4
    return n


def flatten_transcript(messages):
    out = []
    for m in messages:
        role = m.get("role")
        c = m.get("content")
        if isinstance(c, list):
            c = " | ".join(
                p.get("text", "<image>") for p in c
            )
        if m.get("tool_calls"):
            c = (c or "") + " | tool_calls: " + ", ".join(
                tc.get("function", {}).get("name", "?")
                for tc in m["tool_calls"]
            )
        out.append(f"[{role}] {c}")
    return "\n".join(out)


def is_context_error(err):
    e = err.lower()
    return any(
        k in e
        for k in (
            "context", "maximum context", "too long", "too many tokens",
            "exceed", "length", "max tokens", "prompt is too long",
        )
    )


def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def log_entry(sess, role, text, reasoning="", tool_calls=None):
    e = {
        "idx": sess.log_n,
        "role": role,
        "text": (text or "")[:4000],
        "reasoning": (reasoning or "")[:2000],
        "tool_calls": [
            {"name": (t.get("function", t) or {}).get("name", "?"),
             "args": ((t.get("function", t) or {}).get(
                 "arguments") or t.get("args", ""))[:500]}
            for t in (tool_calls or [])
        ],
    }
    sess.log.append(e)
    sess.log_n += 1
    return e


def format_recall(r):
    out = []
    if r.get("touched"):
        out.append("paths (entry #s):")
        for t in r["touched"][:40]:
            out.append(f"  {t['path']}  #{'/ #'.join(map(str, t['entries']))}")
    for e in r.get("expanded") or []:
        out.append(f"--- #{e['idx']} [{e['role']}] ---")
        out.append((e.get("text") or "")[:3000])
        if e.get("reasoning"):
            out.append("  reasoning: " + e["reasoning"][:1000])
        for tc in e.get("tool_calls") or []:
            out.append(f"  * {tc['name']} {tc['args'][:200]}")
    if r.get("hits"):
        out.append(f"hits ({r['total']} total):")
        for h in r["hits"]:
            out.append(f"  #{h['idx']} [{h['role']}] (score {h['score']}) {h['snippet']}")
    if not out:
        out.append("no matches")
    return "\n".join(out)


def do_compact(sess, model, image_tokens, emit, mode="summary"):
    """Compact older history. mode: 'summary' (LLM) or 'vcc' (algorithmic). Returns True if compacted."""
    keep = CONFIG.get("compact", {}).get("keep_recent", 6)
    if len(sess.history) <= keep + 2:
        return False
    if mode == "vcc":
        old = sess.log[sess.log_offset: max(sess.log_offset, len(sess.log) - keep)]
        if len(old) < 2:
            return False
        summary, sess.vcc_sections = V.compact_vcc(old, sess.vcc_sections, keep=0)
        if not summary:
            return False
        sess.log_offset = max(sess.log_offset, len(sess.log) - keep)
    else:
        old = sess.history[:-keep]
        prompt_msgs = [
            {
                "role": "system",
                "content": (
                    "You compact chat history into a faithful, dense summary. Preserve: "
                    "user goals, decisions, facts, numbers, file paths, error messages, "
                    "current task state, open questions. Plain text, no preamble."
                ),
            },
            {"role": "user", "content": flatten_transcript(old)[:180000]},
        ]
        ct = CONFIG.get("compact", {}).get("max_tokens", 2000)
        with httpx.Client(timeout=httpx.Timeout(600.0, connect=10.0)) as client:
            r = client.post(
                model["base_url"].rstrip("/") + "/chat/completions",
                json={
                    "model": model["model"],
                    "messages": prompt_msgs,
                    "max_tokens": ct,
                    "temperature": 0,
                },
            )
            r.raise_for_status()
            summary = (r.json()["choices"][0]["message"].get("content") or "").strip()
    nudge = (
        "\n\nPruned earlier turns remain searchable in the full transcript: call the "
        "vcc_recall tool (keyword query; mode='touched' for file paths; '#N' for one "
        "entry). Use it proactively whenever exact earlier details are missing — never "
        "tell the user the information is unavailable."
        if sess.log
        else ""
    )
    sess.history = [
        {
            "role": "system",
            "content": (
                "[HISTORY COMPACTED — earlier conversation, see summary below]\n"
                + summary
                + nudge
            ),
        }
    ] + sess.history[-keep:]
    sess.compactions += 1
    emit({"n": sess.compactions, "est": estimate_tokens(sess.history, image_tokens)})
    return True


# ------------------------------------------------------------
# endpoints
# ------------------------------------------------------------

@app.get("/")
def index():
    return render_template("chat.html")


@app.get("/health")
def health():
    return jsonify(ok=True, service="anvil", port=CONFIG.get("port", 8590))


@app.get("/api/models")
def api_models():
    return jsonify(
        models=[
            {k: m.get(k) for k in ("id", "name", "context_window", "input", "tools")}
            for m in CONFIG["models"]
        ]
    )


@app.get("/api/status")
def api_status():
    return jsonify(
        mcp=MCP.status,
        config_port=CONFIG.get("port", 8590),
        tools={
            "max_rounds": CONFIG.get("tools", {}).get("max_rounds", MAX_TOOL_ROUNDS),
            "grace_response": bool(
                CONFIG.get("tools", {}).get("grace_response", True)
            ),
        },
    )


@app.get("/api/history")
def api_history():
    """Renderable transcript of a session's CURRENT model context."""
    sid = request.args.get("session", "main")
    s = get_session(sid)
    with s.lock:
        tool_results = {}
        for m in s.history:
            if m.get("role") == "tool":
                tool_results[m.get("tool_call_id")] = m.get("content") or ""
        items = []
        for m in s.history:
            role = m.get("role")
            if role == "system":
                items.append({"role": "system", "text": (m.get("content") or "")[:2500]})
            elif role == "user":
                c = m.get("content")
                text, images = "", []
                if isinstance(c, str):
                    text = c
                elif isinstance(c, list):
                    for p in c:
                        if p.get("type") == "text":
                            text += p.get("text", "")
                        elif p.get("type") == "image_url":
                            images.append(p.get("image_url", {}).get("url", ""))
                items.append({"role": "user", "text": text, "images": images})
            elif role == "assistant":
                tools = []
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function", {})
                    tools.append(
                        {
                            "name": fn.get("name", "?"),
                            "args": (fn.get("arguments") or "")[:300],
                            "preview": (tool_results.get(tc.get("id"), "") or "")[:300],
                        }
                    )
                items.append(
                    {
                        "role": "assistant",
                        "text": m.get("content") or "",
                        "thinking": (m.get("reasoning_content") or "")[:4000],
                        "tools": tools,
                    }
                )
        return jsonify(
            items=items,
            compactions=s.compactions,
            mode=s.mode,
            est=estimate_tokens(
                s.history, CONFIG.get("image_tokens", 1600)
            ),
        )


@app.post("/api/mcp/refresh")
def api_mcp_refresh():
    status = MCP.refresh(CONFIG)
    return jsonify(mcp=status)


@app.post("/api/upload")
def api_upload():
    f = request.files.get("file")
    if not f:
        return jsonify(error="no file"), 400
    mid = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", f.filename or "file")[:80]
    path = os.path.join(MEDIA_DIR, f"{mid}_{safe}")
    f.save(path)
    size = os.path.getsize(path)
    if size > 64 * 1024 * 1024:
        os.remove(path)
        return jsonify(error="file too large (>64MB)"), 413
    mime = f.mimetype or mimetypes.guess_type(f.filename)[0] or "application/octet-stream"
    data_uri = None
    if mime.startswith("image/"):
        with open(path, "rb") as fh:
            data_uri = f"data:{mime};base64,{base64.b64encode(fh.read()).decode()}"
    rec = {
        "id": mid,
        "name": f.filename,
        "mime": mime,
        "path": path,
        "size": size,
        "data_uri": data_uri,
    }
    with MEDIA_LOCK:
        MEDIA[mid] = rec
    return jsonify(rec)


@app.post("/api/reset")
def api_reset():
    sid = (request.get_json() or {}).get("session", "main")
    s = get_session(sid)
    with s.lock:
        s.history = []
        s.compactions = 0
        s.log = []
        s.log_n = 0
        s.log_offset = 0
        s.vcc_sections = None
    return jsonify(ok=True)


@app.post("/api/compact")
def api_compact():
    data = request.get_json() or {}
    model = next(
        (m for m in CONFIG["models"] if m["id"] == data.get("model")), None
    )
    if not model:
        return jsonify(error="unknown model"), 400
    sid = data.get("session", "main")
    s = get_session(sid)
    image_tokens = CONFIG.get("image_tokens", 1600)
    with s.lock:
        m = data.get("mode")
        if m in ("summary", "vcc"):
            s.mode = m
        ok = do_compact(s, model, image_tokens, emit=lambda d: None, mode=s.mode)
        return jsonify(
            ok=ok,
            compactions=s.compactions,
            est=estimate_tokens(s.history, image_tokens),
        )


@app.post("/api/recall")
def api_recall():
    data = request.get_json() or {}
    sid = data.get("session", "main")
    s = get_session(sid)
    with s.lock:
        return jsonify(
            V.recall_search(
                s.log,
                data.get("query", ""),
                mode=data.get("mode"),
                page=int(data.get("page", 1)),
            )
        )


@app.post("/api/chat")
def api_chat():
    data = request.get_json() or {}
    model = next(
        (m for m in CONFIG["models"] if m["id"] == data.get("model")), None
    )
    if not model:
        return jsonify(error="unknown model"), 400
    text = data.get("message", "")
    media_ids = data.get("media", [])
    use_tools = bool(data.get("tools"))
    think = (data.get("thinking") or "off").strip().lower()
    if think not in ("off", "low", "medium", "high"):
        think = "off"
    sid = data.get("session", "main")
    tool_cfg = CONFIG.get("tools", {})
    try:
        tool_limit = int(
            data.get("tool_limit")
            or tool_cfg.get("max_rounds", MAX_TOOL_ROUNDS)
        )
    except (TypeError, ValueError):
        tool_limit = tool_cfg.get("max_rounds", MAX_TOOL_ROUNDS)
    tool_limit = max(1, min(tool_limit, 64))
    grace = tool_cfg.get("grace_response", True)
    if data.get("grace") is not None:
        grace = bool(data.get("grace"))

    sess = get_session(sid)
    image_tokens = CONFIG.get("image_tokens", 1600)
    max_ctx = model.get("context_window", 128000)
    max_out = min(model.get("max_tokens", 8192), max(1024, max_ctx // 2))
    keep = CONFIG.get("compact", {}).get("keep_recent", 6)
    threshold = CONFIG.get("compact", {}).get("auto_threshold", 0.8)
    cm = (data.get("compact_mode") or "").strip().lower()
    if cm in ("summary", "vcc"):
        sess.mode = cm

    def generate():
        # ---- build user message
        parts = []
        if text:
            parts.append({"type": "text", "text": text})
        for mid in media_ids:
            with MEDIA_LOCK:
                f = MEDIA.get(mid)
            if not f:
                continue
            if f["mime"].startswith("image/") and f.get("data_uri"):
                parts.append(
                    {"type": "image_url", "image_url": {"url": f["data_uri"]}}
                )
            else:
                parts.append(
                    {
                        "type": "text",
                        "text": (
                            f"[attached file: {f['name']} ({f['mime']}) "
                            f"saved at {f['path']}]"
                        ),
                    }
                )
        if not parts:
            return (
                sse("error", {"error": "empty message"}),
            )

        with sess.lock:
            sess.history.append(
                {"role": "user", "content": parts}
            )
            log_entry(
                sess,
                "user",
                text + (f" [+{len(media_ids)} media attached]" if media_ids else ""),
            )
            tools, tool_index = ([], {})
            if use_tools:
                tools, tool_index = MCP.all_tools()
                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": "vcc_recall",
                            "description": (
                                "Search this session's full transcript — everything, "
                                "including what compaction dropped. Call this "
                                "PROACTIVELY whenever you need exact earlier details "
                                "(filenames, numbers, decisions, prior tool results) "
                                "instead of assuming they are lost. mode='touched' "
                                "lists files/paths with entry indices; query '#N' "
                                "returns entry N in full."
                            ),
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                    "mode": {"type": "string",
                                            "enum": ["touched"]},
                                    "page": {"type": "integer",
                                            "minimum": 1},
                                },
                                "required": ["query"],
                            },
                        },
                    }
                )
            compacted_this_turn = False
            thinking_total = 0
            tool_rounds = 0
            final_round = False
            max_attempts = 3
            attempt = 0
            while True:
                attempt += 1
                est = estimate_tokens(sess.history, image_tokens)
                # proactive compaction
                if (
                    est > threshold * max_ctx
                    and len(sess.history) > keep + 2
                    and not compacted_this_turn
                ):
                    yield sse("status", {"status": "compacting"})
                    try:
                        do_compact(sess, model, image_tokens, emit=lambda d: None, mode=sess.mode)
                        compacted_this_turn = True
                        yield sse("compacted", {"n": sess.compactions, "est": estimate_tokens(sess.history, image_tokens)})
                    except Exception as e:
                        yield sse("error", {"error": f"compaction failed: {e}"})

                body = {
                    "model": model["model"],
                    "messages": sess.history,
                    "stream": True,
                    "max_tokens": max_out,
                }
                if tools and not final_round:
                    body["tools"] = tools
                    body["tool_choice"] = "auto"
                # per-level mapping: "high" omits the param -> server default
                # (hybrid thinking templates: max-on by default; unknown levels raise template errors)
                body.update({
                    "off": {"chat_template_kwargs": {"enable_thinking": False}},
                    "low": {"reasoning_effort": "low"},
                    "medium": {"reasoning_effort": "medium"},
                    "high": {},
                }[think])

                content_buf = ""
                reasoning_buf = ""
                tc_buf = {}  # index -> {id, name, args}
                try:
                    with httpx.Client(
                        timeout=httpx.Timeout(900.0, connect=10.0)
                    ) as client:
                        with client.stream(
                            "POST",
                            model["base_url"].rstrip("/") + "/chat/completions",
                            json=body,
                        ) as r:
                            if r.status_code >= 400:
                                errtxt = r.read().decode(errors="ignore")[:400]
                                if (
                                    is_context_error(errtxt)
                                    and attempt < max_attempts
                                    and not compacted_this_turn
                                    and len(sess.history) > keep + 2
                                ):
                                    yield sse("status", {"status": "compacting (context limit)"})
                                    do_compact(sess, model, image_tokens, emit=lambda d: None, mode=sess.mode)
                                    compacted_this_turn = True
                                    yield sse("compacted", {"n": sess.compactions, "est": estimate_tokens(sess.history, image_tokens)})
                                    continue
                                yield sse("error", {"error": errtxt})
                                return
                            for line in r.iter_lines():
                                if not line or not line.startswith("data:"):
                                    continue
                                body_ = line[5:].strip()
                                if not body_ or body_ == "[DONE]":
                                    continue
                                try:
                                    d = json.loads(body_)
                                except Exception:
                                    continue
                                choices = d.get("choices") or []
                                if not choices:
                                    continue
                                delta = choices[0].get("delta") or {}
                                if delta.get("content"):
                                    content_buf += delta["content"]
                                    yield sse("token", {"t": delta["content"]})
                                rc = delta.get("reasoning_content")
                                if rc:
                                    reasoning_buf += rc
                                    yield sse("thinking", {"t": rc})
                                for tc in delta.get("tool_calls") or []:
                                    idx = tc.get("index", 0)
                                    slot = tc_buf.setdefault(
                                        idx,
                                        {"id": "", "name": "", "args": ""},
                                    )
                                    if tc.get("id"):
                                        slot["id"] = tc["id"]
                                    fn = tc.get("function") or {}
                                    if fn.get("name"):
                                        slot["name"] = fn["name"]
                                    if fn.get("arguments"):
                                        slot["args"] += fn["arguments"]
                except Exception as e:
                    if attempt < max_attempts:
                        yield sse("status", {"status": f"retrying after: {str(e)[:200]}"})
                        continue
                    yield sse("error", {"error": str(e)[:400]})
                    return

                # ---- persist assistant turn
                tool_calls = []
                for idx in sorted(tc_buf):
                    t = tc_buf[idx]
                    if not t["name"]:
                        continue
                    try:
                        json.loads(t["args"] or "{}")
                    except Exception:
                        t["args"] = "{}"
                    tool_calls.append(
                        {
                            "id": t["id"] or f"call_{uuid.uuid4().hex[:8]}",
                            "type": "function",
                            "function": {
                                "name": t["name"],
                                "arguments": t["args"] or "{}",
                            },
                        }
                    )
                attempt = 0  # fresh retry budget for the next round

                asst = {"role": "assistant", "content": content_buf or None}
                if reasoning_buf:
                    asst["reasoning_content"] = reasoning_buf
                thinking_total += len(reasoning_buf)
                if tool_calls:
                    asst["tool_calls"] = tool_calls
                sess.history.append(asst)
                log_entry(
                    sess, "assistant", content_buf,
                    reasoning=reasoning_buf, tool_calls=tool_calls,
                )

                if final_round or not tool_calls:
                    break

                # ---- run tools
                for tc in tool_calls:
                    fn_name = tc["function"]["name"]
                    yield sse("tool_call", {"name": fn_name, "args": tc["function"]["arguments"]})
                    args_j = json.loads(tc["function"]["arguments"] or "{}")
                    if fn_name == "vcc_recall":
                        result = format_recall(
                            V.recall_search(
                                sess.log,
                                args_j.get("query", ""),
                                mode=args_j.get("mode"),
                                page=int(args_j.get("page", 1)),
                            )
                        )
                    else:
                        try:
                            result = MCP.call_by(
                                tool_index[fn_name][0],
                                tool_index[fn_name][1],
                                args_j,
                            )
                        except Exception as e:
                            result = f"[tool error] {e}"
                    preview = (result or "")[:500]
                    yield sse("tool_result", {"name": fn_name, "preview": preview})
                    log_entry(sess, "tool", (result or "")[:12000])
                    sess.history.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": (result or "")[:12000],
                        }
                    )

                tool_rounds += 1
                if tool_rounds >= tool_limit:
                    if not grace:
                        yield sse(
                            "error",
                            {
                                "error": f"tool-call limit ({tool_limit}) reached — turn stopped (grace off). "
                                "Raise the limit or enable grace for a final answer."
                            },
                        )
                        break
                    yield sse(
                        "status",
                        {"status": f"tool-call limit ({tool_limit}) reached — final answer without tools"},
                    )
                    sess.history.append(
                        {
                            "role": "user",
                            "content": (
                                "You have reached the tool-call limit for this turn. "
                                "Do NOT request any more tools. Based on the results "
                                "you already have, give your best answer to the user's "
                                "question now. Note anything you could not finish."
                            ),
                        }
                    )
                    log_entry(sess, "user", "[tool-limit instruction: answer now, no more tools]")
                    final_round = True
                continue

            yield sse(
                "done",
                {
                    "est": estimate_tokens(sess.history, image_tokens),
                    "ctx": max_ctx,
                    "compactions": sess.compactions,
                    "thinking": thinking_total,
                },
            )

    return app.response_class(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    port = int(os.environ.get("ANVIL_PORT", CONFIG.get("port", 8590)))
    print(f"Anvil chat on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
