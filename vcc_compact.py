"""
vcc_compact.py — algorithmic conversation compaction + recall for anvil.

No LLM calls. Deterministic. Inspired by pi-vcc (View-oriented Conversation
Compiler).

An "entry" is a dict:
    {"idx": int, "role": "user"|"assistant"|"tool",
     "text": str, "reasoning": str, "tool_calls": [{"name": str, "args": str}]}

compact_vcc(entries, prev_sections, keep=6) -> (summary_text, new_sections)
recall_search(entries, query, ...) -> dict with hits / touched / expanded
"""
import re

SECTION_CAPS = {
    "goal": 8,
    "files": 14,
    "outstanding": 8,
    "prefs": 6,
}
TRANSCRIPT_WINDOW = 120
TEXT_LIMIT = 180  # per-line truncation in the collapsed transcript


# ---------------------------------------------------------------
# text helpers
# ---------------------------------------------------------------

def _first_line(text: str, limit: int = 140) -> str:
    t = " ".join((text or "").split())
    return t[:limit] + ("…" if len(t) > limit else "")


def _clip(text: str, limit: int = TEXT_LIMIT) -> str:
    t = " ".join((text or "").split())
    return t[:limit] + ("…" if len(t) > limit else "")


_PATH_RE = re.compile(r"""["']?((?:/|\./)?[\w.-]+(?:/[\w.-]+)*/[\w-]+\.\w{1,8})["']?""")
_FILE_RE = re.compile(r"""["']?([\w-]+\.\w{1,8})["']?""")


def extract_paths(entry: dict) -> list:
    paths = []
    for tc in entry.get("tool_calls") or []:
        args = tc.get("args") or ""
        for m in _PATH_RE.finditer(args):
            p = m.group(1)
            if p not in paths:
                paths.append(p)
        if not _PATH_RE.search(args):
            for m in _FILE_RE.finditer(args):
                f = m.group(1)
                if len(f) > 4 and f not in paths:
                    paths.append(f)
    return paths


# ---------------------------------------------------------------
# section extraction
# ---------------------------------------------------------------

_SCOPE_RE = re.compile(
    r"(now also|scope|changed to|additionally|instead,|don't do|no longer)",
    re.I,
)
_OUTSTANDING_RE = re.compile(
    r"(error|failed|failing|still (waiting|pending|broken)|TODO|unresolved|blocked|can't|cannot )",
    re.I,
)
_PREF_RE = re.compile(r"(always|never|prefer|don't |do not |stop )", re.I)


def _split_lines(text: str) -> list:
    return [l.strip() for l in (text or "").splitlines() if l.strip()]


def _merge_cap(key: str, prev: list, new: list) -> list:
    out = list(prev)
    for line in new:
        if line not in out:
            out.append(line)
    return out[: SECTION_CAPS[key]]


def build_sections(entries: list, prev: dict = None) -> dict:
    prev = prev or {"goal": [], "files": [], "outstanding": [], "prefs": []}
    goal, files, outstanding, prefs = [], [], [], []

    first_user = True
    for e in entries:
        text = e.get("text") or ""
        if e["role"] == "user" and text:
            if first_user:
                goal.append("G: " + _first_line(text, 160))
                first_user = False
            for line in _split_lines(text):
                if _SCOPE_RE.search(line):
                    goal.append("Δ: " + _first_line(line))
                if _PREF_RE.search(line):
                    prefs.append(_first_line(line, 120))
        elif e["role"] in ("assistant", "tool"):
            for line in _split_lines(text):
                if _OUTSTANDING_RE.search(line) and len(line) > 8:
                    outstanding.append(_first_line(line, 140))
        for p in extract_paths(e):
            verb = "W" if any(tc.get("name", "").lower().startswith(
                ("write", "create", "new")) for tc in e.get("tool_calls") or []) else "M"
            line = f"{verb} {p}"
            if line not in files:
                files.append(line)

    return {
        "goal": _merge_cap("goal", prev.get("goal", []), goal),
        "files": _merge_cap("files", prev.get("files", []), files),
        "outstanding": _merge_cap("outstanding", prev.get("outstanding", []), outstanding),
        "prefs": _merge_cap("prefs", prev.get("prefs", []), prefs),
    }


def render_sections(sections: dict) -> list:
    out = []
    labels = {
        "goal": "[Session Goal]",
        "files": "[Files And Changes]",
        "outstanding": "[Outstanding Context]",
        "prefs": "[User Preferences]",
    }
    for key in ("goal", "files", "outstanding", "prefs"):
        lines = sections.get(key) or []
        if lines:
            out.append(labels[key])
            out.extend("- " + l for l in lines)
    return out


# ---------------------------------------------------------------
# transcript collapse
# ---------------------------------------------------------------

def collapse_entry(e: dict) -> list:
    lines = []
    label = {"user": "user", "assistant": "assistant", "tool": "tool"}[e["role"]]
    if e.get("text"):
        lines.append(f"[{label}] {_clip(e['text'])}")
    for tc in e.get("tool_calls") or []:
        arg = " ".join((tc.get("args") or "").split())[:80]
        lines.append(f"* {tc.get('name', '?')} \"{arg}\" (#{e['idx']})")
    return lines


def build_transcript(entries: list) -> list:
    lines = []
    for e in entries:
        lines.extend(collapse_entry(e))
    omitted = max(0, len(lines) - TRANSCRIPT_WINDOW)
    if omitted:
        lines = [f"...({omitted} earlier lines omitted)"] + lines[-TRANSCRIPT_WINDOW:]
    return lines


def compact_vcc(entries: list, prev_sections: dict = None,
               keep: int = 6) -> tuple:
    """Summarize `entries` algorithmically. Returns (summary_text, new_sections).
    If `entries` is a full history, pass keep = tail to exclude; if the tail
    is already excluded, pass keep=0."""
    if len(entries) - keep < 2:
        return "", (prev_sections or {})
    sections = build_sections(entries, prev_sections)
    parts = [
        "[HISTORY COMPACTED — algorithmic (vcc) summary of earlier conversation; "
        "use recall to search the full transcript]"
    ]
    parts.extend(render_sections(sections))
    parts.extend(build_transcript(entries))
    return "\n".join(parts), sections


# ---------------------------------------------------------------
# recall
# ---------------------------------------------------------------

def _searchable(e: dict) -> str:
    bits = [e.get("text") or "", e.get("reasoning") or ""]
    for tc in e.get("tool_calls") or []:
        bits.append(tc.get("name", ""))
        bits.append(tc.get("args") or "")
    return " ".join(bits)


def recall_search(entries: list, query: str, mode: str = None,
                 page: int = 1, page_size: int = 10) -> dict:
    """Search the transcript log. Keyword/regex ranked; mode=touched for paths."""
    query = (query or "").strip()

    # expand: return full content of given indices
    idxs = [e["idx"] for e in entries]
    if isinstance(query, list) or query.startswith("#"):
        want = [int(x) for x in re.findall(r"#?(\d+)", str(query)) if int(x) in idxs]
        return {"hits": [], "touched": [],
                "expanded": [e for e in entries if e["idx"] in want],
                "total": len(want)}

    if mode == "touched":
        by_path = {}
        for e in entries:
            for p in extract_paths(e):
                by_path.setdefault(p, []).append(e["idx"])
        touched = [{"path": p, "entries": i} for p, i in
                   sorted(by_path.items())]
        return {"hits": [], "touched": touched, "expanded": [],
                "total": len(touched)}

    if not query:
        return {"hits": [], "touched": [], "expanded": [], "total": 0}

    # regex if the query has regex metacharacters
    if re.search(r"[.*|(){}\[\]^$\\]", query):
        try:
            rx = re.compile(query, re.I)
        except re.error:
            rx = None
        if rx:
            hits = []
            for e in entries:
                m = rx.search(_searchable(e))
                if m:
                    hits.append(_hit(e, m.start(), m.end(), 1.0))
            hits.sort(key=lambda h: (-h["score"], h["idx"]))
            total = len(hits)
            lo = (page - 1) * page_size
            return {"hits": hits[lo:lo + page_size], "touched": [],
                    "expanded": [], "total": total}

    # keyword search, IDF-weighted
    terms = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 1]
    if not terms:
        terms = [query.lower()]
    docfreq = {}
    docs = [(e, _searchable(e).lower()) for e in entries]
    for _e, s in docs:
        for t in set(terms):
            if t in s:
                docfreq[t] = docfreq.get(t, 0) + 1
    idf = {t: 1.0 / (1 + docfreq.get(t, 0)) for t in terms}

    hits = []
    for e, s in docs:
        score = 0.0
        for t in terms:
            n = s.count(t)
            if n:
                score += idf[t] * min(n, 5)
        if score:
            pos = s.find(min(terms, key=lambda t: s.count(t)))
            hits.append(_hit(e, pos, pos + 12, score))
    hits.sort(key=lambda h: (-h["score"], h["idx"]))
    total = len(hits)
    lo = (page - 1) * page_size
    return {"hits": hits[lo:lo + page_size], "touched": [],
            "expanded": [], "total": total}


def _hit(e: dict, a: int, b: int, score: float) -> dict:
    s = _searchable(e)
    start = max(0, a - 60)
    snippet = ("…" if start > 0 else "") + " ".join(
        s[start:b].split())[:180]
    return {"idx": e["idx"], "role": e["role"], "score": round(score, 3),
            "snippet": snippet}
