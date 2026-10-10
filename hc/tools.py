"""Tools the model can call. Each tool has a JSON schema, a permission kind, and an implementation."""

import difflib
import fnmatch
import glob as globmod
import html
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request

from .llm import Cancelled
from .ui import diff_stats

IGNORE_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache", ".tox",
               ".idea", ".cache"}


class ToolError(Exception):
    pass


class ToolContext:
    """Mutable state shared by tools within one agent."""

    def __init__(self, cfg, cwd):
        self.cfg = cfg
        self.cwd = cwd
        self.read_files = {}  # abs path -> mtime when last read/written by the agent
        self.todos = []
        self.subagent_runner = None
        self.turn = 0
        self.backups = []  # (turn, path, original content or None if the file didn't exist)
        self.cancel = threading.Event()  # set when the user interrupts; the agent shares the client's event

    def resolve(self, path):
        if not path:
            raise ToolError("path is empty")
        path = os.path.expanduser(str(path))
        if not os.path.isabs(path):
            path = os.path.join(self.cwd, path)
        return os.path.normpath(path)

    def rel(self, path):
        r = os.path.relpath(path, self.cwd)
        return path if r.startswith("../..") else r

    def backup(self, path):
        if any(t == self.turn and p == path for t, p, _ in self.backups):
            return
        original = None
        if os.path.exists(path):
            with open(path, encoding="utf-8", errors="surrogateescape") as f:
                original = f.read()
        self.backups.append((self.turn, path, original))


def truncate(s, limit):
    if len(s) <= limit:
        return s
    head = limit * 2 // 3
    tail = limit - head
    return f"{s[:head]}\n\n... [{len(s) - limit} characters truncated] ...\n\n{s[-tail:]}"


def _read_text(path):
    with open(path, "rb") as f:
        data = f.read()
    if b"\0" in data[:8192]:
        raise ToolError(f"{path} appears to be a binary file")
    return data.decode("utf-8", errors="surrogateescape")


def _write_text(path, text):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8", errors="surrogateescape") as f:
        f.write(text)


def _missing_file_error(path):
    msg = f"File does not exist: {path}"
    parent = os.path.dirname(path)
    if os.path.isdir(parent):
        close = difflib.get_close_matches(os.path.basename(path), os.listdir(parent), n=3)
        if close:
            msg += f". Did you mean: {', '.join(os.path.join(parent, c) for c in close)}?"
    return ToolError(msg)


def _check_fresh(ctx, path):
    if path not in ctx.read_files:
        raise ToolError(f"You must Read {path} before modifying it.")
    if os.path.getmtime(path) != ctx.read_files[path]:
        raise ToolError(f"{path} was modified on disk since you last read it. Read it again before modifying it.")


# ---------------------------------------------------------------- Read

def t_read(ctx, file_path, offset=1, limit=1000):
    path = ctx.resolve(file_path)
    if not os.path.exists(path):
        raise _missing_file_error(path)
    if os.path.isdir(path):
        raise ToolError(f"{path} is a directory; use LS to list it")
    lines = _read_text(path).splitlines()
    ctx.read_files[path] = os.path.getmtime(path)
    if not lines:
        return "(file is empty)"
    start = max(int(offset or 1), 1)
    if start > len(lines):
        raise ToolError(f"offset {start} is past the end of the file ({len(lines)} lines)")
    end = min(start - 1 + int(limit or 1000), len(lines))
    budget = ctx.cfg.max_tool_output
    out = []
    for n in range(start, end + 1):
        line = lines[n - 1]
        if len(line) > 1000:
            line = line[:1000] + " …[line truncated]"
        entry = f"{n:6}\t{line}"
        budget -= len(entry) + 1
        if budget < 0 and out:
            end = n - 1
            break
        out.append(entry)
    if end < len(lines):
        out.append(f"... ({len(lines) - end} more lines; call Read with offset={end + 1} to continue)")
    return "\n".join(out)


# ---------------------------------------------------------------- Write

def p_write(ctx, file_path, content):
    path = ctx.resolve(file_path)
    if os.path.isdir(path):
        raise ToolError(f"{path} is a directory")
    old = ""
    if os.path.exists(path):
        _check_fresh(ctx, path)
        old = _read_text(path)
    return path, old, content


def t_write(ctx, file_path, content):
    path, old, new = p_write(ctx, file_path, content)
    existed = os.path.exists(path)
    ctx.backup(path)
    _write_text(path, new)
    ctx.read_files[path] = os.path.getmtime(path)
    return f"{'Updated' if existed else 'Created'} {path} ({len(new.splitlines())} lines)"


# ---------------------------------------------------------------- Edit

_LINE_PREFIX = re.compile(r"^ *\d+\t", re.M)


def _fuzzy_find(text, old):
    """Unique match ignoring trailing whitespace on each line. Returns (start, end) or None."""
    tlines = text.split("\n")
    olines = old.split("\n")
    trailing_nl = len(olines) > 1 and olines[-1] == ""
    if trailing_nl:
        olines = olines[:-1]
    target = [l.rstrip() for l in olines]
    stripped = [l.rstrip() for l in tlines]
    n = len(target)
    hits = [i for i in range(len(tlines) - n + 1) if stripped[i:i + n] == target]
    if len(hits) != 1:
        return None
    offsets = [0]
    for l in tlines:
        offsets.append(offsets[-1] + len(l) + 1)
    i = hits[0]
    start, end = offsets[i], offsets[i + n] - 1
    if trailing_nl:
        end += 1
    return start, min(end, len(text))


def p_edit(ctx, file_path, old_string, new_string, replace_all=False):
    path = ctx.resolve(file_path)
    if not os.path.exists(path):
        if old_string == "":
            return path, "", new_string
        raise _missing_file_error(path)
    _check_fresh(ctx, path)
    text = _read_text(path)
    if old_string == new_string:
        raise ToolError("old_string and new_string are identical; nothing to change")
    if old_string == "":
        raise ToolError("old_string is empty. Provide the exact text to replace (use Write to replace a whole file).")
    count = text.count(old_string)
    if count == 0 and _LINE_PREFIX.search(old_string) and all(
            _LINE_PREFIX.match(l) for l in old_string.split("\n") if l):
        # Model copied the "   12\t" prefixes from Read output.
        old_string = _LINE_PREFIX.sub("", old_string)
        if all(_LINE_PREFIX.match(l) for l in new_string.split("\n") if l):
            new_string = _LINE_PREFIX.sub("", new_string)
        count = text.count(old_string)
    if count == 0:
        span = _fuzzy_find(text, old_string)
        if span is None:
            raise ToolError("old_string was not found in the file. It must match exactly, including indentation and "
                            "whitespace, and must not include the line-number prefix from Read. Re-Read the file "
                            "and copy the text exactly.")
        return path, text, text[:span[0]] + new_string + text[span[1]:]
    if count > 1 and not replace_all:
        raise ToolError(f"old_string occurs {count} times in the file. Include more surrounding lines to make it "
                        f"unique, or set replace_all=true to replace every occurrence.")
    return path, text, text.replace(old_string, new_string) if replace_all else text.replace(old_string, new_string, 1)


def t_edit(ctx, **args):
    path, old, new = p_edit(ctx, **args)
    ctx.backup(path)
    _write_text(path, new)
    ctx.read_files[path] = os.path.getmtime(path)
    add, rem = diff_stats(old, new)
    return f"Edited {path} (+{add} -{rem} lines)"


# ---------------------------------------------------------------- Bash

_CWD_MARK = "__HC_CWD__"


def t_bash(ctx, command, timeout=None, description=None):
    timeout = max(1, min(int(timeout or ctx.cfg.bash_timeout), 600))
    script = f'{command}\n__hc_rc=$?\nprintf "\\n{_CWD_MARK}%s" "$(pwd)"\nexit $__hc_rc\n'
    proc = subprocess.Popen(
        ["bash", "-c", script], cwd=ctx.cwd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, errors="replace", start_new_session=True,
        env={**os.environ, "PAGER": "cat", "GIT_PAGER": "cat", "TERM": "dumb"})

    def kill():
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    timed_out = False
    deadline = time.time() + timeout
    try:
        while True:
            try:
                output, _ = proc.communicate(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                if ctx.cancel.is_set():
                    kill()
                    proc.communicate()
                    raise Cancelled()
                if time.time() > deadline:
                    kill()
                    output, _ = proc.communicate()
                    timed_out = True
                    break
    except KeyboardInterrupt:
        kill()
        proc.communicate()
        raise

    notes = []
    idx = output.rfind(_CWD_MARK)
    if idx >= 0:
        new_cwd = output[idx + len(_CWD_MARK):].strip()
        output = output[:idx]
        if output.endswith("\n"):
            output = output[:-1]
        if new_cwd and os.path.isdir(new_cwd) and new_cwd != ctx.cwd:
            ctx.cwd = new_cwd
            notes.append(f"[working directory is now {new_cwd}]")
    if timed_out:
        notes.append(f"[command timed out after {timeout}s and was killed]")
    elif proc.returncode:
        notes.append(f"[exit code {proc.returncode}]")
    output = truncate(output.rstrip(), ctx.cfg.max_tool_output)
    return "\n".join(x for x in [output or "(no output)", *notes] if x)


# ---------------------------------------------------------------- Glob / Grep / LS

def _ignored(rel_path):
    return any(part in IGNORE_DIRS for part in rel_path.split(os.sep))


def t_glob(ctx, pattern, path=None):
    base = ctx.resolve(path or ".")
    if not os.path.isdir(base):
        raise ToolError(f"{base} is not a directory")
    hits = []
    for rel in globmod.iglob(pattern, root_dir=base, recursive=True):
        full = os.path.join(base, rel)
        if os.path.isfile(full) and not _ignored(rel):
            hits.append(full)
    if not hits:
        return "No files found."
    hits.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    shown = [ctx.rel(p) for p in hits[:200]]
    if len(hits) > 200:
        shown.append(f"... ({len(hits) - 200} more; narrow the pattern)")
    return "\n".join(shown)


def _python_grep(ctx, regex, base, file_glob, mode, context):
    out = []
    files = [base] if os.path.isfile(base) else None
    if files is None:
        files = []
        for root, dirs, names in os.walk(base):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            for n in names:
                full = os.path.join(root, n)
                if file_glob and not (fnmatch.fnmatch(n, file_glob) or
                                      fnmatch.fnmatch(os.path.relpath(full, base), file_glob)):
                    continue
                files.append(full)
    for full in files:
        try:
            if os.path.getsize(full) > 5_000_000:
                continue
            lines = _read_text(full).splitlines()
        except (ToolError, OSError):
            continue
        matched = [i for i, l in enumerate(lines) if regex.search(l)]
        if not matched:
            continue
        rel = ctx.rel(full)
        if mode == "files_with_matches":
            out.append(rel)
        elif mode == "count":
            out.append(f"{rel}:{len(matched)}")
        else:
            show = sorted({j for i in matched for j in range(max(0, i - context), min(len(lines), i + context + 1))})
            mset = set(matched)
            prev = None
            for j in show:
                if prev is not None and j != prev + 1:
                    out.append("--")
                out.append(f"{rel}{':' if j in mset else '-'}{j + 1}{':' if j in mset else '-'}{lines[j][:300]}")
                prev = j
    return out


def t_grep(ctx, pattern, path=None, glob=None, output_mode="files_with_matches", case_insensitive=False,
           context=0, head_limit=200):
    base = ctx.resolve(path or ".")
    if not os.path.exists(base):
        raise _missing_file_error(base)
    if output_mode not in ("files_with_matches", "content", "count"):
        raise ToolError("output_mode must be one of: files_with_matches, content, count")
    context = max(0, int(context or 0))
    if shutil.which("rg"):
        cmd = ["rg", "--no-heading", "--color=never", "--max-columns=300", "--max-columns-preview"]
        if case_insensitive:
            cmd.append("-i")
        if glob:
            cmd += ["--glob", glob]
        if output_mode == "files_with_matches":
            cmd.append("-l")
        elif output_mode == "count":
            cmd.append("-c")
        else:
            cmd.append("-n")
            if context:
                cmd += ["-C", str(context)]
        cmd += ["-e", pattern, "--", os.path.relpath(base, ctx.cwd)]
        r = subprocess.run(cmd, cwd=ctx.cwd, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                           errors="replace", timeout=60)
        if r.returncode == 2 and not r.stdout:
            raise ToolError(r.stderr.strip() or "rg failed")
        lines = [l[2:] if l.startswith("./") else l for l in r.stdout.splitlines()]
    else:
        try:
            regex = re.compile(pattern, re.I if case_insensitive else 0)
        except re.error as e:
            raise ToolError(f"invalid regex: {e}") from e
        lines = _python_grep(ctx, regex, base, glob, output_mode, context)
    if not lines:
        return "No matches found."
    limit = int(head_limit or 200)
    if len(lines) > limit:
        lines = lines[:limit] + [f"... ({len(lines) - limit} more results; narrow the search)"]
    return "\n".join(lines)


def t_ls(ctx, path=None):
    base = ctx.resolve(path or ".")
    if not os.path.isdir(base):
        raise ToolError(f"{base} is not a directory" if os.path.exists(base) else f"{base} does not exist")
    entries = sorted(os.scandir(base), key=lambda e: (not e.is_dir(), e.name.lower()))
    lines = [f"{base}/"]
    for e in entries[:500]:
        if e.name in IGNORE_DIRS:
            lines.append(f"  {e.name}/ (ignored)")
        elif e.is_dir():
            lines.append(f"  {e.name}/")
        else:
            try:
                size = e.stat().st_size
            except OSError:
                size = 0
            lines.append(f"  {e.name}  ({size:,} bytes)")
    if len(entries) > 500:
        lines.append(f"  ... ({len(entries) - 500} more)")
    return "\n".join(lines)


# ---------------------------------------------------------------- WebFetch

def t_webfetch(ctx, url):
    if not url.startswith(("http://", "https://")):
        raise ToolError("url must start with http:// or https://")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (harness coding agent)"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read(3_000_000)
            ctype = r.headers.get("Content-Type", "")
            charset = r.headers.get_content_charset() or "utf-8"
    except (urllib.error.URLError, OSError) as e:
        raise ToolError(f"fetch failed: {e}") from e
    text = raw.decode(charset, errors="replace")
    if "html" in ctype or text.lstrip()[:15].lower().startswith(("<!doctype", "<html")):
        text = re.sub(r"(?is)<(script|style|noscript|svg|head)\b.*?</\1>", " ", text)
        text = re.sub(r"(?i)<br\s*/?>|</(p|div|li|h[1-6]|tr|pre)>", "\n", text)
        text = html.unescape(re.sub(r"<[^>]+>", " ", text))
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    return truncate(text, ctx.cfg.max_tool_output)


# ---------------------------------------------------------------- TodoWrite / Task

def t_todo(ctx, todos):
    if not isinstance(todos, list):
        raise ToolError("todos must be an array")
    clean = []
    for t in todos:
        if not isinstance(t, dict) or not t.get("content"):
            raise ToolError("each todo needs 'content' and 'status'")
        status = t.get("status", "pending")
        if status not in ("pending", "in_progress", "completed"):
            raise ToolError(f"invalid status {status!r}")
        clean.append({"content": str(t["content"]), "status": status})
    ctx.todos = clean
    return format_todos(clean)


def format_todos(todos):
    if not todos:
        return "(no todos)"
    mark = {"pending": "☐", "in_progress": "◐", "completed": "☒"}
    return "\n".join(f"{mark[t['status']]} {t['content']}" for t in todos)


def t_task(ctx, description, prompt):
    if not ctx.subagent_runner:
        raise ToolError("sub-agents are not available here")
    return ctx.subagent_runner(description, prompt)


# ---------------------------------------------------------------- registry

class Tool:
    def __init__(self, name, description, params, required, run, kind="read", summary=None, prepare=None):
        self.name = name
        self.description = description
        self.params = params
        self.required = required
        self.run_fn = run
        self.kind = kind  # read | edit | exec | web | meta
        self.summary_key = summary
        self.prepare_fn = prepare

    def schema(self):
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": {"type": "object", "properties": self.params, "required": self.required}}}

    def summary(self, args):
        v = args.get(self.summary_key, "") if self.summary_key else ""
        v = str(v).replace("\n", " ⏎ ")
        return v if len(v) <= 100 else v[:99] + "…"

    def coerce(self, args):
        """Validate and coerce model-supplied arguments (small models often send ints as strings, etc.)."""
        if not isinstance(args, dict):
            raise ToolError("arguments must be a JSON object")
        for k in self.required:
            if k not in args:
                raise ToolError(f"missing required parameter '{k}'")
        clean = {}
        for k, v in args.items():
            spec = self.params.get(k)
            if spec is None:
                continue
            t = spec.get("type")
            try:
                if t == "integer" and not isinstance(v, int):
                    v = int(str(v).strip())
                elif t == "boolean" and not isinstance(v, bool):
                    v = str(v).strip().lower() in ("true", "1", "yes")
                elif t == "array" and isinstance(v, str):
                    v = json.loads(v)
                elif t == "string" and not isinstance(v, str):
                    v = v if v is None else (json.dumps(v) if isinstance(v, (dict, list)) else str(v))
            except (ValueError, json.JSONDecodeError) as e:
                raise ToolError(f"parameter '{k}' should be of type {t}") from e
            if v is not None:
                clean[k] = v
        return clean

    def prepare(self, ctx, args):
        """Dry run for previews: returns (path, old, new) for file changes, else None. May raise ToolError."""
        return self.prepare_fn(ctx, **args) if self.prepare_fn else None

    def run(self, ctx, args):
        return self.run_fn(ctx, **args)


S = lambda d: {"type": "string", "description": d}  # noqa: E731
I = lambda d: {"type": "integer", "description": d}  # noqa: E731
B = lambda d: {"type": "boolean", "description": d}  # noqa: E731


def build_tools(include_task=True):
    tools = [
        Tool("Read", "Read a text file. Returns lines prefixed with line numbers (`   12\\t...`). Use offset/limit "
             "for large files. You must Read a file before you Edit or overwrite it.",
             {"file_path": S("Path to the file (absolute or relative to the working directory)"),
              "offset": I("1-based line number to start reading from (optional)"),
              "limit": I("Maximum number of lines to read (optional, default 1000)")},
             ["file_path"], t_read, "read", "file_path"),
        Tool("Write", "Create a new file or completely overwrite an existing file with `content`. Prefer Edit for "
             "changing existing files. Parent directories are created automatically.",
             {"file_path": S("Path to the file"), "content": S("The full content of the file")},
             ["file_path", "content"], t_write, "edit", "file_path", p_write),
        Tool("Edit", "Replace text in a file. `old_string` must match the file contents exactly (including "
             "indentation) and be unique unless replace_all is true. Do not include Read's line-number prefixes. "
             "Include a few surrounding lines to make the match unique.",
             {"file_path": S("Path to the file"),
              "old_string": S("Exact text to replace"),
              "new_string": S("Replacement text"),
              "replace_all": B("Replace every occurrence (default false)")},
             ["file_path", "old_string", "new_string"], t_edit, "edit", "file_path", p_edit),
        Tool("Bash", "Run a shell command with bash and return its combined stdout/stderr. The working directory "
             "persists between calls. Commands must be non-interactive. Prefer Read/Grep/Glob/LS over cat/grep/find.",
             {"command": S("The command to run"),
              "timeout": I("Timeout in seconds (default 120, max 600)"),
              "description": S("Short description of what the command does")},
             ["command"], t_bash, "exec", "command"),
        Tool("Glob", "Find files by glob pattern, e.g. `**/*.py` or `src/**/test_*.js`. Results are sorted by "
             "modification time, newest first.",
             {"pattern": S("Glob pattern"), "path": S("Directory to search in (default: working directory)")},
             ["pattern"], t_glob, "read", "pattern"),
        Tool("Grep", "Search file CONTENTS with a regular expression (ripgrep syntax). To find files by name, use "
             "Glob instead.",
             {"pattern": S("Regular expression to search for inside files, e.g. `def login` or `TODO|FIXME`. "
                           "This is a regex, not a file glob: use the `glob` parameter to filter files"),
              "path": S("File or directory to search (default: working directory)"),
              "glob": S("Only search files matching this glob, e.g. `*.py`"),
              "output_mode": {"type": "string", "enum": ["files_with_matches", "content", "count"],
                              "description": "files_with_matches (default), content (matching lines), or count"},
              "case_insensitive": B("Case-insensitive search"),
              "context": I("Lines of context around each match (content mode only)"),
              "head_limit": I("Maximum number of result lines (default 200)")},
             ["pattern"], t_grep, "read", "pattern"),
        Tool("LS", "List the entries of a directory.",
             {"path": S("Directory path (default: working directory)")}, [], t_ls, "read", "path"),
        Tool("WebFetch", "Fetch a URL and return its content as plain text.",
             {"url": S("The http(s) URL to fetch")}, ["url"], t_webfetch, "web", "url"),
        Tool("TodoWrite", "Create or replace your task list. Use it for multi-step tasks: write all steps, keep "
             "exactly one in_progress, and mark items completed as soon as they are done. Send the full list "
             "every time.",
             {"todos": {"type": "array", "description": "The complete todo list",
                        "items": {"type": "object", "properties": {
                            "content": {"type": "string"},
                            "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}},
                            "required": ["content", "status"]}}},
             ["todos"], t_todo, "meta"),
    ]
    if include_task:
        tools.append(Tool(
            "Task", "Launch a read-only sub-agent to research a question in the codebase (searching and reading "
            "many files) and report back a summary. Use it for broad exploration to keep your own context small.",
            {"description": S("A short (3-5 word) description of the task"),
             "prompt": S("Detailed instructions for the sub-agent, including what to report back")},
            ["description", "prompt"], t_task, "read", "description"))
    return {t.name: t for t in tools}


READ_ONLY = ("Read", "Glob", "Grep", "LS", "TodoWrite", "Task", "WebFetch")
