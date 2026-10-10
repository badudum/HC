"""Full-screen terminal app (opencode / Claude Code style) on top of the agent.

The agent runs on a worker thread and reports through TuiView, which turns events into transcript blocks. The
main thread owns curses: it draws the screen ~12 times a second and handles keys. Shared state is guarded by
App.lock.
"""

import curses
import difflib
import json
import locale
import os
import re
import sys
import threading
import time
import traceback
import unicodedata

from . import __version__, prompts
from .config import HOME, mask_key, save_credentials
from .llm import Cancelled, LLMError
from .permissions import MODE_HELP, MODES
from .session import Session, list_sessions
from .tools import format_todos, t_bash, t_ls, t_read, ToolError
from .view import View

SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
HISTORY_FILE = os.path.join(HOME, "tui_history.json")
LOG_FILE = os.path.join(HOME, "tui.log")

COMMANDS = {
    "/help": "keys and commands",
    "/clear": "start a new conversation",
    "/compact": "summarize the conversation now (optional focus)",
    "/context": "context window usage",
    "/summary": "show the latest summary",
    "/mode": "permission mode: " + ", ".join(MODES),
    "/model": "show or set the model",
    "/allow": "auto-approve bash commands starting with a prefix",
    "/undo": "revert the file changes from the last turn",
    "/todos": "show the todo list",
    "/init": "write a HARNESS.md describing this project",
    "/resume": "list sessions, or /resume N to open one",
    "/think": "how long the model may think per step: low, medium, high, max",
    "/login": "set the API key",
    "/stats": "token usage",
    "/config": "active settings",
    "/exit": "quit",
}

KEYS_HELP = """\
enter send · ctrl-j / alt-enter newline · tab complete · ↑↓ history
esc interrupt the model · ctrl-c twice quit · shift-tab cycle permission mode
ctrl-t show/hide thinking · ctrl-o expand/collapse tool output · ctrl-b sidebar
pgup/pgdn or mouse wheel scroll · end jump to bottom · ctrl-l redraw
@path attach a file · !cmd run a shell command · shift+drag to select text"""

# ------------------------------------------------------------------ styles & text measurement

S = {}


def init_styles(mouse):
    curses.start_color()
    try:
        curses.use_default_colors()
        bg = -1
    except curses.error:
        bg = curses.COLOR_BLACK
    colors = {"accent": curses.COLOR_MAGENTA, "green": curses.COLOR_GREEN, "red": curses.COLOR_RED,
              "yellow": curses.COLOR_YELLOW, "cyan": curses.COLOR_CYAN, "blue": curses.COLOR_BLUE}
    for i, (name, fg) in enumerate(colors.items(), 1):
        curses.init_pair(i, fg, bg)
        S[name] = curses.color_pair(i)
    if curses.COLORS >= 16:
        curses.init_pair(10, 8, bg)
        S["gray"] = curses.color_pair(10)
    else:
        S["gray"] = curses.A_DIM
    curses.init_pair(11, curses.COLOR_BLACK, curses.COLOR_MAGENTA)
    S["badge"] = curses.color_pair(11) | curses.A_BOLD
    curses.init_pair(12, curses.COLOR_BLACK, curses.COLOR_YELLOW)
    S["badge_warn"] = curses.color_pair(12) | curses.A_BOLD
    S["normal"] = curses.A_NORMAL
    S["bold"] = curses.A_BOLD
    S["italic"] = getattr(curses, "A_ITALIC", 0)
    S["rev"] = curses.A_REVERSE
    if mouse:
        wheel_down = getattr(curses, "BUTTON5_PRESSED", 0x200000)
        curses.mousemask(curses.BUTTON4_PRESSED | wheel_down)
        curses.mouseinterval(0)


def cw(ch):
    o = ord(ch)
    if o < 32 or 0x7F <= o < 0xA0 or unicodedata.combining(ch):
        return 0
    return 2 if unicodedata.east_asian_width(ch) in "WF" else 1


def sw(s):
    return sum(cw(c) for c in s)


_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07|\x1b.")
_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def clean(text):
    """Strip ANSI escapes and control characters (e.g. from command output) before drawing."""
    return _CTRL.sub("", _ANSI.sub("", text.replace("\t", "    ").replace("\r\n", "\n")))


def clip(text, width):
    out, w = [], 0
    for ch in text:
        c = cw(ch)
        if w + c > width:
            break
        out.append(ch)
        w += c
    return "".join(out)


def ellipsize(text, width):
    return text if sw(text) <= width else clip(text, max(0, width - 1)) + "…"


_TOK = re.compile(r"\s+|\S+")


def wrap(segs, width, first=(), rest=None, hard=False):
    """Word-wraps styled segments [(text, attr)] to `width` columns. Returns a list of lines (segment lists)."""
    rest = first if rest is None else rest
    pw_first, pw_rest = sum(sw(t) for t, _ in first), sum(sw(t) for t, _ in rest)
    width = max(width, pw_rest + 4, pw_first + 4)
    lines, cur, curw = [], list(first), pw_first

    def newline():
        nonlocal cur, curw
        lines.append(cur)
        cur, curw = list(rest), pw_rest

    for text, attr in segs:
        for tok in (list(text) if hard else _TOK.findall(text)):
            tw = sw(tok)
            if curw + tw <= width:
                cur.append((tok, attr))
                curw += tw
            elif tok.isspace():
                newline()
            elif not hard and tw <= width - pw_rest:
                newline()
                cur.append((tok, attr))
                curw += tw
            else:
                for ch in tok:
                    if curw + cw(ch) > width:
                        newline()
                    cur.append((ch, attr))
                    curw += cw(ch)
    lines.append(cur)
    return lines


def inline(text, base=0):
    """`code` and **bold** spans -> segments."""
    segs, buf, code, bold = [], [], False, False
    i = 0

    def flush():
        if buf:
            attr = S["cyan"] if code else base
            segs.append(("".join(buf), attr | (curses.A_BOLD if bold else 0)))
            buf.clear()

    while i < len(text):
        if text[i] == "`":
            flush()
            code = not code
            i += 1
        elif text.startswith("**", i) and not code:
            flush()
            bold = not bold
            i += 2
        else:
            buf.append(text[i])
            i += 1
    flush()
    return segs


def markdown(text, width, margin="  "):
    out, in_code = [], False
    m0 = [(margin, 0)]
    for raw in clean(text).split("\n"):
        s = raw.strip()
        if s.startswith("```"):
            in_code = not in_code
            if in_code and s[3:].strip():
                out.append(m0 + [("╭ " + s[3:].strip(), S["gray"])])
            continue
        if in_code:
            out += wrap([(raw, S["cyan"])], width, m0 + [("│ ", S["gray"])], hard=True)
        elif (m := re.match(r"(#{1,6})\s+(.*)", s)):
            out += wrap(inline(m.group(2), S["accent"] | curses.A_BOLD), width, m0)
        elif (m := re.match(r"(\s*)([-*+]|\d+[.)])\s+(.*)", raw)):
            bullet = "• " if m.group(2) in "-*+" else m.group(2) + " "
            ind = m.group(1)
            out += wrap(inline(m.group(3)), width, m0 + [(ind + bullet, S["accent"])],
                        m0 + [(" " * (len(ind) + sw(bullet)), 0)])
        elif s.startswith(">"):
            out += wrap(inline(s[1:].strip(), S["gray"]), width, m0 + [("▎ ", S["gray"])])
        elif re.fullmatch(r"[-*_]{3,}", s):
            out.append(m0 + [("─" * min(40, width - len(margin)), S["gray"])])
        else:
            out += wrap(inline(raw), width, m0)
    while out and not any(t.strip() for t, _ in out[-1]):
        out.pop()
    return out


def fmt_secs(s):
    s = int(s)
    return f"{s // 60}m{s % 60:02d}s" if s >= 60 else f"{s}s"


def fmt_k(n):
    return f"{n / 1000:.1f}k"


# ------------------------------------------------------------------ transcript blocks

class Block:
    gap = True  # blank line before this block

    def __init__(self):
        self.version = 0
        self._cache = None

    def touch(self):
        self.version += 1

    def lines(self, width, app):
        key = (width, self.version, app.show_thinking, app.expand_tools)
        if self._cache is None or self._cache[0] != key:
            self._cache = (key, self.render(width, app))
        return self._cache[1]

    def render(self, width, app):
        return []


class UserBlock(Block):
    def __init__(self, text):
        super().__init__()
        self.text = text

    def render(self, width, app):
        out = []
        for i, line in enumerate(clean(self.text).split("\n")):
            out += wrap([(line, S["bold"])], width, [("❯ " if i == 0 else "  ", S["accent"] | S["bold"])],
                        [("  ", 0)])
        return out


class AssistantBlock(Block):
    def __init__(self, text=""):
        super().__init__()
        self.text = text

    def append(self, t):
        self.text += t
        self.touch()

    def render(self, width, app):
        lines = markdown(self.text, width)
        if lines:
            first = lines[0]
            lines[0] = [("● ", S["normal"])] + ([(first[0][0][2:], first[0][1])] + first[1:]
                                                if first and first[0][0].startswith("  ") else first)
        return lines


class ThinkingBlock(Block):
    def __init__(self):
        super().__init__()
        self.text, self.tokens, self.t0, self.t1 = "", 0, time.time(), None

    def append(self, t):
        self.text += t
        self.tokens += 1
        self.touch()

    def finish(self):
        if self.t1 is None:
            self.t1 = time.time()
            self.touch()

    def render(self, width, app):
        running = self.t1 is None
        label = "Thinking…" if running else f"Thought for {fmt_secs(self.t1 - self.t0)}"
        head = [("✻ ", S["accent"]), (label, S["gray"] | S["italic"]), (f" · {self.tokens} tok", S["gray"])]
        if not app.show_thinking:
            return [head + [("  (ctrl-t to show)", S["gray"])]]
        out = [head]
        for para in clean(self.text.strip()).split("\n"):
            out += wrap([(para, S["gray"] | S["italic"])], width, [("  ", 0)])
        return out


class ToolBlock(Block):
    DOT = {"running": "yellow", "ok": "green", "error": "red", "denied": "yellow"}

    def __init__(self, name, summary, sub=False):
        super().__init__()
        self.name, self.summary, self.sub = name, summary, sub
        self.state, self.result, self.secs, self.diff = "running", "", 0.0, None

    def set_diff(self, path, old, new):
        lines = list(difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=2))[2:]
        self.diff = lines or ["(no changes)"]
        self.touch()

    def finish(self, text, state, secs):
        self.result, self.state, self.secs = text, state, secs
        self.touch()

    def render(self, width, app):
        ind = "    " if self.sub else ""
        title = self.name if self.name != "!" else "Shell"
        out = wrap([(title, S["bold"]), (f"({clean(self.summary)})" if self.summary else "", S["gray"])],
                   width, [(ind + "● ", S[self.DOT[self.state]])], [(ind + "  ", 0)])
        expanded = app.expand_tools
        if self.diff:
            limit = 400 if expanded else 30
            for d in self.diff[:limit]:
                attr = (S["green"] if d.startswith("+") else S["red"] if d.startswith("-")
                        else S["cyan"] if d.startswith("@@") else S["gray"])
                out += wrap([(clean(d), attr)], width, [(ind + "    ", 0)], hard=True)
            if len(self.diff) > limit:
                out.append([(ind + f"    … +{len(self.diff) - limit} diff lines (ctrl-o)", S["gray"])])
        if self.state == "running" or (self.sub and self.state == "ok"):
            return out
        text = clean(self.result.rstrip()) or "(no output)"
        rows = text.split("\n")
        attr = {"error": S["red"], "denied": S["yellow"]}.get(self.state, S["gray"])
        if self.name == "TodoWrite" and self.state == "ok":
            attr, limit = S["cyan"], 100
        else:
            limit = 300 if expanded else (8 if self.name == "Task" else 4)
        first, rest = [(ind + "  ⎿  ", S["gray"])], [(ind + "     ", 0)]
        for i, r in enumerate(rows[:limit]):
            if expanded or self.name == "TodoWrite":
                out += wrap([(r, attr)], width, first if i == 0 else rest, rest, hard=True)
            else:
                pre = first if i == 0 else rest
                out.append(pre + [(ellipsize(r, width - 7 - len(ind)), attr)])
        extra = []
        if len(rows) > limit:
            extra.append(f"… +{len(rows) - limit} lines (ctrl-o to expand)")
        if self.secs > 3:
            extra.append(f"{self.secs:.1f}s")
        if extra:
            out.append(rest + [(" · ".join(extra), S["gray"])])
        return out


class NoticeBlock(Block):
    def __init__(self, text, level="info", gap=True):
        super().__init__()
        self.text, self.level, self.gap = text, level, gap

    def render(self, width, app):
        attr = {"info": S["gray"], "warn": S["yellow"], "error": S["red"]}[self.level]
        out = []
        for line in clean(self.text).split("\n"):
            out += wrap([(line, attr)], width, [("  ", 0)])
        return out


class MarkdownBlock(Block):
    def __init__(self, title, text):
        super().__init__()
        self.title, self.text = title, text

    def render(self, width, app):
        return [[("  " + self.title, S["accent"] | S["bold"])]] + markdown(self.text, width, margin="    ")


# ------------------------------------------------------------------ input editor

class Editor:
    def __init__(self):
        self.buf, self.cur = "", 0
        self.history, self.hidx, self.saved = [], None, ""

    def set(self, text):
        self.buf, self.cur = text, len(text)

    def insert(self, s):
        self.buf = self.buf[:self.cur] + s + self.buf[self.cur:]
        self.cur += len(s)

    def backspace(self):
        if self.cur:
            self.buf = self.buf[:self.cur - 1] + self.buf[self.cur:]
            self.cur -= 1

    def delete(self):
        self.buf = self.buf[:self.cur] + self.buf[self.cur + 1:]

    def delete_word(self):
        i = self.cur
        while i and self.buf[i - 1].isspace():
            i -= 1
        while i and not self.buf[i - 1].isspace():
            i -= 1
        self.buf, self.cur = self.buf[:i] + self.buf[self.cur:], i

    def line_bounds(self):
        s = self.buf.rfind("\n", 0, self.cur) + 1
        e = self.buf.find("\n", self.cur)
        return s, len(self.buf) if e < 0 else e

    def home(self):
        self.cur = self.line_bounds()[0]

    def end(self):
        self.cur = self.line_bounds()[1]

    def kill_to_start(self):
        s = self.line_bounds()[0]
        self.buf, self.cur = self.buf[:s] + self.buf[self.cur:], s

    def kill_to_end(self):
        e = self.line_bounds()[1]
        self.buf = self.buf[:self.cur] + self.buf[e:]

    def layout(self, width):
        rows, start, col, i, b = [], 0, 0, 0, self.buf
        while i < len(b):
            if b[i] == "\n":
                rows.append((start, i))
                start, col = i + 1, 0
                i += 1
                continue
            c = cw(b[i])
            if col + c > width and col > 0:
                rows.append((start, i))
                start, col = i, 0
                continue
            col += c
            i += 1
        rows.append((start, len(b)))
        return rows

    def cursor_rc(self, width):
        rows = self.layout(width)
        for r, (s, e) in enumerate(rows):
            if s <= self.cur <= e:
                if self.cur == e and r + 1 < len(rows) and rows[r + 1][0] == e:
                    continue  # soft-wrapped: the cursor belongs at the start of the next row
                return r, sw(self.buf[s:self.cur]), rows
        return len(rows) - 1, sw(self.buf[rows[-1][0]:self.cur]), rows

    def vertical(self, width, d):
        r, col, rows = self.cursor_rc(width)
        if not 0 <= r + d < len(rows):
            return False
        s, e = rows[r + d]
        i, w = s, 0
        while i < e and w + cw(self.buf[i]) <= col:
            w += cw(self.buf[i])
            i += 1
        self.cur = i
        return True

    def history_move(self, d):
        if not self.history:
            return
        if self.hidx is None:
            if d > 0:
                return
            self.saved, self.hidx = self.buf, len(self.history)
        self.hidx += d
        if self.hidx < 0:
            self.hidx = 0
        if self.hidx >= len(self.history):
            self.hidx = None
            self.set(self.saved)
        else:
            self.set(self.history[self.hidx])

    def remember(self, text):
        if text.strip() and (not self.history or self.history[-1] != text):
            self.history.append(text)
        self.history = self.history[-500:]
        self.hidx = None


# ------------------------------------------------------------------ the view the agent reports to

class TuiView(View):
    def __init__(self, app):
        self.app = app
        self.think = None
        self.answer = None
        self.last_tool = {}
        self.n = 0
        self.t_first = None

    def model_start(self, sub=False):
        self.n, self.t_first = 0, None
        self.think = self.answer = None
        self.app.set_status("Researching (sub-agent)" if sub else "Waiting for the model (reading the prompt)")

    def reasoning(self, text, sub=False):
        self.n += 1
        now = time.time()
        self.t_first = self.t_first or now
        rate = f" · {self.n / (now - self.t_first):.1f} tok/s" if now - self.t_first > 1 else ""
        self.app.set_status(f"{'Researching' if sub else 'Thinking'} · {self.n} tok{rate}")
        if sub:
            return
        with self.app.lock:
            if self.think is None:
                self.think = self.app.add(ThinkingBlock())
            self.think.append(text)

    def text(self, text, sub=False):
        if sub:
            return
        self.app.set_status("Writing")
        with self.app.lock:
            if self.think:
                self.think.finish()
            if self.answer is None:
                self.answer = self.app.add(AssistantBlock())
            self.answer.append(text)

    def model_end(self, sub=False):
        with self.app.lock:
            if self.think:
                self.think.finish()
        self.app.set_status("Working")

    def status(self, label):
        self.app.set_status(label or "Working")

    def tool_start(self, name, summary, sub=False):
        with self.app.lock:
            self.last_tool[sub] = self.app.add(ToolBlock(name, summary, sub))
        self.app.set_status(f"Running {name}")

    def tool_diff(self, path, old, new, sub=False):
        with self.app.lock:
            self.last_tool[sub].set_diff(path, old, new)

    def tool_result(self, name, text, status="ok", secs=0.0, sub=False):
        with self.app.lock:
            block = self.last_tool.get(sub)
            if block is not None and block.state == "running":
                block.finish(text, status, secs)
        self.app.set_status("Working")

    def file_changed(self, path, added, removed):
        with self.app.lock:
            a, r = self.app.modified.get(path, (0, 0))
            self.app.modified[path] = (a + added, r + removed)

    def todos(self, todos):
        self.app.todos = list(todos)

    def notice(self, msg, level="info"):
        with self.app.lock:
            self.app.add(NoticeBlock(msg.strip(), level))

    def turn_end(self, stats):
        speed = f"{stats['tps']:.1f} tok/s · " if stats["tps"] else ""
        with self.app.lock:
            self.app.add(NoticeBlock(f"✻ {stats['tokens']} tok · {speed}{fmt_secs(stats['secs'])}", "info",
                                     gap=False))

    def interrupted(self):
        with self.app.lock:
            if self.think:
                self.think.finish()
            for b in self.last_tool.values():
                if b.state == "running":
                    b.finish("Interrupted", "denied", 0)
            self.app.add(NoticeBlock("⎿ Interrupted · tell the model what to do instead", "warn"))


# ------------------------------------------------------------------ the app

class App:
    SIDEBAR_W = 34

    def __init__(self, cfg, client, perms, agent, session):
        self.cfg, self.client, self.perms, self.agent, self.session = cfg, client, perms, agent, session
        self.lock = threading.RLock()
        self.view = TuiView(self)
        agent.view = self.view
        perms.prompter = self.ask_permission
        client.notify = lambda msg: self.view.notice(msg.strip(), "warn")

        self.blocks = []
        self.modified = {}
        self.todos = list(agent.ctx.todos)
        self.pending = []  # context added to the next message (shell output, undo notes)
        self.queue = []  # messages typed while the model was busy

        self.ed = Editor()
        self.mode = "normal"  # normal | feedback | secret
        self.perm = None  # pending permission request
        self.popup, self.popup_sel, self.popup_off = [], 0, False

        self.busy = False
        self.busy_since = 0.0
        self.status_label = ""
        self.running = True
        self.scroll = 0
        self._last_total = 0
        self.show_thinking = cfg.show_reasoning
        self.expand_tools = False
        self.show_sidebar = True
        self.server = "checking"
        self.last_ctrl_c = 0.0
        self.flash = ("", 0.0)
        self._ctx_cache = (0.0, 0)
        self._load_history()

    # ---------------------------------------------------------------- state helpers

    def add(self, block):
        with self.lock:
            self.blocks.append(block)
        return block

    def set_status(self, label):
        self.status_label = label

    def notify(self, text, level="info"):
        self.add(NoticeBlock(text, level))

    def say(self, text):
        self.flash = (text, time.time())

    def context_used(self):
        t, v = self._ctx_cache
        if time.time() - t > 0.5:
            try:
                v = self.agent.context_tokens()
            except (RuntimeError, ValueError):
                pass
            self._ctx_cache = (time.time(), v)
        return v

    def _load_history(self):
        try:
            with open(HISTORY_FILE) as f:
                self.ed.history = json.load(f)[-500:]
        except (OSError, json.JSONDecodeError):
            pass

    def _save_history(self):
        try:
            os.makedirs(HOME, exist_ok=True)
            with open(HISTORY_FILE, "w") as f:
                json.dump(self.ed.history[-500:], f)
        except OSError:
            pass

    def save_session(self):
        try:
            self.session.save(self.agent.messages, self.cfg.model, self.agent.ctx.todos, self.agent.summaries)
        except OSError as e:
            self.notify(f"could not save session: {e}", "error")

    # ---------------------------------------------------------------- background work

    def run_worker(self, fn, label="Working"):
        self.busy, self.busy_since, self.status_label = True, time.time(), label
        self.scroll = 0

        def target():
            try:
                fn()
            except Cancelled:
                self.view.interrupted()
            except LLMError as e:
                self.notify(f"API error: {e}", "error")
            except Exception as e:  # keep the app alive; log details
                self.notify(f"Internal error: {type(e).__name__}: {e} (details in {LOG_FILE})", "error")
                try:
                    with open(LOG_FILE, "a") as f:
                        f.write(f"--- {time.ctime()}\n{traceback.format_exc()}\n")
                except OSError:
                    pass
            finally:
                self.todos = list(self.agent.ctx.todos)
                self.save_session()
                self.busy = False
                self.status_label = ""

        threading.Thread(target=target, daemon=True).start()

    def interrupt(self):
        if self.perm:
            self.resolve_permission("n", "")
        self.agent.cancel()
        self.set_status("Interrupting")

    def health_loop(self):
        while self.running:
            self.server = "online" if self.client.health(timeout=8) else "offline"
            for _ in range(300):
                if not self.running:
                    return
                time.sleep(0.1)

    # ---------------------------------------------------------------- permissions (called from the worker)

    def ask_permission(self, tool, args, always):
        req = {"tool": tool, "args": args, "always": always, "event": threading.Event(), "result": ("n", ""),
               "sel": 0}
        with self.lock:
            self.perm = req
        self.set_status("Waiting for your approval")
        self.scroll = 0
        req["event"].wait()
        self.set_status("Working")
        return req["result"]

    def resolve_permission(self, choice, feedback=""):
        req = self.perm
        if req is None:
            return
        req["result"] = (choice, feedback)
        self.perm = None
        if self.mode == "feedback":
            self.mode = "normal"
            self.ed.set(req.get("draft", ""))
        req["event"].set()

    def permission_choose(self, idx):
        if idx == 2:  # "No": ask what to do instead
            self.perm["draft"] = self.ed.buf
            self.ed.set("")
            self.mode = "feedback"
        else:
            self.resolve_permission("ya"[idx])

    # ---------------------------------------------------------------- submitting input

    def submit(self):
        text = self.ed.buf
        if self.mode == "feedback":
            self.resolve_permission("n", text.strip())
            return
        if self.mode == "secret":
            self.mode = "normal"
            self.ed.set("")
            self.finish_login(text.strip())
            return
        if not text.strip():
            return
        self.ed.remember(text)
        self.ed.set("")
        self.popup = []
        stripped = text.strip()
        if stripped.startswith("/") and "\n" not in stripped:
            self.command(stripped)
        elif stripped.startswith("!"):
            self.shell(stripped[1:].strip())
        elif self.busy:
            self.queue.append(text)
            self.say(f"queued — sent when the model finishes ({len(self.queue)} waiting)")
        else:
            self.send(text)

    def expand_mentions(self, text):
        ctx, attachments = self.agent.ctx, []
        for m in re.finditer(r"(?<![\w@])@([\w./~-][^\s]*)", text):
            raw = m.group(1).rstrip(".,;:!?)'\"")
            try:
                path = ctx.resolve(raw)
                if os.path.isfile(path):
                    attachments.append(f'<file path="{path}">\n{t_read(ctx, path)}\n</file>')
                elif os.path.isdir(path):
                    attachments.append(f'<directory path="{path}">\n{t_ls(ctx, path)}\n</directory>')
                else:
                    continue
            except (ToolError, OSError) as e:
                self.notify(f"could not attach {raw}: {e}", "warn")
                continue
            self.add(NoticeBlock(f"⎿ attached {ctx.rel(path)}", gap=False))
        if attachments:
            text += "\n\nAttached by the user:\n" + "\n\n".join(attachments)
        return text

    def send(self, text, display=None):
        self.add(UserBlock(display or text))
        content = self.expand_mentions(text)
        if self.pending:
            content = "\n\n".join(self.pending) + "\n\n" + content
            self.pending = []
        self.run_worker(lambda: self.agent.run(content), "Waiting for the model (reading the prompt)")

    def shell(self, cmd):
        if not cmd:
            return
        if self.busy:
            self.say("wait for the current turn to finish (esc to interrupt)")
            return

        def fn():
            self.client.reset_cancel()
            with self.lock:
                block = self.add(ToolBlock("!", cmd))
            t0 = time.time()
            try:
                output = t_bash(self.agent.ctx, cmd)
            except Cancelled:
                block.finish("Interrupted", "denied", time.time() - t0)
                return
            block.finish(output, "ok", time.time() - t0)
            self.pending.append(f"<user-shell-command>{cmd}</user-shell-command>\n"
                                f"<user-shell-output>\n{output[: self.cfg.max_tool_output]}\n</user-shell-output>")

        self.run_worker(fn, f"Running {cmd[:40]}")

    # ---------------------------------------------------------------- slash commands

    def command(self, line):
        name, _, arg = line.partition(" ")
        arg = arg.strip()
        a = self.agent
        if name in ("/exit", "/quit"):
            self.running = False
            return
        busy_blocked = {"/clear", "/compact", "/init", "/resume", "/undo", "/model", "/mode"}
        if self.busy and name in busy_blocked:
            self.say("wait for the current turn to finish (esc to interrupt)")
            return
        if name == "/help":
            cmds = "\n".join(f"{c.ljust(10)} {d}" for c, d in COMMANDS.items())
            self.notify(f"Commands\n{cmds}\n\nKeys\n{KEYS_HELP}")
        elif name == "/clear":
            a.reset()
            with self.lock:
                self.blocks, self.modified, self.todos = [], {}, []
            self.session = Session(a.ctx.cwd)
            self.say("new conversation")
        elif name == "/compact":
            self.run_worker(lambda: (a.compact(arg or None, manual=True)), "Summarizing")
        elif name == "/context":
            used, win = self.context_used(), self.cfg.context_window
            self.notify(f"context {fmt_k(used)} / {fmt_k(win)} tokens ({used / win:.0%}) · {len(a.messages)} "
                        f"messages · summarized automatically at {self.cfg.compact_threshold:.0%} · "
                        f"{len(a.summaries)} summaries so far")
        elif name == "/summary":
            if a.summaries:
                last = a.summaries[-1]
                self.add(MarkdownBlock(f"Summary {len(a.summaries)} · made from ~{fmt_k(last['tokens_before'])} "
                                       "tokens", last["summary"]))
            else:
                self.notify("no summaries yet — one is made automatically when the context fills up, or "
                            "with /compact")
        elif name == "/mode":
            if arg and arg not in MODES:
                self.say(f"unknown mode; choose from {', '.join(MODES)}")
                return
            self.set_mode(arg or MODES[(MODES.index(self.perms.mode) + 1) % len(MODES)])
        elif name == "/model":
            if arg:
                self.cfg.model = arg
                a.refresh_system_prompt()
                self.say(f"model set to {arg}")
            else:
                def fn():
                    try:
                        models = self.client.list_models()
                        self.notify(f"current: {self.cfg.model}\navailable: {', '.join(models)}")
                    except LLMError as e:
                        self.notify(str(e), "error")
                threading.Thread(target=fn, daemon=True).start()
        elif name == "/allow":
            if arg:
                self.perms.allowed_prefixes.append(arg)
                self.say(f"`{arg}` commands will run without asking")
            else:
                self.notify("allowed prefixes: " + (", ".join(self.perms.allowed_prefixes) or "(none)"))
        elif name == "/undo":
            restored, errors = a.undo_last_turn()
            for e in errors:
                self.notify(f"could not restore {e}", "error")
            if restored:
                for p in restored:
                    self.modified.pop(p, None)
                self.notify("⎿ reverted: " + ", ".join(restored))
                self.pending.append(f"[The user reverted your file changes from your previous turn "
                                    f"({', '.join(restored)}). Read these files again before editing them.]")
            else:
                self.say("nothing to undo")
        elif name == "/todos":
            self.notify(format_todos(a.ctx.todos))
        elif name == "/init":
            self.send(prompts.INIT, display="/init")
        elif name == "/resume":
            self.resume(arg)
        elif name == "/think":
            from .cli import set_thinking
            self.say(set_thinking(self.cfg, arg))
        elif name == "/login":
            self.start_login()
        elif name == "/stats":
            self.notify(f"prompt tokens {a.total_prompt:,} · output tokens {a.total_completion:,} · "
                        f"last speed {a.last_tps:.1f} tok/s")
        elif name == "/config":
            self.notify("\n".join(f"{k.ljust(18)} {v}" for k, v in self.cfg.as_dict().items()))
        else:
            self.say(f"unknown command {name} — /help lists them")

    def set_mode(self, mode):
        self.perms.mode = mode
        self.agent.refresh_system_prompt()
        self.say(f"mode: {mode} — {MODE_HELP[mode]}")

    def resume(self, arg):
        sessions = list_sessions(self.agent.ctx.cwd)
        if not sessions:
            self.say("no saved sessions in this directory")
            return
        if not arg.isdigit():
            rows = [f"{i:>2}. {s.get('title') or '(untitled)'} · {len(s['messages'])} msgs · "
                    f"{time.strftime('%b %d %H:%M', time.localtime(s.get('updated', 0)))}"
                    for i, s in enumerate(sessions[:20], 1)]
            self.notify("Sessions in this directory — open one with /resume N\n" + "\n".join(rows))
            return
        i = int(arg) - 1
        if 0 <= i < len(sessions):
            self.load(sessions[i])

    def load(self, data):
        a = self.agent
        a.reset()
        a.messages = data["messages"]
        a.ctx.todos = data.get("todos") or []
        a.summaries = data.get("summaries") or []
        self.todos = list(a.ctx.todos)
        self.session = Session(a.ctx.cwd, sid=data["id"], title=data.get("title"))
        with self.lock:
            self.blocks, self.modified = [], {}
            self.replay(data["messages"])
            self.add(NoticeBlock(f"resumed “{data.get('title', '')}”", "info"))

    def replay(self, messages):
        results = {m.get("tool_call_id"): m.get("content", "") for m in messages if m["role"] == "tool"}
        tools = self.agent.tools
        for m in messages:
            if m["role"] == "user":
                c = m["content"] if isinstance(m["content"], str) else str(m["content"])
                if c.startswith("This session is being continued"):
                    self.add(NoticeBlock("✻ earlier conversation summarized (/summary to view)"))
                else:
                    self.add(UserBlock(c.split("\n\nAttached by the user:")[0][:2000]))
            elif m["role"] == "assistant":
                if m.get("content", "").strip():
                    self.add(AssistantBlock(m["content"]))
                for tc in m.get("tool_calls") or []:
                    name = tc["function"]["name"]
                    try:
                        args = json.loads(tc["function"]["arguments"] or "{}")
                        summary = tools[name].summary(args) if name in tools else ""
                    except (json.JSONDecodeError, AttributeError):
                        summary = ""
                    res = results.get(tc["id"], "")
                    b = self.add(ToolBlock(name, summary))
                    b.finish(res, "error" if res.startswith("Error") else "ok", 0)

    # ---------------------------------------------------------------- login

    def start_login(self):
        self.mode = "secret"
        self.ed.set("")

    def finish_login(self, key):
        if not key:
            self.say("login cancelled")
            return

        def fn():
            old, self.cfg.api_key = self.cfg.api_key, key
            try:
                self.client.list_models()
            except LLMError as e:
                if e.status == 401:
                    self.cfg.api_key = old
                    self.notify("✗ the server rejected that key", "error")
                    return
                self.notify(f"could not verify the key right now ({e}); saved anyway", "warn")
            save_credentials(key)
            self.notify(f"✓ key {mask_key(key)} saved (readable only by you)")

        threading.Thread(target=fn, daemon=True).start()

    # ---------------------------------------------------------------- completion popup

    def update_popup(self):
        b, cur = self.ed.buf, self.ed.cur
        items = []
        if self.mode == "normal" and not self.perm and not self.popup_off:
            if b.startswith("/") and " " not in b[:cur] and "\n" not in b:
                items = [(c, d, "cmd") for c, d in COMMANDS.items() if c.startswith(b[:cur])]
            elif (m := re.search(r"(?:^|\s)@([^\s]*)$", b[:cur])):
                partial = m.group(1)
                d, base = os.path.split(partial)
                full = os.path.join(self.agent.ctx.cwd, os.path.expanduser(d) or ".")
                try:
                    names = sorted(os.listdir(full))
                except OSError:
                    names = []
                for n in names:
                    if n.startswith(base) and (base.startswith(".") or not n.startswith(".")):
                        is_dir = os.path.isdir(os.path.join(full, n))
                        items.append((os.path.join(d, n) + ("/" if is_dir else ""), "dir" if is_dir else "",
                                      "path"))
                    if len(items) >= 50:
                        break
        if [i[0] for i in items] != [i[0] for i in self.popup]:
            self.popup_sel = 0
        self.popup = items

    def accept_popup(self, submit_cmd=False):
        value, _, kind = self.popup[self.popup_sel]
        if kind == "cmd":
            self.ed.set(value + ("" if submit_cmd else " "))
            self.popup = []
            if submit_cmd:
                self.submit()
            return
        b, cur = self.ed.buf, self.ed.cur
        start = b.rfind("@", 0, cur) + 1
        tail = "" if value.endswith("/") else " "
        self.ed.buf = b[:start] + value + tail + b[cur:]
        self.ed.cur = start + len(value) + len(tail)
        self.update_popup()

    # ---------------------------------------------------------------- keys

    def handle_keys(self, keys):
        before = self.ed.buf
        i = 0
        paste = False
        while i < len(keys):
            k = keys[i]
            i += 1
            if k == "\x1b":
                rest = "".join(x for x in keys[i:i + 5] if isinstance(x, str))
                if rest.startswith("[200~"):
                    paste, i = True, i + 5
                    continue
                if rest.startswith("[201~"):
                    paste, i = False, i + 5
                    continue
                if i < len(keys) and keys[i] in ("\r", "\n"):  # alt-enter
                    self.ed.insert("\n")
                    i += 1
                    continue
                if i < len(keys) and keys[i] in ("[", "O"):
                    # Unrecognized CSI/SS3 sequence: skip through its final byte.
                    i += 1
                    while i < len(keys) and isinstance(keys[i], str) and not ("@" <= keys[i] <= "~"):
                        i += 1
                    if i < len(keys) and keys[i] == "Z":  # shift-tab sent raw
                        self.on_key(curses.KEY_BTAB)
                    i += 1
                    continue
                if i < len(keys) and isinstance(keys[i], str) and keys[i] != "\x1b":
                    i += 1  # alt+key: ignore
                    continue
                self.on_escape()
                continue
            if paste:
                if isinstance(k, str):
                    self.ed.insert("\n" if k in "\r\n" else k)
                continue
            # A newline that arrives together with more keys is part of a paste (no bracketed paste support).
            if k == "\r" and i < len(keys) and not (self.perm and self.mode != "feedback"):
                self.ed.insert("\n")
                continue
            self.on_key(k)
        if self.ed.buf != before:
            self.popup_off = False  # typing re-opens a dismissed completion popup
        self.update_popup()

    def on_escape(self):
        if self.popup:
            self.popup, self.popup_off = [], True
        elif self.mode == "feedback":
            self.resolve_permission("n", "")
        elif self.perm:
            self.resolve_permission("n", "")
        elif self.mode == "secret":
            self.mode = "normal"
            self.ed.set("")
        elif self.busy:
            self.interrupt()

    def on_key(self, k):
        ed, width = self.ed, self.input_width()
        # permission prompt keys (when not typing feedback)
        if self.perm and self.mode != "feedback":
            if k in ("y", "1"):
                self.permission_choose(0)
            elif k in ("a", "2"):
                self.permission_choose(1)
            elif k in ("n", "3"):
                self.permission_choose(2)
            elif k in (curses.KEY_UP, curses.KEY_BTAB):
                self.perm["sel"] = (self.perm["sel"] - 1) % 3
            elif k in (curses.KEY_DOWN, "\t"):
                self.perm["sel"] = (self.perm["sel"] + 1) % 3
            elif k in ("\r", curses.KEY_ENTER):
                self.permission_choose(self.perm["sel"])
            elif k == "\x03":
                self.interrupt()
            elif k in (curses.KEY_PPAGE, curses.KEY_NPAGE, curses.KEY_MOUSE):
                self.scroll_key(k)
            return
        if k in ("\r", curses.KEY_ENTER):
            if self.popup:
                self.accept_popup(submit_cmd=True)
            else:
                self.submit()
        elif k == "\n":
            ed.insert("\n")
        elif k == "\t":
            if self.popup:
                self.accept_popup()
        elif k == curses.KEY_BTAB:
            if not self.busy:
                self.set_mode(MODES[(MODES.index(self.perms.mode) + 1) % len(MODES)])
        elif k in ("\x7f", "\x08", curses.KEY_BACKSPACE):
            ed.backspace()
        elif k == curses.KEY_DC:
            ed.delete()
        elif k == curses.KEY_LEFT:
            ed.cur = max(0, ed.cur - 1)
        elif k == curses.KEY_RIGHT:
            ed.cur = min(len(ed.buf), ed.cur + 1)
        elif k in (curses.KEY_HOME, "\x01"):
            ed.home()
        elif k in (curses.KEY_END, "\x05"):
            if not ed.buf and k == curses.KEY_END:
                self.scroll = 0
            ed.end()
        elif k == curses.KEY_UP:
            if self.popup:
                self.popup_sel = (self.popup_sel - 1) % len(self.popup)
            elif not ed.vertical(width, -1):
                ed.history_move(-1)
        elif k == curses.KEY_DOWN:
            if self.popup:
                self.popup_sel = (self.popup_sel + 1) % len(self.popup)
            elif not ed.vertical(width, 1):
                ed.history_move(1)
        elif k == "\x15":
            ed.kill_to_start()
        elif k == "\x0b":
            ed.kill_to_end()
        elif k == "\x17":
            ed.delete_word()
        elif k == "\x03":
            self.on_ctrl_c()
        elif k == "\x04":
            if not ed.buf and not self.busy:
                self.running = False
        elif k == "\x14":
            self.show_thinking = not self.show_thinking
        elif k == "\x0f":
            self.expand_tools = not self.expand_tools
        elif k == "\x02":
            self.show_sidebar = not self.show_sidebar
        elif k == "\x0c":
            self.stdscr.clear()
        elif k in (curses.KEY_PPAGE, curses.KEY_NPAGE, curses.KEY_MOUSE):
            self.scroll_key(k)
        elif k == curses.KEY_RESIZE:
            pass
        elif isinstance(k, str) and k.isprintable():
            ed.insert(k)

    def on_ctrl_c(self):
        if self.busy:
            self.interrupt()
        elif self.ed.buf:
            self.ed.set("")
        elif self.mode != "normal":
            self.mode = "normal"
        elif time.time() - self.last_ctrl_c < 1.5:
            self.running = False
        else:
            self.last_ctrl_c = time.time()
            self.say("press ctrl-c again to quit")

    def scroll_key(self, k):
        page = max(1, self.trans_h - 3)
        if k == curses.KEY_PPAGE:
            self.scroll += page
        elif k == curses.KEY_NPAGE:
            self.scroll = max(0, self.scroll - page)
        elif k == curses.KEY_MOUSE:
            try:
                _, _, _, _, bstate = curses.getmouse()
            except curses.error:
                return
            if bstate & curses.BUTTON4_PRESSED:
                self.scroll += 3
            elif bstate & getattr(curses, "BUTTON5_PRESSED", 0x200000):
                self.scroll = max(0, self.scroll - 3)

    # ---------------------------------------------------------------- drawing

    def input_width(self):
        return max(10, self.width - 6)

    def put(self, y, x, segs, maxw):
        w = 0
        for text, attr in segs:
            if w >= maxw or not text:
                continue
            t = clip(text, maxw - w)
            try:
                self.stdscr.addstr(y, x + w, t, attr)
            except curses.error:
                pass
            w += sw(t)
        return w

    def bottom_lines(self, width):
        """Status line + input box (or permission prompt) + hints, as (lines, cursor position or None)."""
        lines, cursor = [], None
        inner = width - 4
        # status line
        if self.busy:
            frame = SPIN[int(time.time() * 10) % len(SPIN)]
            left = [(f"{frame} ", S["accent"]), (f"{self.status_label} · {fmt_secs(time.time() - self.busy_since)}",
                                                 S["accent"] if "approval" in self.status_label else S["normal"])]
            right = "esc to interrupt" + (f" · {len(self.queue)} queued" if self.queue else "")
        else:
            msg, t = self.flash
            left = [(msg, S["yellow"])] if msg and time.time() - t < 4 else []
            right = ""
        rw = sw(right)
        lw = sum(sw(t) for t, _ in left)
        pad = max(1, width - 2 - lw - rw)
        lines.append([(" ", 0)] + left + [(" " * pad, 0), (right, S["gray"])])

        if self.perm and self.mode != "feedback":
            lines += self.permission_lines(width)
        else:
            if self.mode == "feedback":
                title, border = " What should the model do instead? (enter to send, esc to skip) ", S["yellow"]
            elif self.mode == "secret":
                title, border = " Paste your API key (hidden) — enter to save, esc to cancel ", S["yellow"]
            else:
                badge = self.perms.mode
                title, border = f" {badge} ", S["gray"] if self.busy else S["accent"]
            top = "╭─" + title + "─" * max(0, width - 4 - sw(title)) + "─╮"
            lines.append([(clip(top, width), border)])
            if self.mode == "secret":
                shown = "•" * len(self.ed.buf)
                rows = [(shown, 0, len(shown))]
                r, col = 0, min(sw(shown), inner - 2)
                rows_text = [clip(shown[-(inner - 2):], inner - 2)]
            else:
                r, col, rws = self.ed.cursor_rc(inner - 2)
                rows_text = [self.ed.buf[s:e] for s, e in rws]
            max_rows = 8
            first = max(0, r - max_rows + 1)
            visible = rows_text[first:first + max_rows]
            for j, row in enumerate(visible):
                prompt = "❯ " if j + first == 0 else "  "
                lines.append([("│ ", border), (prompt, S["accent"] | S["bold"]),
                              (row, 0), (" " * max(0, inner - 2 - sw(row)), 0), (" │", border)])
            if not self.ed.buf and self.mode == "normal":
                ph = "Message the agent…  (/ for commands, @ to attach files)" if not self.busy else \
                    "Type to queue a message…"
                lines[-1] = [("│ ", border), ("❯ ", S["accent"] | S["bold"]), (ph, S["gray"]),
                             (" " * max(0, inner - 2 - sw(ph)), 0), (" │", border)]
            cursor = (len(lines) - len(visible) + (r - first), 4 + col)
            lines.append([("╰" + "─" * (width - 2) + "╯", border)])
        hints = ("enter send · ctrl-j newline · esc interrupt · shift-tab mode · ctrl-t thinking · "
                 "ctrl-o outputs · ctrl-b sidebar · /help")
        lines.append([(" " + ellipsize(hints, width - 2), S["gray"])])
        return lines, cursor

    def permission_lines(self, width):
        p = self.perm
        tool, args = p["tool"], p["args"]
        inner = width - 4
        border = S["yellow"]
        if tool.kind == "exec":
            question = "Run this command?"
            detail = args.get("command", "")
        elif tool.kind == "edit":
            question = f"Apply this edit to {self.agent.ctx.rel(self.agent.ctx.resolve(args.get('file_path', '')))}?"
            detail = "(the diff is shown above)"
        else:
            question = f"Allow {tool.name}?"
            detail = args.get("url", json.dumps(args))
        title = " Permission required "
        out = [[("╭─" + title + "─" * max(0, width - 4 - sw(title)) + "─╮", border)]]

        def row(segs):
            w = sum(sw(t) for t, _ in segs)
            out.append([("│ ", border)] + segs + [(" " * max(0, inner - w), 0), (" │", border)])

        row([(ellipsize(question, inner), S["bold"])])
        for dl in [x for line in clean(detail).split("\n")[:6] for x in wrap([(line, S["cyan"])], inner - 2)][:6]:
            row([("  ", 0)] + [(ellipsize("".join(t for t, _ in dl), inner - 2), S["cyan"])])
        options = ["Yes", f"Yes, {p['always']}", "No, and tell the model what to do instead"]
        for idx, opt in enumerate(options):
            sel = idx == p["sel"]
            row([("❯ " if sel else "  ", S["accent"]), (f"{idx + 1}. ", S["gray"]),
                 (ellipsize(opt, inner - 5), S["bold"] if sel else 0)])
        out.append([("╰" + "─" * (width - 2) + "╯", border)])
        return out

    def sidebar_lines(self, width):
        L = []

        def head(t):
            if L:
                L.append([])
            L.append([(t, S["bold"])])

        head("Session")
        title = self.session.title or next((b.text for b in self.blocks if isinstance(b, UserBlock)), "New session")
        for line in wrap([(clean(title).split("\n")[0], S["gray"])], width)[:2]:
            L.append(line)

        head("Context")
        used, win = self.context_used(), self.cfg.context_window
        pct = min(1.0, used / win) if win else 0
        bar_w = width - 6
        filled = int(bar_w * pct)
        color = S["red"] if pct >= self.cfg.compact_threshold else S["yellow"] if pct >= 0.5 else S["accent"]
        L.append([("█" * filled, color), ("░" * (bar_w - filled), S["gray"]), (f" {pct:>3.0%}", 0)])
        L.append([(f"{fmt_k(used)} / {fmt_k(win)} tokens", S["gray"])])
        L.append([(f"summary at {self.cfg.compact_threshold:.0%}", S["gray"])] +
                 ([(f" · {len(self.agent.summaries)} made", S["gray"])] if self.agent.summaries else []))

        head("Server")
        dot = {"online": S["green"], "offline": S["red"]}.get(self.server, S["gray"])
        L.append([("● ", dot), (self.server, 0)])
        L.append([(ellipsize(self.cfg.model, width), S["gray"])])
        L.append([("mode ", S["gray"]), (self.perms.mode, S["yellow"] if self.perms.mode == "bypass" else 0)])
        from .cli import thinking_label
        L.append([("think ", S["gray"]), (thinking_label(self.cfg), 0)])

        if self.todos:
            head("Todos")
            done = sum(1 for t in self.todos if t["status"] == "completed")
            L[-1].append((f" {done}/{len(self.todos)}", S["gray"]))
            for t in self.todos:
                mark, attr = {"completed": ("☒ ", S["gray"]), "in_progress": ("◐ ", S["yellow"]),
                              "pending": ("☐ ", 0)}[t["status"]]
                L += wrap([(t["content"], attr)], width, [(mark, attr)], [("  ", 0)])

        if self.modified:
            head("Modified files")
            for path, (a, r) in sorted(self.modified.items()):
                stats = f" +{a} -{r}"
                L.append([(ellipsize(path, width - sw(stats)), 0), (f" +{a}", S["green"]), (f" -{r}", S["red"])])

        head("Usage")
        L.append([(f"↑ {fmt_k(self.agent.total_prompt)} in · ↓ {fmt_k(self.agent.total_completion)} out",
                   S["gray"])])
        if self.agent.last_tps:
            L.append([(f"{self.agent.last_tps:.1f} tok/s last response", S["gray"])])
        return L

    def draw(self):
        scr = self.stdscr
        h, w = scr.getmaxyx()
        self.width = w
        scr.erase()
        if h < 12 or w < 50:
            self.put(0, 0, [("terminal too small", S["red"])], w)
            scr.refresh()
            return
        side_w = self.SIDEBAR_W if self.show_sidebar and w >= 100 else 0
        main_w = w - side_w - (1 if side_w else 0)

        # header
        cwd = self.agent.ctx.cwd.replace(os.path.expanduser("~"), "~", 1)
        right = f"{self.cfg.model}  v{__version__} "
        left = [(" ✻ hc ", S["badge"]), (f"  {ellipsize(cwd, max(10, w - sw(right) - 12))}", S["bold"])]
        self.put(0, 0, left, w)
        self.put(0, max(0, w - sw(right)), [(right, S["gray"])], w)

        bottom, cursor = self.bottom_lines(w)
        top = 2
        self.trans_h = trans_h = max(1, h - top - len(bottom))
        content_w = main_w - 2

        with self.lock:
            lines = []
            for b in self.blocks:
                if b.gap and lines:
                    lines.append([])
                lines.extend(b.lines(content_w, self))
        total = len(lines)
        if self.scroll and total > self._last_total:
            self.scroll += total - self._last_total  # keep the view still while scrolled up
        self._last_total = total
        self.scroll = max(0, min(self.scroll, total - trans_h)) if total > trans_h else 0
        end = total - self.scroll
        visible = lines[max(0, end - trans_h):end]
        if not lines:
            self.welcome(top, content_w)
        for j, line in enumerate(visible):
            self.put(top + j, 1, line, content_w)
        if self.scroll:
            tag = f" ↓ {self.scroll} more lines · end to jump "
            self.put(top + trans_h - 1, max(0, main_w - sw(tag) - 1), [(tag, S["badge_warn"])], main_w)

        # sidebar
        if side_w:
            for y in range(1, top + trans_h):
                self.put(y, main_w, [("│", S["gray"])], 1)
            for j, line in enumerate(self.sidebar_lines(side_w - 2)[:trans_h]):
                self.put(top + j, main_w + 2, line, side_w - 2)

        # completion popup (overlays the transcript, just above the input)
        by = h - len(bottom)
        if self.popup:
            items = self.popup[:8]
            off = max(0, self.popup_sel - 7)
            items = self.popup[off:off + 8]
            pw = min(64, w - 4)
            for j, (value, desc, kind) in enumerate(items):
                sel = off + j == self.popup_sel
                y = by - len(items) + j
                text = f" {value.ljust(12) if kind == 'cmd' else value} "
                seg = [(text, S["rev"] if sel else S["accent"]), (" " + desc, S["gray"])]
                self.put(y, 1, [(" " * (pw + 1), 0)], pw + 1)
                self.put(y, 2, seg, pw)

        for j, line in enumerate(bottom):
            self.put(by + j, 0, line, w)

        if cursor and (self.mode != "normal" or not self.perm):
            try:
                curses.curs_set(1)
                scr.move(by + cursor[0], min(w - 1, cursor[1]))
            except curses.error:
                pass
        else:
            try:
                curses.curs_set(0)
            except curses.error:
                pass
        scr.noutrefresh()
        curses.doupdate()

    def welcome(self, top, width):
        art = [
            [("✻ ", S["accent"]), ("Harness", S["bold"]), (f"  v{__version__}", S["gray"])],
            [],
            [("  model   ", S["gray"]), (self.cfg.model, 0)],
            [("  server  ", S["gray"]), (self.cfg.base_url, 0)],
            [("  key     ", S["gray"]), (mask_key(self.cfg.api_key), 0)],
            [("  mode    ", S["gray"]), (f"{self.perms.mode} — {MODE_HELP[self.perms.mode]}", 0)],
            [("  context ", S["gray"]), (f"{fmt_k(self.cfg.context_window)} tokens per request", 0)],
            [],
            [("  Ask for a change, a fix, or an explanation of this codebase.", 0)],
            [("  /help lists commands and keys.", S["gray"])],
            [],
            [("  Each step takes 20s–3min while the model thinks; the status line shows progress.", S["gray"])],
        ]
        mem = prompts.find_memory_files(self.agent.ctx.cwd)
        if mem:
            art.append([("  project instructions: " + ", ".join(os.path.basename(m) for m in mem), S["gray"])])
        for j, line in enumerate(art):
            self.put(top + j, 2, line, width)

    # ---------------------------------------------------------------- main loop

    def main(self, stdscr):
        self.stdscr = stdscr
        try:
            curses.set_escdelay(25)
        except AttributeError:
            pass
        curses.raw()
        curses.nonl()
        stdscr.keypad(True)
        init_styles(getattr(self.cfg, "tui_mouse", True))
        sys.stdout.write("\033[?2004h")  # bracketed paste
        sys.stdout.flush()
        self.width = stdscr.getmaxyx()[1]
        self.trans_h = 10
        threading.Thread(target=self.health_loop, daemon=True).start()
        if not self.cfg.api_key:
            self.notify("No API key is set. Paste one below (or press esc and use /login later).", "warn")
            self.start_login()
        stdscr.timeout(80)
        try:
            while self.running:
                if not self.busy and self.queue and not self.perm:
                    self.send(self.queue.pop(0))
                self.draw()
                try:
                    k = stdscr.get_wch()
                except curses.error:
                    continue
                keys = [k]
                stdscr.nodelay(True)
                while len(keys) < 100000:
                    try:
                        keys.append(stdscr.get_wch())
                    except curses.error:
                        break
                stdscr.timeout(80)
                self.handle_keys(keys)
        finally:
            sys.stdout.write("\033[?2004l")
            sys.stdout.flush()
            if self.busy:
                self.interrupt()
                for _ in range(30):
                    if not self.busy:
                        break
                    time.sleep(0.1)
            self.resolve_permission("n", "")
            self._save_history()
            self.save_session()


def run_tui(cfg, client, perms, agent, session, initial=None, loaded=None):
    locale.setlocale(locale.LC_ALL, "")
    os.environ.setdefault("ESCDELAY", "25")
    app = App(cfg, client, perms, agent, session)
    if loaded:
        app.load(loaded)
    if initial:
        app.send(initial)
    curses.wrapper(app.main)
    print(f"session saved: {app.session.id} (resume with hc -c)")
    return 0
