"""Terminal output: colors, spinner, streaming markdown, diffs."""

import difflib
import os
import re
import shutil
import sys
import threading
import time

COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")

CODES = {
    "bold": "1", "dim": "2", "italic": "3", "underline": "4",
    "red": "31", "green": "32", "yellow": "33", "blue": "34",
    "magenta": "35", "cyan": "36", "gray": "90",
}


def style(text, *names):
    if not COLOR or not names or not text:
        return text
    return f"\033[{';'.join(CODES[n] for n in names)}m{text}\033[0m"


def write(s):
    sys.stdout.write(s)
    sys.stdout.flush()


def out(s=""):
    write(s + "\n")


def info(msg):
    out(style(msg, "gray"))


def warn(msg):
    out(style(msg, "yellow"))


def error(msg):
    out(style(msg, "red"))


def width():
    return shutil.get_terminal_size((100, 24)).columns


def indent_block(text, prefix="  ⎿  ", cont="     ", max_lines=4, color="gray"):
    """Print the first few lines of a tool result, Claude Code style."""
    lines = text.rstrip("\n").split("\n") if text.strip() else ["(no output)"]
    w = max(width() - len(cont) - 1, 20)
    for i, line in enumerate(lines[:max_lines]):
        if len(line) > w:
            line = line[: w - 1] + "…"
        out(style((prefix if i == 0 else cont) + line, color))
    if len(lines) > max_lines:
        out(style(f"{cont}… +{len(lines) - max_lines} lines", "gray"))


class Spinner:
    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, label="Thinking"):
        self.label = label
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if not COLOR:
            return self
        self._t0 = time.time()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        i = 0
        while not self._stop.wait(0.1):
            secs = time.time() - self._t0
            write(f"\r\033[2K{style(self.FRAMES[i % len(self.FRAMES)], 'magenta')} "
                  f"{style(f'{self.label}… ({secs:.0f}s · ctrl-c to interrupt)', 'gray')}")
            i += 1

    def stop(self):
        if self._thread:
            self._stop.set()
            self._thread.join()
            self._thread = None
            write("\r\033[2K")


class MarkdownStream:
    """Renders markdown incrementally as tokens arrive.

    Line kind (code fence, header, bullet) is decided once the start of a line is
    known; inline `code` and **bold** are toggled character by character, so text
    appears as soon as it streams in.
    """

    def __init__(self):
        self.pending = ""
        self.line_start = True
        self.kind = "text"
        self.in_fence = False
        self.code = False
        self.bold = False
        self.wrote = False
        self.at_newline = True

    def _styles(self):
        if self.kind == "fence":
            return ("gray",)
        if self.kind == "code":
            return ("cyan",)
        if self.kind == "header":
            return ("bold",)
        if self.code:
            return ("cyan",)
        if self.bold:
            return ("bold",)
        return ()

    def _emit(self, s, styles=()):
        if s:
            write(style(s, *styles))
            self.wrote = True
            self.at_newline = s.endswith("\n")

    def feed(self, text):
        # Skip leading blank lines at the very start of a response.
        if not self.wrote and not self.pending:
            text = text.lstrip("\n")
        self.pending += text
        self._drain(final=False)

    def flush(self):
        self._drain(final=True)
        if self.wrote and not self.at_newline:
            write("\n")
            self.at_newline = True

    def _drain(self, final):
        p, i = self.pending, 0
        while i < len(p):
            if self.line_start:
                rest = p[i:]
                head = rest.lstrip(" ")
                if "\n" not in rest and len(head) < 3 and not final:
                    break  # not enough characters yet to classify the line
                self.line_start = False
                if head.startswith("```"):
                    self.kind = "fence"
                elif self.in_fence:
                    self.kind = "code"
                elif (m := re.match(r"#{1,6}\s+", head)):
                    self.kind = "header"
                    i += len(rest) - len(head) + len(m.group(0))
                else:
                    self.kind = "text"
                    if (m := re.match(r"( *)[-*] ", rest)):
                        self._emit(m.group(1) + "• ")
                        i += len(m.group(0))
                continue
            ch = p[i]
            if ch == "\n":
                self._emit("\n")
                if self.kind == "fence":
                    self.in_fence = not self.in_fence
                self.code = self.bold = False
                self.line_start = True
                i += 1
                continue
            if self.kind in ("text", "header"):
                if ch == "`":
                    self.code = not self.code
                    i += 1
                    continue
                if ch == "*" and not self.code:
                    if i + 1 >= len(p) and not final:
                        break  # might be the first half of **
                    if i + 1 < len(p) and p[i + 1] == "*":
                        self.bold = not self.bold
                        i += 2
                        continue
            self._emit(ch, self._styles())
            i += 1
        self.pending = p[i:]


def print_diff(old, new, path, max_lines=80):
    lines = list(difflib.unified_diff(
        old.splitlines(), new.splitlines(), fromfile=path, tofile=path, lineterm="", n=2))[2:]
    if not lines:
        info("     (no changes)")
        return
    for line in lines[:max_lines]:
        if line.startswith("+"):
            out("     " + style(line, "green"))
        elif line.startswith("-"):
            out("     " + style(line, "red"))
        elif line.startswith("@@"):
            out("     " + style(line, "cyan"))
        else:
            out("     " + style(line, "gray"))
    if len(lines) > max_lines:
        info(f"     … +{len(lines) - max_lines} more diff lines")


def diff_stats(old, new):
    add = rem = 0
    for line in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=0):
        if line.startswith("+") and not line.startswith("+++"):
            add += 1
        elif line.startswith("-") and not line.startswith("---"):
            rem += 1
    return add, rem
