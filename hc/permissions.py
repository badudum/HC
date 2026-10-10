"""Permission modes and interactive approval of tool calls."""

import re
import shlex

from . import ui

MODES = ("default", "acceptEdits", "plan", "bypass")

MODE_HELP = {
    "default": "ask before edits, shell commands and web requests",
    "acceptEdits": "auto-approve file edits; ask before shell commands",
    "plan": "read-only: the model explores and proposes a plan, no changes",
    "bypass": "approve everything without asking (careful!)",
}

# Commands considered read-only when they contain no shell metacharacters.
SAFE_SINGLE = {"ls", "pwd", "cat", "head", "tail", "wc", "which", "whoami", "date", "uname", "file", "stat",
               "tree", "du", "df", "rg", "grep", "find", "echo", "env", "printenv", "basename", "dirname",
               "realpath", "readlink", "sort", "uniq", "diff", "less", "nproc", "free", "id", "hostname"}
SAFE_TWO = {"git status", "git diff", "git log", "git show", "git branch", "git remote", "git blame",
            "git rev-parse", "git ls-files", "python --version", "python3 --version", "node --version",
            "npm ls", "pip list", "pip show", "cargo --version", "go version"}
_META = re.compile(r"[;&|`$<>(){}\n\\]")
_UNSAFE_ARGS = {"-exec", "-execdir", "-delete", "-ok", "-okdir", "-fprint", "-fprintf", "-fls"}
_TWO_WORD = {"git", "npm", "pnpm", "yarn", "cargo", "docker", "pip", "uv", "make", "go", "systemctl", "kubectl",
             "python", "python3", "npx", "bun", "deno", "podman", "sudo"}


def is_safe_command(cmd):
    cmd = cmd.strip()
    if not cmd or _META.search(cmd):
        return False
    try:
        toks = shlex.split(cmd)
    except ValueError:
        return False
    if not toks or any(t in _UNSAFE_ARGS or t.startswith("--output") for t in toks):
        return False
    return toks[0] in SAFE_SINGLE or " ".join(toks[:2]) in SAFE_TWO


def command_prefix(cmd):
    try:
        toks = shlex.split(cmd)
    except ValueError:
        toks = cmd.split()
    if not toks:
        return cmd
    if toks[0] in _TWO_WORD and len(toks) > 1 and (re.fullmatch(r"[a-z][\w-]*", toks[1]) or toks[1] == "-m"):
        # A subcommand (`git commit`, `npm test`, `python3 -m`), not a file argument (`python3 fib.py`).
        return " ".join(toks[:3] if toks[1] == "-m" and len(toks) > 2 else toks[:2])
    return toks[0]


class Permissions:
    def __init__(self, mode="default", allowed_commands=None):
        if mode not in MODES:
            raise ValueError(f"unknown permission mode {mode!r}; choose from {', '.join(MODES)}")
        self.mode = mode
        self.allowed_prefixes = list(allowed_commands or [])
        self.session_kinds = set()  # kinds approved with "always" this session
        self.prompter = console_prompter  # (tool, args, always_label) -> (choice, feedback)

    def _prefix_allowed(self, cmd):
        if _META.search(cmd):
            # Only allow chained commands if every segment is allowed.
            segments = [s.strip() for s in re.split(r"&&|\|\||;|\|", cmd) if s.strip()]
            if any(_META.search(s) for s in segments):
                return False
            return all(self._prefix_allowed(s) or is_safe_command(s) for s in segments)
        return any(cmd == p or cmd.startswith(p + " ") for p in self.allowed_prefixes)

    def check(self, tool, args):
        """Returns (allowed, message_for_model_if_denied)."""
        kind = tool.kind
        if kind in ("read", "meta") or self.mode == "bypass":
            return True, None
        if self.mode == "plan":
            return False, ("Plan mode is active: you may only read and explore. Do not modify anything; present "
                           "your plan to the user instead.")
        if kind == "edit" and (self.mode == "acceptEdits" or "edit" in self.session_kinds):
            return True, None
        if kind == "web" and "web" in self.session_kinds:
            return True, None
        if kind == "exec":
            cmd = args.get("command", "").strip()
            if is_safe_command(cmd) or self._prefix_allowed(cmd):
                return True, None
        return self._ask(tool, args)

    def _ask(self, tool, args):
        prefix = command_prefix(args.get("command", "")) if tool.kind == "exec" else None
        if tool.kind == "exec":
            always = f"don't ask again for `{prefix}` commands this session"
        elif tool.kind == "edit":
            always = "allow all edits this session"
        else:
            always = "allow all web requests this session"
        choice, feedback = self.prompter(tool, args, always)
        if choice == "y":
            return True, None
        if choice == "a":
            if tool.kind == "exec":
                self.allowed_prefixes.append(prefix)
            else:
                self.session_kinds.add(tool.kind)
            return True, None
        msg = "The user denied this tool call."
        return False, msg + (f" User feedback: {feedback}" if feedback else
                             " Do not retry it as-is; ask the user how to proceed if unsure.")


def console_prompter(tool, args, always):
    """Asks on stdin. Returns (choice, feedback) with choice in y | a | n."""
    prompt = (f"  {ui.style('Allow?', 'yellow', 'bold')} "
              f"[{ui.style('y', 'bold')}]es · [{ui.style('a', 'bold')}] yes, {always} · "
              f"[{ui.style('n', 'bold')}]o, tell the model what to do instead: ")
    while True:
        try:
            ans = input(prompt).strip().lower()
        except EOFError:
            return "n", "(no interactive user available)"
        if ans in ("y", "yes", ""):
            return "y", ""
        if ans in ("a", "always"):
            return "a", ""
        if ans in ("n", "no"):
            try:
                fb = input(f"  {ui.style('Feedback for the model (enter to skip):', 'gray')} ").strip()
            except EOFError:
                fb = ""
            return "n", fb
