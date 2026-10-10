"""Interactive REPL and command-line entry point."""

import argparse
import atexit
import getpass
import os
import re
import sys
import time

from . import __version__, prompts, ui
from .agent import Agent
from .config import (CREDENTIALS, DEFAULT_CONTEXT, HOME, THINKING, delete_credentials, load_config, mask_key,
                     save_credentials)
from .llm import Client, LLMError
from .permissions import MODE_HELP, MODES, Permissions
from .session import Session, list_sessions, load_session
from .tools import ToolError, format_todos, t_bash, t_ls, t_read

try:
    import readline
except ImportError:  # pragma: no cover
    readline = None

COMMANDS = {
    "/help": "show this help",
    "/clear": "start a fresh conversation",
    "/compact": "summarize the conversation to free context (optional: focus instructions)",
    "/context": "show how much of the context window is used",
    "/model": "show or switch the model; lists models on the server",
    "/mode": "show or switch permission mode: " + ", ".join(MODES),
    "/allow": "always allow bash commands starting with the given prefix this session",
    "/todos": "show the current todo list",
    "/undo": "revert the file changes made during the last turn",
    "/init": "analyze the project and write a HARNESS.md with instructions for the agent",
    "/resume": "resume an earlier session from this directory",
    "/summary": "show the latest conversation summary (made when the context fills up)",
    "/thinking": "toggle streaming the model's reasoning",
    "/think": "how long the model may think per step: low, medium, high, max",
    "/login": "set or replace the API key (stored in ~/.harness/credentials.json)",
    "/stats": "show token usage for this session",
    "/config": "show the active configuration",
    "/exit": "quit (or ctrl-d)",
}

MENTION = re.compile(r"(?<![\w@])@([\w./~-][^\s]*)")


def _rl(s):
    """Wrap ANSI sequences so readline computes the prompt width correctly."""
    return re.sub(r"(\033\[[0-9;]*m)", "\001\\1\002", s)


def _ago(ts):
    d = time.time() - ts
    for unit, secs in (("d", 86400), ("h", 3600), ("m", 60)):
        if d >= secs:
            return f"{int(d // secs)}{unit} ago"
    return "just now"


class App:
    def __init__(self, cfg, client, perms, agent, session):
        self.cfg = cfg
        self.client = client
        self.perms = perms
        self.agent = agent
        self.session = session
        self.pending = []  # context to prepend to the next user message (shell output, undo notes)
        self.last_sigint = 0.0

    # ------------------------------------------------------------ setup

    def setup_readline(self):
        if not readline:
            return
        os.makedirs(HOME, exist_ok=True)
        hist = os.path.join(HOME, "history")
        try:
            readline.read_history_file(hist)
        except OSError:
            pass
        readline.set_history_length(2000)
        atexit.register(lambda: _safe(readline.write_history_file, hist))
        readline.set_completer_delims(" \t\n")
        readline.set_completer(self._complete)
        readline.parse_and_bind("tab: complete")

    def _complete(self, text, state):
        if text.startswith("/"):
            opts = [c for c in COMMANDS if c.startswith(text)]
        elif text.startswith("@"):
            partial = text[1:]
            d, base = os.path.split(partial)
            full = os.path.join(self.agent.ctx.cwd, os.path.expanduser(d) or ".")
            try:
                names = os.listdir(full)
            except OSError:
                names = []
            opts = []
            for n in sorted(names):
                if n.startswith(base) and (base.startswith(".") or not n.startswith(".")):
                    p = os.path.join(d, n)
                    opts.append("@" + p + ("/" if os.path.isdir(os.path.join(full, n)) else ""))
        else:
            opts = []
        return opts[state] if state < len(opts) else None

    def banner(self):
        ui.out(ui.style("✻ Harness", "magenta", "bold") + ui.style(f" v{__version__}", "gray"))
        ui.out(ui.style("  model  ", "gray") + f"{self.cfg.model} " + ui.style(f"@ {self.cfg.base_url}", "gray"))
        ui.out(ui.style("  cwd    ", "gray") + self.agent.ctx.cwd)
        ui.out(ui.style("  mode   ", "gray") + f"{self.perms.mode} " + ui.style(f"({MODE_HELP[self.perms.mode]})", "gray"))
        mem = prompts.find_memory_files(self.agent.ctx.cwd)
        if mem:
            ui.out(ui.style("  memory ", "gray") + ", ".join(mem))
        ui.info("  /help for commands · @path to attach a file · !cmd to run shell · "
                "ctrl-c to interrupt · ctrl-d to exit")
        ui.out()

    def check_server(self):
        if not self.client.health():
            ui.warn(f"⚠ {self.cfg.base_url} is not responding. The server may be restarting or down; requests "
                    f"will be retried automatically.\n")
            return
        if not self.cfg.api_key:
            ui.warn("⚠ No API key is set. The server requires one.")
            if sys.stdin.isatty() and not login(self.cfg, self.client):
                ui.info("  Run /login when you have a key.\n")
            return
        try:
            models = self.client.list_models()
        except LLMError as e:
            ui.warn(f"⚠ {e}\n")
            return
        if models and self.cfg.model not in models:
            ui.warn(f"⚠ model '{self.cfg.model}' is not on the server. Available: {', '.join(models)}\n"
                    f"  Switch with /model <name> or --model.\n")

    # ------------------------------------------------------------ input

    def prompt_str(self):
        tag = "" if self.perms.mode == "default" else ui.style(f"{self.perms.mode} ", "yellow")
        return _rl(tag + ui.style("❯ ", "bold", "magenta"))

    def read_input(self):
        line = input(self.prompt_str())
        if line.strip() == '"""':
            lines = []
            while True:
                nxt = input(_rl(ui.style("… ", "gray")))
                if nxt.strip() == '"""':
                    return "\n".join(lines)
                lines.append(nxt)
        lines = [line]
        while lines[-1].endswith("\\"):
            lines[-1] = lines[-1][:-1]
            lines.append(input(_rl(ui.style("… ", "gray"))))
        return "\n".join(lines)

    def expand_mentions(self, text):
        attachments = []
        ctx = self.agent.ctx
        for m in MENTION.finditer(text):
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
                ui.warn(f"  could not attach {raw}: {e}")
                continue
            ui.info(f"  ⎿ attached {ctx.rel(path)}")
        if attachments:
            text += "\n\nAttached by the user:\n" + "\n\n".join(attachments)
        return text

    # ------------------------------------------------------------ turns

    def submit(self, text):
        content = self.expand_mentions(text)
        if self.pending:
            content = "\n\n".join(self.pending) + "\n\n" + content
            self.pending = []
        ui.out()
        try:
            self.agent.run(content)
        except LLMError as e:
            ui.error(f"⎿ API error: {e}")
        self.save()
        ui.out()

    def save(self):
        self.session.save(self.agent.messages, self.cfg.model, self.agent.ctx.todos, self.agent.summaries)

    def shell(self, cmd):
        try:
            output = t_bash(self.agent.ctx, cmd)
        except KeyboardInterrupt:
            ui.warn("  interrupted")
            return
        ui.indent_block(output, max_lines=40, color="gray")
        self.pending.append(f"<user-shell-command>{cmd}</user-shell-command>\n"
                            f"<user-shell-output>\n{output[: self.cfg.max_tool_output]}\n</user-shell-output>")

    # ------------------------------------------------------------ slash commands

    def command(self, line):
        """Returns False when the app should exit."""
        name, _, arg = line.strip().partition(" ")
        arg = arg.strip()
        a = self.agent
        if name in ("/exit", "/quit"):
            return False
        elif name == "/help":
            for c, h in COMMANDS.items():
                ui.out(f"  {ui.style(c.ljust(10), 'bold')} {h}")
            ui.info('\n  Multi-line input: end a line with \\ or wrap text in """ lines.\n'
                    "  @path attaches a file, !cmd runs a shell command and shares its output with the model.")
        elif name == "/clear":
            a.reset()
            self.session = Session(a.ctx.cwd)
            ui.info("  ⎿ started a new conversation")
        elif name == "/compact":
            try:
                a.compact(arg or None, manual=True)
                self.save()
            except LLMError as e:
                ui.error(f"  compaction failed: {e}")
        elif name == "/context":
            used = a.context_tokens()
            pct = 100 * used / self.cfg.context_window
            bar_w = 30
            filled = min(bar_w, int(bar_w * pct / 100))
            ui.out(f"  {ui.style('█' * filled, 'magenta')}{ui.style('░' * (bar_w - filled), 'gray')} "
                   f"~{used / 1000:.1f}k / {self.cfg.context_window / 1000:.0f}k tokens ({pct:.0f}%)")
            ui.info(f"  {len(a.messages)} messages · system prompt ~{len(a.system) // 3} tok · "
                    f"summarized automatically at {self.cfg.compact_threshold:.0%} · "
                    f"{len(a.summaries)} summaries so far")
        elif name == "/model":
            if arg:
                self.cfg.model = arg
                a.refresh_system_prompt()
                ui.info(f"  ⎿ model set to {arg}")
            else:
                ui.out(f"  current: {ui.style(self.cfg.model, 'bold')}")
                try:
                    for m in self.client.list_models():
                        ui.out(f"  {'●' if m == self.cfg.model else '○'} {m}")
                except LLMError as e:
                    ui.warn(f"  {e}")
        elif name == "/mode":
            if arg and arg not in MODES:
                ui.warn(f"  unknown mode; choose from {', '.join(MODES)}")
                return True
            self.perms.mode = arg or MODES[(MODES.index(self.perms.mode) + 1) % len(MODES)]
            a.refresh_system_prompt()
            ui.info(f"  ⎿ mode: {self.perms.mode} ({MODE_HELP[self.perms.mode]})")
        elif name == "/allow":
            if not arg:
                ui.out("  allowed prefixes: " + (", ".join(self.perms.allowed_prefixes) or "(none)"))
            else:
                self.perms.allowed_prefixes.append(arg)
                ui.info(f"  ⎿ bash commands starting with `{arg}` will run without asking")
        elif name == "/todos":
            ui.out(format_todos(a.ctx.todos))
        elif name == "/undo":
            self.undo()
        elif name == "/init":
            self.submit(prompts.INIT)
        elif name == "/resume":
            self.resume(arg)
        elif name == "/summary":
            if not a.summaries:
                ui.info("  ⎿ no summaries yet; one is made automatically when the context fills up, or with /compact")
            else:
                last = a.summaries[-1]
                ui.info(f"  summary {len(a.summaries)} of {len(a.summaries)} · made at "
                        f"{time.strftime('%H:%M', time.localtime(last['time']))} from "
                        f"~{last['tokens_before'] / 1000:.1f}k tokens")
                md = ui.MarkdownStream()
                md.feed(last["summary"] + "\n")
                md.flush()
        elif name == "/thinking":
            self.cfg.show_reasoning = not self.cfg.show_reasoning
            ui.info(f"  ⎿ reasoning is now {'streamed' if self.cfg.show_reasoning else 'hidden (token count only)'}")
        elif name == "/think":
            ui.info("  ⎿ " + set_thinking(self.cfg, arg))
        elif name == "/login":
            login(self.cfg, self.client)
        elif name == "/stats":
            ui.out(f"  prompt tokens: {a.total_prompt:,}  ·  completion tokens: {a.total_completion:,}")
        elif name == "/config":
            for k, v in self.cfg.as_dict().items():
                ui.out(f"  {k.ljust(18)} {v}")
        else:
            ui.warn(f"  unknown command {name}; try /help")
        return True

    def undo(self):
        restored, errors = self.agent.undo_last_turn()
        for e in errors:
            ui.error(f"  could not restore {e}")
        if not restored:
            ui.info("  ⎿ nothing to undo")
            return
        ui.info("  ⎿ reverted: " + ", ".join(restored))
        self.pending.append(undo_note(restored))

    def resume(self, sid=""):
        sessions = list_sessions(self.agent.ctx.cwd)
        if sid:
            sessions = [s for s in sessions if s["id"].startswith(sid)]
        if not sessions:
            ui.info("  ⎿ no saved sessions for this directory")
            return
        if len(sessions) > 1:
            for i, s in enumerate(sessions[:15], 1):
                ui.out(f"  {ui.style(str(i).rjust(2), 'bold')}. {s.get('title') or '(untitled)'} "
                       + ui.style(f"· {_ago(s.get('updated', 0))} · {len(s['messages'])} msgs", "gray"))
            try:
                choice = input("  resume which? ").strip()
            except (EOFError, KeyboardInterrupt):
                ui.out()
                return
            if not choice.isdigit() or not 1 <= int(choice) <= min(15, len(sessions)):
                return
            data = sessions[int(choice) - 1]
        else:
            data = sessions[0]
        self.load(data)

    def load(self, data):
        self.agent.reset()
        self.agent.messages = data["messages"]
        self.agent.ctx.todos = data.get("todos") or []
        self.agent.summaries = data.get("summaries") or []
        self.session = Session(self.agent.ctx.cwd, sid=data["id"], title=data.get("title"))
        ui.info(f"  ⎿ resumed \"{data.get('title', '')}\" ({len(data['messages'])} messages)")
        for m in reversed(data["messages"]):
            if m["role"] == "assistant" and m.get("content"):
                ui.indent_block(m["content"], prefix="  last reply: ", cont="              ", max_lines=3)
                break

    # ------------------------------------------------------------ main loop

    def run(self, initial=None):
        self.setup_readline()
        self.banner()
        self.check_server()
        if initial:
            self.submit(initial)
        while True:
            try:
                line = self.read_input()
            except KeyboardInterrupt:
                now = time.time()
                ui.out()
                if now - self.last_sigint < 1.5:
                    break
                self.last_sigint = now
                ui.info("  (press ctrl-c again or ctrl-d to exit)")
                continue
            except EOFError:
                ui.out()
                break
            if not line.strip():
                continue
            if line.startswith("/"):
                if not self.command(line):
                    break
            elif line.startswith("!"):
                self.shell(line[1:].strip())
            else:
                self.submit(line)
        ui.info(f"  session saved: {self.session.id} (resume with hc --continue)")
        return 0


def thinking_label(cfg):
    first, later = cfg.thinking_budgets()
    return f"{cfg.thinking} (unlimited)" if first < 0 else f"{cfg.thinking} ({first}/{later} tokens)"


def set_thinking(cfg, level):
    if not level:
        return f"thinking: {thinking_label(cfg)} — choose from {', '.join(THINKING)}"
    if level not in THINKING:
        return f"unknown level {level!r}; choose from {', '.join(THINKING)}"
    cfg.thinking = level
    return f"thinking: {thinking_label(cfg)}"


def detect_context(cfg, client):
    """Use the server's real per-slot context unless the user configured one."""
    if not cfg.context_window:
        cfg.context_window = client.server_context(timeout=8) or DEFAULT_CONTEXT


def undo_note(restored):
    return (f"[The user reverted your file changes from your previous turn ({', '.join(restored)}). "
            "Read these files again before editing them.]")


def login(cfg, client):
    """Prompt for an API key, verify it against the server, and store it. Returns True on success."""
    ui.info(f"  Paste your API key for {cfg.base_url} (input is hidden; ask the server owner for one).")
    try:
        key = getpass.getpass("  API key: ").strip()
    except (EOFError, KeyboardInterrupt):
        ui.out()
        return False
    if not key:
        return False
    old, cfg.api_key = cfg.api_key, key
    try:
        client.list_models()
    except LLMError as e:
        if e.status == 401:
            ui.error("  ✗ the server rejected that key")
            cfg.api_key = old
            return False
        ui.warn(f"  could not verify the key right now ({e}); saving it anyway")
    save_credentials(key)
    ui.out(ui.style("  ✓ ", "green") + f"key {mask_key(key)} saved to {CREDENTIALS} (readable only by you)")
    return True


def _safe(fn, *a):
    try:
        fn(*a)
    except OSError:
        pass


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["login"]:
        cfg = load_config()
        return 0 if login(cfg, Client(cfg)) else 1
    if argv[:1] == ["logout"]:
        ui.info("  removed stored API key" if delete_credentials() else "  no stored API key")
        return 0

    ap = argparse.ArgumentParser(prog="hc", description="Claude Code-style coding agent for local LLMs "
                                 "(any OpenAI-compatible server: Ollama, llama.cpp, vLLM, LM Studio).")
    ap.add_argument("prompt", nargs="*", help="initial prompt")
    ap.add_argument("-p", "--print", action="store_true", help="run the prompt non-interactively, print, and exit")
    ap.add_argument("-m", "--model", help="model name")
    ap.add_argument("--base-url", help="OpenAI-compatible base URL, e.g. http://arch:11434/v1")
    ap.add_argument("--api-key", help="API key (prefer `hc login` or HARNESS_API_KEY: flags are visible in `ps`)")
    ap.add_argument("--context", type=int, dest="context_window", help="context window size in tokens")
    ap.add_argument("--mode", choices=MODES, help="permission mode")
    ap.add_argument("--think", choices=list(THINKING), dest="thinking",
                    help="thinking budget per step (default: medium)")
    ap.add_argument("--yolo", action="store_true", help="same as --mode bypass")
    ap.add_argument("-c", "--continue", dest="cont", action="store_true", help="continue the latest session here")
    ap.add_argument("-r", "--resume", nargs="?", const="", metavar="ID", help="pick a session to resume")
    ap.add_argument("--classic", action="store_true", help="line-by-line REPL instead of the full-screen app")
    ap.add_argument("--version", action="version", version=f"hc {__version__}")
    ap.epilog = "subcommands: `hc login` stores an API key, `hc logout` removes it"
    args = ap.parse_args(argv)

    cwd = os.getcwd()
    cfg = load_config({
        "model": args.model, "base_url": args.base_url, "api_key": args.api_key,
        "context_window": args.context_window, "permission_mode": "bypass" if args.yolo else args.mode,
        "thinking": args.thinking,
    }, cwd=cwd)
    try:
        perms = Permissions(cfg.permission_mode, cfg.allowed_commands)
    except ValueError as e:
        ui.error(str(e))
        return 2
    client = Client(cfg)
    detect_context(cfg, client)
    agent = Agent(cfg, client, perms, cwd)
    prompt = " ".join(args.prompt)

    use_tui = (not args.print and not args.classic and sys.stdin.isatty() and sys.stdout.isatty()
               and os.environ.get("TERM", "dumb") != "dumb")
    if use_tui:
        from .tui import run_tui
        loaded = None
        sessions = list_sessions(cwd) if (args.cont or args.resume is not None) else []
        if args.cont and sessions:
            loaded = sessions[0]
        elif args.resume:
            loaded = next((s for s in sessions if s["id"].startswith(args.resume)), None)
        elif args.resume is not None and sessions:
            loaded = sessions[0] if len(sessions) == 1 else None
            if loaded is None:
                print("Several sessions here; starting fresh. Use /resume inside the app to pick one.")
        return run_tui(cfg, client, perms, agent, Session(cwd), initial=prompt or None, loaded=loaded)

    app = App(cfg, client, perms, agent, Session(cwd))

    if args.cont:
        sessions = list_sessions(cwd)
        if sessions:
            app.load(sessions[0])
        else:
            ui.info("no previous session in this directory; starting fresh")
    elif args.resume is not None:
        if args.resume:
            try:
                app.load(load_session(args.resume))
            except OSError:
                app.resume(args.resume)
        else:
            app.resume()

    if args.print:
        if not prompt and not sys.stdin.isatty():
            prompt = sys.stdin.read()
        if not prompt.strip():
            ui.error("no prompt given")
            return 2
        try:
            result = agent.run(app.expand_mentions(prompt))
        except LLMError as e:
            ui.error(f"API error: {e}")
            return 1
        app.save()
        return 0 if result is not None else 1
    return app.run(prompt or None)
