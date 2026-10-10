"""Where the agent's output goes. ConsoleView prints line by line; the TUI provides its own View."""

from . import ui


class View:
    """Interface the agent reports to. Methods may be called from a worker thread."""

    def model_start(self, sub=False): ...
    def reasoning(self, text, sub=False): ...
    def text(self, text, sub=False): ...
    def model_end(self, sub=False): ...
    def status(self, label): ...  # long-running activity; None clears it
    def tool_start(self, name, summary, sub=False): ...
    def tool_diff(self, path, old, new, sub=False): ...
    def tool_result(self, name, text, status="ok", secs=0.0, sub=False): ...  # status: ok | error | denied
    def file_changed(self, path, added, removed): ...
    def todos(self, todos): ...
    def notice(self, msg, level="info"): ...  # level: info | warn | error
    def turn_end(self, stats): ...
    def interrupted(self): ...


class ConsoleView(View):
    def __init__(self, cfg):
        self.cfg = cfg
        self.spinner = None
        self.md = None
        self.mode = None
        self.thought = 0

    def _spin(self, label):
        if self.spinner is None:
            self.spinner = ui.Spinner(label).start()
        else:
            self.spinner.label = label

    def _stop(self):
        if self.spinner is not None:
            self.spinner.stop()
            self.spinner = None

    def model_start(self, sub=False):
        self.md = ui.MarkdownStream()
        self.mode = None
        self.thought = 0
        self._spin("Researching" if sub else "Waiting for the model")

    def reasoning(self, text, sub=False):
        self.thought += 1
        if sub or not self.cfg.show_reasoning:
            if self.spinner:
                self.spinner.label = f"{'Researching' if sub else 'Thinking'} · {self.thought} tok"
            return
        if self.mode is None:
            self._stop()
            ui.write(ui.style("✻ ", "magenta"))
        self.mode = "reasoning"
        ui.write(ui.style(text, "gray", "italic"))

    def text(self, text, sub=False):
        if sub:
            return
        if self.mode != "text":
            self._stop()
            if self.mode == "reasoning":
                ui.write("\n\n")
            self.mode = "text"
        self.md.feed(text)

    def model_end(self, sub=False):
        self._stop()
        if self.mode == "text":
            self.md.flush()
        elif self.mode == "reasoning":
            ui.write("\n")
        self.mode = None

    def status(self, label):
        if label is None:
            self._stop()
        else:
            self._spin(label)

    def tool_start(self, name, summary, sub=False):
        self._stop()
        ui.out(("    " if sub else "") + ui.style("● ", "green") + ui.style(name, "bold") +
               ui.style(f"({summary})", "gray"))

    def tool_diff(self, path, old, new, sub=False):
        if not sub:
            self._stop()
            ui.print_diff(old, new, path)

    def tool_result(self, name, text, status="ok", secs=0.0, sub=False):
        if sub and status == "ok":
            return
        self._stop()
        if status == "error":
            ui.indent_block(text, color="red")
        elif status == "denied":
            ui.indent_block(text, color="yellow", max_lines=2)
        elif name == "TodoWrite":
            ui.indent_block(text, max_lines=50, color="cyan")
        elif name == "Task":
            ui.indent_block(text, max_lines=6)
        else:
            ui.indent_block(text + (f"\n({secs:.1f}s)" if secs > 3 else ""), max_lines=4)

    def notice(self, msg, level="info"):
        self._stop()
        {"info": ui.info, "warn": ui.warn, "error": ui.error}[level](msg)

    def turn_end(self, stats):
        speed = f"{stats['tps']:.1f} tok/s · " if stats["tps"] else ""
        pct = stats["context"] / stats["window"]
        line = (f"  ✻ {stats['tokens']} tok · {speed}{stats['secs']:.0f}s · "
                f"context {stats['context'] / 1000:.1f}k/{stats['window'] / 1000:.0f}k ({pct:.0%})")
        ui.out(ui.style(line, "yellow" if pct >= 0.5 else "gray"))

    def interrupted(self):
        self._stop()
        ui.out()
        ui.warn("⎿ Interrupted. What should I do instead?")
