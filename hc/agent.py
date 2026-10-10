"""The agent loop: call the model, run the tools it asks for, feed results back, repeat."""

import json
import os
import re
import time

from . import prompts
from .llm import Cancelled, LLMError
from .tools import READ_ONLY, ToolContext, ToolError, build_tools, format_todos, truncate
from .ui import diff_stats
from .view import ConsoleView


class ReasoningLoop(Exception):
    """Raised from the reasoning stream when the model starts repeating itself."""


class LoopDetector:
    """Flags reasoning whose latest 120 characters already occurred twice before (a repeating passage)."""

    PROBE, REPEATS, EVERY = 120, 3, 40

    def __init__(self):
        self.text = ""
        self.next_check = self.PROBE * self.REPEATS

    def feed(self, chunk):
        self.text += chunk
        if len(self.text) < self.next_check:
            return False
        self.next_check = len(self.text) + self.EVERY
        norm = re.sub(r"\s+", " ", self.text)
        tail = norm[-self.PROBE:]
        if len(tail) < self.PROBE:
            return False
        # Count overlapping occurrences: a one-sentence loop repeats with a period shorter than the probe.
        count, i = 0, norm.find(tail)
        while i >= 0:
            count += 1
            if count >= self.REPEATS:
                return True
            i = norm.find(tail, i + 1)
        return False


class Agent:
    def __init__(self, cfg, client, perms, cwd, subagent=False, messages=None, todos=None, view=None):
        self.cfg = cfg
        self.client = client
        self.perms = perms
        self.subagent = subagent
        self.view = view or ConsoleView(cfg)
        self.ctx = ToolContext(cfg, cwd)
        self.ctx.cancel = client.cancel_event
        self.ctx.todos = todos or []
        self.ctx.subagent_runner = None if subagent else self._run_subagent
        all_tools = build_tools(include_task=not subagent)
        if subagent:
            all_tools = {k: v for k, v in all_tools.items() if k in ("Read", "Glob", "Grep", "LS")}
        self.tools = all_tools
        self.messages = messages or []
        self.total_prompt = 0
        self.total_completion = 0
        self.last_tps = 0.0
        self.summaries = []  # every summary produced by compaction, oldest first
        self.last_request = None  # the latest user message, quoted verbatim after a compaction
        self._overhead_warned = False
        self._half_warned = False
        self._post_compact = 0  # conversation size right after the last compaction
        self._reset_usage()
        self.refresh_system_prompt()

    def _reset_usage(self):
        self._usage_prompt = 0  # prompt tokens of the latest request, as reported by the server
        self._usage_mark = 0  # number of messages that request contained

    # The system prompt is deliberately kept stable between requests so the server can reuse its KV cache
    # for the shared prefix. Rebuild it only on /clear, mode changes and compaction.
    def refresh_system_prompt(self):
        self.system = prompts.build_system_prompt(
            self.cfg, self.ctx.cwd, self.perms.mode, subagent=self.subagent)

    def active_tools(self):
        if self.perms.mode == "plan":
            return [t.schema() for n, t in self.tools.items() if n in READ_ONLY]
        return [t.schema() for t in self.tools.values()]

    def estimate_tokens(self):
        """Rough size of the next prompt from character counts (~3 chars per token; errs high)."""
        chars = len(self.system) + len(json.dumps(self.active_tools())) + len(json.dumps(self.messages))
        return chars // 3

    def context_tokens(self):
        """Size of the next prompt: the server's count for the last request plus an estimate for newer messages."""
        if self._usage_prompt and self._usage_mark <= len(self.messages):
            return self._usage_prompt + len(json.dumps(self.messages[self._usage_mark:])) // 3
        return self.estimate_tokens()

    def reset(self):
        self.messages = []
        self.ctx.todos = []
        self.ctx.read_files = {}
        self.summaries = []
        self.last_request = None
        self._post_compact = 0
        self._half_warned = False
        self._reset_usage()
        self.refresh_system_prompt()

    def cancel(self):
        """Interrupt the running turn (safe to call from another thread)."""
        self.client.cancel()

    # ------------------------------------------------------------ main loop

    def run(self, user_content):
        """Runs one user turn to completion. Returns the final assistant text, or None if interrupted."""
        if not self.subagent:
            self.client.reset_cancel()
        self.ctx.turn += 1
        self.last_request = user_content
        self.messages.append({"role": "user", "content": user_content})
        try:
            return self._loop()
        except LLMError:
            # Nothing happened yet: drop the message so the user can simply retry.
            if self.messages and self.messages[-1]["role"] == "user":
                self.messages.pop()
            else:
                self._repair_after_interrupt()
            raise
        except (KeyboardInterrupt, Cancelled):
            self._repair_after_interrupt()
            if self.subagent:
                raise
            self.view.status(None)
            self.view.interrupted()
            return None

    def _loop(self):
        overflow_retry = False
        stats = {"tokens": 0, "gen_time": 0.0, "t0": time.time()}
        for _ in range(self.cfg.max_steps):
            self._maybe_compact()
            try:
                resp = self._call_model()
            except LLMError as e:
                if not overflow_retry and any(w in str(e).lower() for w in ("context", "too long", "exceed")):
                    self.view.notice(f"Server rejected the request ({e}); summarizing and retrying…", "warn")
                    overflow_retry = True
                    self.compact()
                    continue
                raise
            stats["tokens"] += resp["completion_tokens"]
            if resp["tokens_per_sec"]:
                stats["gen_time"] += resp["completion_tokens"] / resp["tokens_per_sec"]
                self.last_tps = resp["tokens_per_sec"]
            msg = {"role": "assistant", "content": resp["content"] or ""}
            if resp["tool_calls"]:
                msg["tool_calls"] = [{"id": tc["id"], "type": "function",
                                      "function": {"name": tc["name"], "arguments": tc["arguments"] or "{}"}}
                                     for tc in resp["tool_calls"]]
            self.messages.append(msg)
            if not resp["tool_calls"]:
                if not self.subagent:
                    if resp["finish_reason"] == "length":
                        self.view.notice("(the model ran out of output tokens" +
                                         (" while still thinking" if not resp["content"].strip() else "") +
                                         ". Ask for a smaller step, or raise max_output_tokens.)", "warn")
                    elif not resp["content"].strip():
                        self.view.notice("(model returned an empty response)")
                    self._turn_stats(stats)
                return resp["content"]
            for tc in resp["tool_calls"]:
                result = self._execute(tc)
                self.messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
        self.view.notice(f"Stopped after {self.cfg.max_steps} model calls in one turn. Say 'continue' to keep "
                         "going.", "warn")
        return None

    def thinking_extra(self, budget):
        if budget is None or budget < 0:
            return {}
        return {"reasoning_budget_tokens": budget, "reasoning_budget_message": self.cfg.thinking_budget_message}

    def step_budget(self):
        """Thinking budget for the next call: more when planning a new request, less when continuing after tools."""
        first, later = self.cfg.thinking_budgets()
        if self.subagent:
            return later
        return first if self.messages and self.messages[-1]["role"] == "user" else later

    def _call_model(self):
        budget = self.step_budget()
        try:
            return self._call_model_once(self.thinking_extra(budget))
        except ReasoningLoop:
            # Retry the step once: a tight budget forces an answer, and a presence penalty discourages the loop.
            retry_budget = 384 if budget is None or budget < 0 else max(256, min(budget, 768) // 2)
            self.view.notice(f"↻ the model's reasoning started repeating itself; retrying this step with a "
                             f"{retry_budget}-token thinking budget", "warn")
            extra = self.thinking_extra(retry_budget)
            extra["presence_penalty"] = 1.5
            return self._call_model_once(extra, detect_loops=False)

    def _call_model_once(self, extra, detect_loops=True):
        sub = self.subagent
        prompt_est = self.context_tokens()
        max_tokens = max(1024, min(self.cfg.max_output_tokens, self.cfg.context_window - prompt_est - 256))
        sent = len(self.messages)
        detector = LoopDetector() if detect_loops and self.cfg.loop_detection else None

        def on_reasoning(t):
            self.view.reasoning(t, sub)
            if detector and detector.feed(t):
                raise ReasoningLoop()

        self.view.model_start(sub)
        try:
            resp = self.client.chat(
                [{"role": "system", "content": self.system}] + self.messages,
                tools=self.active_tools(), max_tokens=max_tokens, extra=extra,
                on_text=lambda t: self.view.text(t, sub), on_reasoning=on_reasoning)
        finally:
            self.view.model_end(sub)

        prompt_toks = resp["usage"].get("prompt_tokens", 0)
        self.total_prompt += prompt_toks
        self.total_completion += resp["completion_tokens"]
        if prompt_toks:
            self._usage_prompt, self._usage_mark = prompt_toks, sent
        return resp

    def _turn_stats(self, stats):
        tps = stats["tokens"] / stats["gen_time"] if stats["gen_time"] > 0.05 else 0
        self.view.turn_end({"tokens": stats["tokens"], "tps": tps, "secs": time.time() - stats["t0"],
                            "context": self.context_tokens(), "window": self.cfg.context_window})

    # ------------------------------------------------------------ tools

    def _execute(self, tc):
        name, sub, view = tc["name"], self.subagent, self.view
        tool = self.tools.get(name)
        if tool is None:
            view.tool_start(name, "", sub)
            msg = f"Error: unknown tool '{name}'. Available tools: {', '.join(self.tools)}"
            view.tool_result(name, msg, "error", sub=sub)
            return msg
        try:
            args = tool.coerce(json.loads(tc["arguments"] or "{}"))
        except (json.JSONDecodeError, ToolError) as e:
            view.tool_start(name, "", sub)
            msg = (f"Error: arguments were not valid JSON ({e}). Retry the call with valid JSON arguments."
                   if isinstance(e, json.JSONDecodeError) else f"Error: {e}")
            view.tool_result(name, msg, "error", sub=sub)
            return msg

        view.tool_start(name, tool.summary(args), sub)
        try:
            change = tool.prepare(self.ctx, args)
        except ToolError as e:
            view.tool_result(name, f"Error: {e}", "error", sub=sub)
            return f"Error: {e}"

        if not sub:
            if change is not None and self.perms.mode != "plan":
                path, old, new = change
                view.tool_diff(self.ctx.rel(path), old, new, sub)
            ok, denial = self.perms.check(tool, args)
            if not ok:
                view.tool_result(name, denial, "denied", sub=sub)
                return denial

        t0 = time.time()
        try:
            result = tool.run(self.ctx, args)
        except ToolError as e:
            view.tool_result(name, f"Error: {e}", "error", sub=sub)
            return f"Error: {e}"
        except Exception as e:  # tool bugs or OS errors shouldn't kill the session
            view.tool_result(name, f"Error: {type(e).__name__}: {e}", "error", sub=sub)
            return f"Error: {type(e).__name__}: {e}"

        view.tool_result(name, result, "ok", time.time() - t0, sub)
        if change is not None:
            path, old, new = change
            view.file_changed(self.ctx.rel(path), *diff_stats(old, new))
        if name == "TodoWrite":
            view.todos(self.ctx.todos)
        return truncate(result, self.cfg.max_tool_output)

    def _run_subagent(self, description, prompt):
        sub = Agent(self.cfg, self.client, self.perms, self.ctx.cwd, subagent=True, view=self.view)
        result = sub.run(prompt)
        self.total_prompt += sub.total_prompt
        self.total_completion += sub.total_completion
        return result.strip() if result and result.strip() else "(the sub-agent returned no report)"

    def undo_last_turn(self):
        """Restores the files changed in the most recent turn that changed files. Returns (restored, errors)."""
        ctx = self.ctx
        if not ctx.backups:
            return [], []
        last = max(t for t, _, _ in ctx.backups)
        restored, errors = [], []
        for _, path, original in reversed([b for b in ctx.backups if b[0] == last]):
            try:
                if original is None:
                    if os.path.exists(path):
                        os.remove(path)
                else:
                    with open(path, "w", encoding="utf-8", errors="surrogateescape") as f:
                        f.write(original)
            except OSError as e:
                errors.append(f"{path}: {e}")
                continue
            ctx.read_files.pop(path, None)
            restored.append(ctx.rel(path))
        ctx.backups = [b for b in ctx.backups if b[0] != last]
        return restored, errors

    # ------------------------------------------------------------ context management

    def _repair_after_interrupt(self):
        """Keep the transcript valid: every tool call needs a result, and partial output is preserved."""
        answered = {m.get("tool_call_id") for m in self.messages if m["role"] == "tool"}
        for m in reversed(self.messages):
            if m["role"] == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    if tc["id"] not in answered:
                        self.messages.append({"role": "tool", "tool_call_id": tc["id"],
                                              "content": "Interrupted by the user before this tool finished."})
                break
            if m["role"] == "user":
                break
        if self.messages and self.messages[-1]["role"] == "user":
            partial = self.client.last_partial.strip()
            self.messages.append({"role": "assistant",
                                  "content": (partial + "\n\n" if partial else "") + "[Interrupted by the user]"})

    def _maybe_compact(self):
        """Summarize before the window fills, so the next request (prompt + reasoning + answer) still fits."""
        window = self.cfg.context_window
        used = self.context_tokens()
        if not self.subagent and not self._half_warned and used >= window * 0.5:
            self._half_warned = True
            self.view.notice(f"(context {used / 1000:.1f}k/{window / 1000:.0f}k — the conversation will be "
                             f"summarized automatically at {self.cfg.compact_threshold:.0%})")
        if used < window * self.cfg.compact_threshold:
            return
        # Summarizing only shrinks the conversation, not the fixed system prompt + tool definitions, so skip it
        # when it can't help, and don't re-summarize until the conversation has grown again.
        conversation = len(json.dumps(self.messages)) // 3
        overhead = max(0, self.estimate_tokens() - conversation)
        if overhead >= window * self.cfg.compact_threshold or conversation < max(
                window * 0.2, self._post_compact + window * 0.25):
            if not self._overhead_warned and not self.subagent:
                self.view.notice(f"⚠ The system prompt and tools alone use ~{overhead / 1000:.1f}k tokens of a "
                                 f"{window / 1000:.0f}k window. Increase context_window.", "warn")
                self._overhead_warned = True
            return
        self.compact()

    def _transcript(self, budget_chars):
        parts = []
        for m in self.messages:
            role = m["role"]
            if role == "user":
                parts.append(f"USER: {m['content']}")
            elif role == "assistant":
                s = f"ASSISTANT: {m.get('content') or ''}"
                for tc in m.get("tool_calls") or []:
                    s += f"\n[tool call {tc['function']['name']}({tc['function']['arguments'][:500]})]"
                parts.append(s)
            elif role == "tool":
                parts.append(f"TOOL RESULT: {truncate(m['content'], 1500)}")
        text = "\n\n".join(parts)
        if len(text) > budget_chars:
            head = min(8000, budget_chars // 4)
            text = text[:head] + "\n\n[... middle of conversation omitted ...]\n\n" + text[-(budget_chars - head):]
        return text

    def _summarize(self, messages, tools, max_tokens):
        label = "Summarizing the conversation to free up context"
        self.view.status(label)
        count = {"n": 0}

        def on_reasoning(_):
            count["n"] += 1
            self.view.status(f"{label} · thinking {count['n']} tok")

        try:
            resp = self.client.chat(messages, tools=tools, max_tokens=max_tokens, on_reasoning=on_reasoning,
                                    extra=self.thinking_extra(self.cfg.thinking_budgets()[1]))
        finally:
            self.view.status(None)
        self.total_prompt += resp["usage"].get("prompt_tokens", 0)
        self.total_completion += resp["completion_tokens"]
        text = resp["content"].strip()
        if resp["tool_calls"] or len(text) < 200:
            return None
        return text

    def compact(self, instructions=None, manual=False):
        if not self.messages:
            return
        window = self.cfg.context_window
        before = self.context_tokens()
        request = prompts.COMPACT + (f"\n\nAdditional focus requested by the user: {instructions}"
                                     if instructions else "")
        summary = None
        room = window - before - 512
        if room >= 3072:
            # Append the request to the unchanged conversation: the server reuses its cached prefix, so only the
            # request itself has to be processed (re-reading ~30k tokens would take ~10 minutes on CPU).
            msgs = ([{"role": "system", "content": self.system}] + self.messages +
                    [{"role": "user", "content": request}])
            summary = self._summarize(msgs, self.active_tools(), min(8192, room))
        if summary is None:
            # Fallback: a condensed transcript in a fresh request, sized to leave room for thinking + summary.
            budget = max(4000, (window - 12000) * 3)
            msgs = [{"role": "system", "content": "You summarize coding-agent sessions accurately and thoroughly."},
                    {"role": "user", "content": f"<conversation>\n{self._transcript(budget)}\n</conversation>\n\n"
                                                f"{request}"}]
            summary = self._summarize(msgs, None, 8192)
        if summary is None:
            raise LLMError("the model did not produce a usable summary; try /compact again")

        self.summaries.append({"time": time.time(), "tokens_before": before, "summary": summary})
        parts = ["This session is being continued from a previous conversation that was summarized because the "
                 "context window was filling up. The summary below covers everything before this point.",
                 f"<summary>\n{summary}\n</summary>"]
        if self.last_request and not manual:
            parts.append(f"The user's most recent request, verbatim:\n<request>\n{truncate(self.last_request, 4000)}"
                         f"\n</request>")
        if self.ctx.todos:
            parts.append(f"Current todo list:\n{format_todos(self.ctx.todos)}")
        parts.append("File contents from before this point are no longer in your context: Read files again "
                     "before editing them.")
        if not manual:
            parts.append("Continue the task from where it left off without asking the user to repeat anything.")
        self.messages = [{"role": "user", "content": "\n\n".join(parts)}]
        if manual:
            self.messages.append({"role": "assistant", "content": "Understood. I have the summary and will continue "
                                                                  "from there."})
        self.ctx.read_files = {}
        self.refresh_system_prompt()
        self._reset_usage()
        self._post_compact = len(json.dumps(self.messages)) // 3
        self._half_warned = False
        self.view.notice(f"✻ Conversation summarized to keep going (~{before / 1000:.1f}k → "
                         f"~{self.context_tokens() / 1000:.1f}k tokens). /summary shows it.")
