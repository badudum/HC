"""Minimal streaming client for OpenAI-compatible /chat/completions (Ollama, llama.cpp, vLLM, LM Studio...)."""

import http.client
import json
import re
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

from . import ui


class Cancelled(BaseException):
    """Raised inside the agent's thread when the user interrupts. BaseException so generic handlers don't eat it."""


class LLMError(Exception):
    def __init__(self, msg, status=None):
        super().__init__(msg)
        self.status = status


class ThinkFilter:
    """Splits <think>...</think> blocks out of streamed content (for models that inline their reasoning)."""

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self):
        self.inside = False
        self.buf = ""

    def feed(self, text):
        self.buf += text
        parts = []
        while self.buf:
            tag = self.CLOSE if self.inside else self.OPEN
            kind = "reasoning" if self.inside else "content"
            idx = self.buf.find(tag)
            if idx >= 0:
                if idx:
                    parts.append((kind, self.buf[:idx]))
                self.buf = self.buf[idx + len(tag):]
                self.inside = not self.inside
                continue
            # Hold back a trailing partial tag such as "<thi".
            keep = 0
            for k in range(min(len(tag) - 1, len(self.buf)), 0, -1):
                if tag.startswith(self.buf[-k:]):
                    keep = k
                    break
            emit = self.buf[: len(self.buf) - keep]
            if emit:
                parts.append((kind, emit))
            self.buf = self.buf[len(self.buf) - keep:]
            break
        return parts

    def flush(self):
        parts = [("reasoning" if self.inside else "content", self.buf)] if self.buf else []
        self.buf = ""
        return parts


# Fallbacks for servers whose chat template doesn't convert tool calls into structured `tool_calls`.
_TOOL_CALL_JSON = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
_QWEN_FUNC = re.compile(r"(?:<tool_call>\s*)?<function=([\w.-]+)>(.*?)</function>(?:\s*</tool_call>)?", re.S)
_QWEN_PARAM = re.compile(r"<parameter=([\w.-]+)>\n?(.*?)\n?</parameter>", re.S)


def parse_text_tool_calls(content):
    """Returns (calls, content_without_calls)."""
    calls = []
    for m in _TOOL_CALL_JSON.finditer(content):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("name"):
            args = obj.get("arguments", obj.get("parameters", {}))
            calls.append({"name": obj["name"], "arguments": args if isinstance(args, str) else json.dumps(args)})
    if calls:
        return calls, _TOOL_CALL_JSON.sub("", content).strip()
    for m in _QWEN_FUNC.finditer(content):
        args = {p.group(1): p.group(2) for p in _QWEN_PARAM.finditer(m.group(2))}
        calls.append({"name": m.group(1), "arguments": json.dumps(args)})
    if calls:
        return calls, _QWEN_FUNC.sub("", content).strip()
    return [], content


class Client:
    def __init__(self, cfg):
        self.cfg = cfg
        self._stream_options = True
        self.last_partial = ""
        self.cancel_event = threading.Event()  # shared with tools so Bash can be interrupted too
        self._conn = None
        self.notify = ui.warn  # where retry messages go; the TUI replaces this

    # ---------------------------------------------------------------- cancellation

    def cancel(self):
        """Abort the current request from another thread, even while waiting for the first byte."""
        self.cancel_event.set()
        conn = self._conn
        if conn is not None and conn.sock is not None:
            try:
                conn.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def reset_cancel(self):
        self.cancel_event.clear()

    def _check_cancel(self):
        if self.cancel_event.is_set():
            raise Cancelled()

    # ---------------------------------------------------------------- simple requests

    def _url(self, path):
        return self.cfg.base_url.rstrip("/") + path

    def _headers(self):
        h = {"Content-Type": "application/json", "User-Agent": "harness-hc"}
        if self.cfg.api_key:
            h["Authorization"] = f"Bearer {self.cfg.api_key}"
        return h

    def _root(self):
        url = self.cfg.base_url.rstrip("/")
        return url[:-3] if url.endswith("/v1") else url

    def health(self, timeout=10):
        """Unauthenticated liveness check (llama.cpp /health). Returns True/False."""
        try:
            with urllib.request.urlopen(self._root() + "/health", timeout=timeout) as r:
                return r.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def server_context(self, timeout=15):
        """Per-slot context size reported by llama.cpp (/props), or None."""
        req = urllib.request.Request(self._root() + "/props", headers=self._headers())
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                n = json.load(r).get("default_generation_settings", {}).get("n_ctx")
        except (urllib.error.URLError, OSError, json.JSONDecodeError, AttributeError):
            return None
        return n if isinstance(n, int) and n > 0 else None

    def list_models(self, timeout=15):
        req = urllib.request.Request(self._url("/models"), headers=self._headers())
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.load(r)
        except urllib.error.HTTPError as e:
            raise self._status_error(e.code, e.read(), e.reason) from e
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
            raise LLMError(f"cannot reach {self.cfg.base_url}: {e}") from e
        return sorted(m.get("id", "?") for m in data.get("data", []))

    def _status_error(self, code, body, reason=""):
        detail = body.decode(errors="replace")[:1000].strip() if body else ""
        if code == 401:
            msg = ("the server rejected the API key (HTTP 401). Run `hc login` (or /login) to set a valid key."
                   if self.cfg.api_key else
                   "the server requires an API key (HTTP 401). Run `hc login` or set HARNESS_API_KEY.")
            return LLMError(msg, status=401)
        if code in (502, 503, 504):
            return LLMError(f"HTTP {code}: the model server is not running or is restarting", status=code)
        return LLMError(f"HTTP {code} from server: {detail or reason}", status=code)

    # ---------------------------------------------------------------- streaming request

    # Connection failures and 5xx usually mean the server is restarting (e.g. after a key was added).
    RETRY_DELAYS = (3, 8, 20, 45)

    def _open(self, body):
        """POSTs to /chat/completions with retries. Returns (connection, response)."""
        url = urllib.parse.urlsplit(self._url("/chat/completions"))
        cls = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
        path = url.path + (f"?{url.query}" if url.query else "")
        data = json.dumps(body).encode()
        for attempt in range(len(self.RETRY_DELAYS) + 1):
            self._check_cancel()
            conn = cls(url.hostname, url.port, timeout=self.cfg.request_timeout)
            self._conn = conn
            try:
                conn.request("POST", path, body=data, headers=self._headers())
                resp = conn.getresponse()
                if resp.status < 400:
                    return conn, resp
                err = self._status_error(resp.status, resp.read(), resp.reason)
                retry = resp.status >= 500 or resp.status == 429
            except (OSError, http.client.HTTPException) as e:
                self._check_cancel()
                err = LLMError(f"cannot reach {self.cfg.base_url}: {e}")
                retry = True
            conn.close()
            self._conn = None
            if not retry or attempt == len(self.RETRY_DELAYS):
                raise err
            delay = self.RETRY_DELAYS[attempt]
            self.notify(f"  {err} — retrying in {delay}s ({attempt + 1}/{len(self.RETRY_DELAYS)})")
            if self.cancel_event.wait(delay):
                raise Cancelled()

    def chat(self, messages, tools=None, on_text=None, on_reasoning=None, max_tokens=None, extra=None):
        """Streams a completion. Returns dict(content, reasoning, tool_calls, usage, finish_reason, ...).

        On KeyboardInterrupt the partial text is left in self.last_partial and the interrupt re-raised.
        """
        body = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": True,
            "max_tokens": max_tokens or self.cfg.max_output_tokens,
        }
        if self.cfg.temperature is not None:
            body["temperature"] = self.cfg.temperature
        if self.cfg.presence_penalty is not None:
            body["presence_penalty"] = self.cfg.presence_penalty
        if tools:
            body["tools"] = tools
        body.update(extra or {})
        if self._stream_options:
            body["stream_options"] = {"include_usage": True}

        t0 = time.time()
        self.last_partial = ""
        try:
            conn, resp = self._open(body)
        except LLMError as e:
            if e.status == 400 and "stream_options" in str(e) and self._stream_options:
                self._stream_options = False
                body.pop("stream_options")
                conn, resp = self._open(body)
            else:
                raise

        content, reasoning = [], []
        calls = {}
        usage, finish = None, None
        first_token = None
        think = ThinkFilter()
        hidden = {"on": False}

        def emit(kind, text):
            nonlocal first_token
            if not text:
                return
            first_token = first_token or time.time()
            if kind == "content":
                content.append(text)
                self.last_partial += text
                if on_text and not hidden["on"]:
                    # Stop echoing once the model starts writing a tool call as plain text.
                    tail = self.last_partial[-(len(text) + 12):]
                    cut = min((i for i in (tail.find("<tool_call>"), tail.find("<function=")) if i >= 0),
                              default=-1)
                    if cut >= 0:
                        hidden["on"] = True
                        text = text[: max(0, len(text) - (len(tail) - cut))]
                    if text:
                        on_text(text)
            else:
                reasoning.append(text)
                if on_reasoning:
                    on_reasoning(text)

        try:
            for raw in resp:
                self._check_cancel()
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if chunk.get("error"):
                    err = chunk["error"]
                    raise LLMError(err.get("message", str(err)) if isinstance(err, dict) else str(err))
                usage = chunk.get("usage") or usage
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    r = delta.get("reasoning_content") or delta.get("reasoning")
                    if r:
                        emit("reasoning", r)
                    if delta.get("content"):
                        for kind, text in think.feed(delta["content"]):
                            emit(kind, text)
                    for tc in delta.get("tool_calls") or []:
                        first_token = first_token or time.time()
                        idx = tc.get("index", len(calls))
                        entry = calls.setdefault(idx, {"id": None, "name": "", "arguments": ""})
                        if tc.get("id"):
                            entry["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name") and not entry["name"]:
                            entry["name"] = fn["name"]
                        args = fn.get("arguments")
                        if isinstance(args, dict):
                            entry["arguments"] = json.dumps(args)
                        elif args:
                            entry["arguments"] += args
                    finish = choice.get("finish_reason") or finish
            for kind, text in think.flush():
                emit(kind, text)
        except (OSError, http.client.HTTPException) as e:
            self._check_cancel()
            raise LLMError(f"connection to the server was lost mid-response ({e}). The server may be "
                           "restarting; try again in a moment.") from e
        finally:
            self._conn = None
            conn.close()
        self._check_cancel()

        text = "".join(content)
        tool_calls = [calls[k] for k in sorted(calls)]
        if not tool_calls and ("<tool_call>" in text or "<function=" in text):
            tool_calls, text = parse_text_tool_calls(text)
        for tc in tool_calls:
            tc["id"] = tc.get("id") or f"call_{uuid.uuid4().hex[:12]}"

        end = time.time()
        completion = (usage or {}).get("completion_tokens") or max(1, (len(text) + len("".join(reasoning))) // 4)
        gen_time = end - (first_token or end)
        return {
            "content": text,
            "reasoning": "".join(reasoning),
            "reasoning_chunks": len(reasoning),
            "tool_calls": tool_calls,
            "usage": usage or {},
            "finish_reason": finish,
            "completion_tokens": completion,
            "elapsed": end - t0,
            "tokens_per_sec": completion / gen_time if gen_time > 0.05 else 0.0,
        }
