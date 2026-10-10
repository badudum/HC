"""Saving and resuming conversations under ~/.harness/sessions."""

import json
import os
import time
import uuid

from .config import HOME

SESSION_DIR = os.path.join(HOME, "sessions")


class Session:
    def __init__(self, cwd, sid=None, title=None):
        self.cwd = cwd
        self.title = title
        self.id = sid or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.path = os.path.join(SESSION_DIR, f"{self.id}.json")

    def save(self, messages, model, todos, summaries=None):
        if not messages:
            return
        if not self.title:
            first = next((m["content"] for m in messages if m["role"] == "user" and isinstance(m["content"], str)), "")
            self.title = first.strip().split("\n")[0][:100]
        data = {"id": self.id, "cwd": self.cwd, "model": model, "updated": time.time(),
                "title": self.title, "messages": messages, "todos": todos, "summaries": summaries or []}
        os.makedirs(SESSION_DIR, mode=0o700, exist_ok=True)
        tmp = self.path + ".tmp"
        with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
            json.dump(data, f)
        os.replace(tmp, self.path)


def list_sessions(cwd=None):
    if not os.path.isdir(SESSION_DIR):
        return []
    out = []
    for name in os.listdir(SESSION_DIR):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(SESSION_DIR, name)) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if cwd is None or data.get("cwd") == cwd:
            out.append(data)
    return sorted(out, key=lambda d: d.get("updated", 0), reverse=True)


def load_session(sid):
    with open(os.path.join(SESSION_DIR, f"{sid}.json")) as f:
        return json.load(f)
