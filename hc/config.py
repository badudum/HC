"""Configuration: defaults <- ~/.harness/config.json <- ./.harness/config.json <- credentials <- env <- CLI flags."""

import json
import os
import stat
from dataclasses import asdict, dataclass, field, fields

HOME = os.path.expanduser(os.environ.get("HARNESS_HOME", "~/.harness"))
CREDENTIALS = os.path.join(HOME, "credentials.json")


@dataclass
class Config:
    # Public endpoint of the llama.cpp server on `arch` (Tailscale Funnel). Requires an API key.
    base_url: str = "https://arch.tailb96076.ts.net/v1"
    api_key: str = ""
    model: str = "qwen3.6-35b-a3b-abliterated"
    # Per-request budget on the server: prompt + reasoning + answer must fit. 0 = ask the server (/props) at
    # startup; falls back to DEFAULT_CONTEXT if the server can't be reached.
    context_window: int = 0
    # Covers the model's reasoning AND its answer; thinking alone is often 300-3,000 tokens.
    max_output_tokens: int = 16384
    # Qwen3.6's recommended sampling for precise coding in thinking mode. The server's defaults (temp 1.0,
    # presence_penalty 1.5) are tuned for general chat; a presence penalty also punishes the repetition code
    # needs (identifiers, indentation). None = use the server's default for that setting.
    temperature: float = 0.6
    presence_penalty: float = 0.0
    # Summarize the conversation when the prompt reaches this fraction of the window, leaving room for the
    # summary itself (reasoning + summary text) and for the next answer.
    compact_threshold: float = 0.70
    permission_mode: str = "default"  # default | acceptEdits | plan | bypass
    bash_timeout: int = 120
    max_tool_output: int = 12000  # characters returned to the model per tool call
    # Seconds to wait for each read from the server. Reading a long prompt can take ~10 minutes before
    # the first token arrives.
    request_timeout: int = 1800
    max_steps: int = 80  # model calls per user turn
    # How much the model may think per step: low | medium | high | max (see THINKING). Past the budget the server
    # injects thinking_budget_message and closes the thinking block, so the model acts on what it has.
    thinking: str = "medium"
    thinking_budget_message: str = "\n\nOkay, I've thought enough. Time to act.\n"
    # Abort and retry a step whose reasoning starts repeating itself.
    loop_detection: bool = True
    show_reasoning: bool = False  # show the model's chain of thought (toggle with ctrl-t / /thinking)
    tui_mouse: bool = True  # mouse-wheel scrolling in the full-screen app (hold shift to select text)
    allowed_commands: list = field(default_factory=list)  # bash prefixes allowed without asking

    def thinking_budgets(self):
        """(first step of a turn, later steps) in tokens; -1 = unlimited."""
        return THINKING.get(self.thinking, THINKING["medium"])

    def as_dict(self):
        d = asdict(self)
        d["api_key"] = mask_key(self.api_key)
        return d


DEFAULT_CONTEXT = 65536

# Thinking budgets per effort level: the first step of a turn plans; later steps (after tool results) only need to
# decide the next action, so they get less.
THINKING = {
    "low": (512, 256),
    "medium": (1536, 768),
    "high": (4096, 2048),
    "max": (-1, -1),
}


def mask_key(key):
    if not key:
        return "(not set — run `hc login`)"
    return key[:5] + "…" + key[-4:] if len(key) > 12 else "***"


ENV_MAP = {
    "HARNESS_BASE_URL": ("base_url", str),
    "HARNESS_MODEL": ("model", str),
    "HARNESS_CONTEXT": ("context_window", int),
    "HARNESS_MODE": ("permission_mode", str),
}
# Checked in order; LLM_API_KEY is the name the server README uses.
KEY_ENV = ("HARNESS_API_KEY", "LLM_API_KEY")


def _merge_file(cfg, path):
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        return
    except (OSError, json.JSONDecodeError) as e:
        print(f"warning: ignoring {path}: {e}")
        return
    names = {f.name for f in fields(Config)}
    for k, v in data.items():
        if k in names:
            setattr(cfg, k, v)


def load_credentials():
    try:
        with open(CREDENTIALS) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    try:
        if os.stat(CREDENTIALS).st_mode & (stat.S_IRWXG | stat.S_IRWXO):
            os.chmod(CREDENTIALS, 0o600)
    except OSError:
        pass
    return data.get("api_key") or None


def save_credentials(api_key):
    os.makedirs(HOME, mode=0o700, exist_ok=True)
    fd = os.open(CREDENTIALS + ".tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump({"api_key": api_key}, f)
    os.replace(CREDENTIALS + ".tmp", CREDENTIALS)


def delete_credentials():
    try:
        os.remove(CREDENTIALS)
        return True
    except FileNotFoundError:
        return False


def load_config(overrides=None, cwd=None):
    cfg = Config()
    _merge_file(cfg, os.path.join(HOME, "config.json"))
    _merge_file(cfg, os.path.join(cwd or os.getcwd(), ".harness", "config.json"))
    cfg.api_key = load_credentials() or cfg.api_key
    for env, (name, typ) in ENV_MAP.items():
        if os.environ.get(env):
            setattr(cfg, name, typ(os.environ[env]))
    for env in KEY_ENV:
        if os.environ.get(env):
            cfg.api_key = os.environ[env].strip()
            break
    for k, v in (overrides or {}).items():
        if v is not None:
            setattr(cfg, k, v)
    return cfg
