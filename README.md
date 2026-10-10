# Harness (`hc`)

A Claude Code-style coding agent in your terminal, powered by the self-hosted model on `arch`. It reads and edits
files, runs commands, plans with a todo list, and asks your permission before changing anything.

By default it talks to the public, API-key-protected endpoint described in the server README (`~/llm/README.md`
on `arch`). It works with any OpenAI-compatible server.

---

## Requirements

| | |
|---|---|
| **Python** | 3.10 or newer. Only the standard library is used, so there's nothing to `pip install`. |
| **OS** | Linux or macOS. Uses `bash`, plus Python's built-in `curses` for the app. |
| **Terminal** | Any modern terminal with UTF-8 and 256 colors (kitty, alacritty, gnome-terminal, iTerm2, tmux…). It needs to be at least 50×12, and 100+ columns to show the sidebar. |
| **API key** | Required by the server. Ask the server owner for one (on `arch`: `~/llm/add-key.sh <name>`). |
| **Network** | HTTPS access to `https://arch.tailb96076.ts.net`. This is public through Tailscale Funnel, so you don't need to be on the tailnet. |
| **Optional** | `rg` (ripgrep) for faster searching; there's a pure-Python fallback. |

## Install and log in

```bash
ln -s "$PWD/bin/hc" ~/.local/bin/hc   # put hc on your PATH (run from this directory)
hc login                             # paste the key; it's checked against the server, then saved
cd ~/some/project && hc
```

### Where the API key comes from

The first source found wins:

1. `--api-key sk-...`. Avoid this one: command-line flags are visible to other users in `ps`.
2. `HARNESS_API_KEY` or `LLM_API_KEY` environment variable.
3. `~/.harness/credentials.json`, written by `hc login` / `/login` with permissions `600`. Remove it with `hc logout`.
4. `"api_key"` in a config file. Not recommended, and never in a project's `.harness/config.json`, which could get
   committed.

The key is never printed: `/config` shows it masked. A `401` error means the key is missing or wrong; run
`hc login` again.

---

## Things to know about this model and server

The server runs **Qwen3.6-35B-A3B (abliterated)** on llama.cpp, on old hardware. The harness is tuned for
it, but expect this:

- **It's slow.** The model thinks before every answer at about 15–20 tokens/s. Left alone it thinks for 300–3,000+
  tokens per step and often repeats itself, so the harness limits it (see "Thinking budget" below). With the
  default limits a step takes roughly 15–90 seconds. The spinner shows the thinking token count; Ctrl-T shows the
  reasoning itself.
- **Long prompts take a long time to read** (about 100 tokens/s). The harness never changes earlier messages
  and keeps the system prompt fixed, so the server only reads what's new each step. Things that force a full
  re-read, which can take several minutes on a long conversation:
  - `/mode`, `/model` and `/clear`
  - restarting a session with `hc -c`
  - another user's requests pushing yours out of the server's cache
- **Limited context per request:** currently 64k tokens per slot, covering prompt, reasoning and answer together.
  The harness reads the real size from the server (`/props`) at startup, so it adapts when the server's settings
  change. See below for what happens when it fills up.
- **Only 2 requests run at once** across all users. More requests queue, and a `Task` sub-agent uses the second
  slot.
- **Server restarts** (for example, after a key is added) show up as connection errors or 5xx responses. The harness
  retries automatically after 3, 8, 20 and 45 seconds.
- **It's uncensored ("abliterated") and can be confidently wrong.** The harness's only safeguard is its permission
  prompts. Read the diffs before approving them, and keep `default` mode for anything that matters.
- **Sampling:** the harness sends Qwen3.6's recommendation for precise coding, `temperature 0.6` and
  `presence_penalty 0`, which overrides the server's general-chat defaults (`1.0` / `1.5`). `top_p 0.95` and
  `top_k 20` come from the server. Set `"temperature"` / `"presence_penalty"` to `null` in the config to use the
  server's values instead.

## Thinking budget

The model is a "thinking" model: before every step it writes private reasoning, which is never shown in the
answer or sent back to it. Unchecked, that reasoning is long and repetitive. The harness controls it in three ways:

- **A thinking budget per step,** enforced by the server (`reasoning_budget_tokens`). When the budget runs out, the
  server injects "Okay, I've thought enough. Time to act." and closes the thinking, so the model acts on what it
  has. The first step of each request gets a bigger budget (it has to plan); steps after a tool result get a
  smaller one (they only decide the next action). Sub-agents and summaries use the smaller one.

  | `/think` level | first step | later steps |
  |---|---|---|
  | `low` | 512 | 256 |
  | `medium` (default) | 1,536 | 768 |
  | `high` | 4,096 | 2,048 |
  | `max` | unlimited | unlimited |

  Use `high` for hard bugs or design questions and `low` for simple edits. Change it with `/think`,
  `hc --think`, or `"thinking"` in the config.
- **Loop detection.** If the reasoning starts repeating a passage, the harness stops that step and retries it once
  with a small budget and a repetition penalty (`presence_penalty 1.5`). You'll see a "↻ retrying" note. Turn it
  off with `"loop_detection": false`.
- **The system prompt** asks the model to keep its reasoning short and not re-check settled points.

There's no repetition penalty on normal steps, on purpose (the harness sends `presence_penalty 0`, overriding the
server's default of 1.5). Penalties like DRY and `presence_penalty` punish *any*
repeated text, and code edits have to repeat code exactly (an Edit's `old_string` and `new_string` are mostly
identical), so they would corrupt edits.

Measured on the previous model (Qwen3-30B-A3B-Thinking) with the same small task (write `fib.py`, run it, check the output): 5m35s and 4,981 tokens
before, 2m21s and 1,998 tokens with the default `medium` budget, with the same correct result.

## Long sessions: automatic summaries

The harness keeps track of how full the context window is, using the server's own token counts. Each turn ends
with a line like `context 12.4k/25k (50%)`, shown in yellow once it passes 50%.

- **At 50%** you get a notice.
- **At 70%**, before the next model call, the harness has the model write a structured summary:
  - your requests, quoted
  - key decisions
  - every file touched and what changed
  - errors and fixes
  - commands and their results
  - your feedback
  - pending tasks
  - current work and the next step

  The summary request is appended to the existing conversation, so the server reuses its cache and doesn't
  re-read everything.
- The conversation is then replaced with the summary, your latest request quoted word for word, and the current
  todo list. The model continues the task without asking you to repeat anything, and re-reads files before
  editing them.
- `/summary` shows the latest summary, and `/compact [focus]` makes one on demand. Summaries are saved with the
  session, so `hc -c` resumes with them.

Leaving 30% of the window free gives the model room to think and write the summary. If you get "ran out of
output tokens" warnings, ask for smaller steps.

---

## Usage

`hc` opens a full-screen terminal app, similar to Claude Code or opencode:

```
 ✻ hc   ~/projects/api                                         qwen3.6-35b-a3b-abliterated
 ❯ fix the failing login test                                   │ Session
                                                                │ fix the failing login test
 ✻ Thought for 48s · 812 tok  (ctrl-t to show)                  │
 ● Grep(def login)                                              │ Context
   ⎿  src/auth.py                                               │ ██████░░░░░░░░░░░░  31%
 ● Edit(src/auth.py)                                            │ 7.6k / 24.6k tokens
     -    if user.password == password:                         │
     +    if check_password(user, password):                    │ Todos 1/2
 ⠧ Thinking · 342 tok · 17.8 tok/s · 31s      esc to interrupt  │ ☒ Find the bug
╭─ default ──────────────────────────────────────────────────╮  │ ◐ Fix and run tests
│ ❯ Message the agent…                                       │  │ Modified files
╰────────────────────────────────────────────────────────────╯  │ src/auth.py +1 -1
```

- **Transcript:** your messages, the model's thinking (collapsed; press Ctrl-T to show it), a card for each tool
  call with its output, colored diffs, and the answer rendered as markdown.
- **Sidebar** (on terminals at least 100 columns wide; Ctrl-B toggles it): context usage, server status, todos,
  modified files and token usage.
- **Permission prompts** take the place of the input box: `1`/`y` yes, `2`/`a` always, `3`/`n` no. Choosing no lets
  you type what the model should do instead.
- **While the model works**, the status line shows what it's doing and for how long. You can keep typing:
  messages are queued and sent when it finishes.

| Key | |
|---|---|
| Enter | send |
| Ctrl-J or Alt-Enter | new line (pasting multi-line text also works) |
| Tab | complete `/commands` and `@paths` |
| ↑ ↓ | move between lines, or through history |
| Esc | interrupt the model or a running command. This works even while the server is still reading the prompt. |
| Shift-Tab | cycle permission mode |
| Ctrl-T / Ctrl-O | show thinking / expand tool output |
| PgUp/PgDn, mouse wheel, End | scroll the transcript; End jumps to the bottom |
| Ctrl-C twice, Ctrl-D, `/exit` | quit; the session is saved |

Hold Shift while dragging to select text with the mouse (the app captures the wheel), or set `"tui_mouse": false`.

Other ways to run it:

```bash
hc "explain how auth works"  # open the app with a first prompt
hc -c                        # continue the last session in this directory
hc -r [ID]                   # resume a specific session
hc -p "add a --verbose flag" # one-shot: no app, prints the result and exits (stdin works too)
hc --classic                 # plain line-by-line REPL instead of the full-screen app
hc --mode plan               # start read-only; --yolo = never ask
```

| Input | What it does |
|---|---|
| `@path/to/file` | attach a file (or directory listing) |
| `!command` | run a shell command yourself; its output goes to the model with your next message |

| Command | |
|---|---|
| `/help` | keys and commands |
| `/login` | set or replace the API key |
| `/context` | how full the context window is |
| `/compact [focus]` | summarize now |
| `/summary` | show the latest summary |
| `/think [level]` | thinking budget: `low`, `medium`, `high`, `max` |
| `/mode [m]` | permission mode, see below |
| `/allow <prefix>` | let bash commands starting with `<prefix>` run without asking |
| `/undo` | revert all file changes from the last turn |
| `/todos` | the model's task list |
| `/init` | have the model write `HARNESS.md` describing the project |
| `/resume [N]`, `/clear` | list or open saved sessions, or start fresh |
| `/model`, `/stats`, `/config` | model, token usage, active settings |

**Permission modes:**
- `default`: asks before edits, shell commands and web fetches. Read-only commands like `ls` and `git status`
  are allowed.
- `acceptEdits`: edits are auto-approved.
- `plan`: read-only; the model proposes a plan.
- `bypass` (`--yolo`): never asks.

At a prompt, `a` approves that command prefix (or all edits) for the session, and `n` lets you tell the model what
to do instead.

**Project instructions:** `HARNESS.md`, `AGENTS.md` and `CLAUDE.md` in the current directory and its parents,
plus `~/.harness/HARNESS.md`, are added to the system prompt. Keep them short, because every token is re-read
whenever the cache is cold.

## Configuration

Later sources override earlier ones: defaults → `~/.harness/config.json` → `./.harness/config.json` →
credentials file → environment (`HARNESS_BASE_URL`, `HARNESS_MODEL`, `HARNESS_CONTEXT`, `HARNESS_MODE`,
`HARNESS_API_KEY`/`LLM_API_KEY`) → CLI flags.

| Key | Default | Notes |
|---|---|---|
| `base_url` | `https://arch.tailb96076.ts.net/v1` | any OpenAI-compatible `/v1` URL |
| `model` | `qwen3.6-35b-a3b-abliterated` | the server only hosts this model |
| `context_window` | `0` | `0` = ask the server at startup (falls back to 65,536); set a number to override |
| `thinking` | `medium` | thinking budget per step: `low`, `medium`, `high`, `max` |
| `thinking_budget_message` | "Okay, I've thought enough. Time to act." | injected when the budget runs out |
| `loop_detection` | `true` | stop and retry steps whose reasoning repeats itself |
| `max_output_tokens` | `16384` | reasoning + answer; automatically capped to the space left in the window |
| `compact_threshold` | `0.70` | when to summarize |
| `temperature` | `0.6` | `null` = server default (1.0) |
| `presence_penalty` | `0` | `null` = server default (1.5); penalties corrupt code edits |
| `show_reasoning` | `false` | show the chain of thought expanded (Ctrl-T toggles) |
| `tui_mouse` | `true` | mouse-wheel scrolling in the app |
| `request_timeout` | `1800` | seconds to wait for each read from the server; reading a long prompt can take ~10 minutes |
| `bash_timeout` | `120` | per command, max 600 |
| `allowed_commands` | `[]` | e.g. `["pytest", "npm test", "make"]` |
| `permission_mode` | `default` | |

## How it works

```
hc/tui.py          the full-screen app (curses): transcript, sidebar, input editor, permission prompts
hc/view.py         the interface the agent reports through (console version for --classic and -p)
hc/cli.py          entry point, classic REPL, login, sessions
hc/agent.py        loop: model → tool calls → results → model…; token accounting, summaries, interrupts
hc/llm.py          streaming /chat/completions client: auth, retries, reasoning_content, tool-call assembly
hc/tools.py        Read, Write, Edit, Bash, Glob, Grep, LS, WebFetch, TodoWrite, Task
hc/permissions.py  permission modes and approval prompts
hc/prompts.py      system prompt, summary prompt, /init prompt
hc/ui.py           spinner, streaming markdown, diffs
```

The tools have safeguards that help a small model:
- Edit requires a prior Read and refuses if the file changed on disk since.
- Edit tolerates trailing-whitespace differences and strips copied line-number prefixes.
- Arguments of the wrong type are converted.
- Tool calls written as `<tool_call>` text are still parsed.
- The model's reasoning is never sent back in the history, as Qwen's chat template expects.

Sessions are stored in `~/.harness/sessions/` (mode 600), and input history in `~/.harness/history`.
