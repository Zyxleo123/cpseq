"""Read `ANTHROPIC_API_KEY` from a dotenv file into the process environment.

Why this exists rather than a shell one-liner: extracting a quoted value from
a dotenv file in bash needs nested quote escaping that is easy to get wrong and
hard to read, and any mistake tends to end with the secret on a command line or
in a log. Here the value never leaves the process -- it is read, put in
`os.environ`, and the SDK picks it up from there.

An already-set environment variable always wins, so a job that was given the
key properly is never silently overridden by a stale file.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DOTENV = "/zfsauton2/home/yixiz/Proteina-Complexa/.env"
KEYS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")


def load_dotenv(path: str | os.PathLike | None = None, keys=KEYS) -> list[str]:
    """Set `keys` from the dotenv file if absent. Returns the names it set."""
    src = Path(path or os.environ.get("LP2CP_DOTENV", DEFAULT_DOTENV))
    if not src.exists():
        return []
    loaded: list[str] = []
    try:
        text = src.read_text()
    except OSError:
        return []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        if name.startswith("export "):
            name = name[len("export "):].strip()
        if name not in keys or os.environ.get(name):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if value:
            os.environ[name] = value
            loaded.append(name)
    return loaded


# A file whose entire contents are the key, with no VAR= prefix. The OpenAI
# key for this project arrived that way, and a dotenv parser silently skips
# such a file (no "=" on the line) -- which reads as "no key available" rather
# than as a format problem, so it is handled explicitly.
RAW_KEY_FILES = {
    "OPENAI_API_KEY": (
        "/zfsauton2/home/yixiz/lp2cp-agent/agent/openai_api",
    ),
}
_KEY_PREFIX = {"OPENAI_API_KEY": "sk-", "ANTHROPIC_API_KEY": "sk-"}


def load_raw_key_files(keys=KEYS) -> list[str]:
    """Set `keys` from single-value key files. Returns the names it set."""
    loaded: list[str] = []
    for name in keys:
        if os.environ.get(name):
            continue
        for cand in RAW_KEY_FILES.get(name, ()):
            try:
                value = Path(cand).read_text().strip()
            except OSError:
                continue
            # Checked rather than trusted: a path that turned out to hold a
            # dotenv file, a note or an empty string would otherwise be
            # exported as a key and produce 401s that look like a dead account.
            if value and value.startswith(_KEY_PREFIX.get(name, "")) and \
                    "\n" not in value and len(value) > 20:
                os.environ[name] = value
                loaded.append(name)
                break
    return loaded


def require_api_key(path: str | os.PathLike | None = None,
                    key: str = "ANTHROPIC_API_KEY") -> bool:
    """True if a key is available. Never prints or returns the value."""
    load_dotenv(path)
    load_raw_key_files()
    return bool(os.environ.get(key))


if __name__ == "__main__":
    for _k in KEYS:
        ok = require_api_key(key=_k)
        print(f"{_k} available: {ok} ({len(os.environ.get(_k, ''))} chars)")
