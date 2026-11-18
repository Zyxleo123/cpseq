#!/usr/bin/env python3
"""Print the OpenAI spend ledgers in a campaigns directory, and the total.

A separate file rather than a heredoc inside the job script: the job script
already contains one heredoc for the key check, and nesting a second confuses
enough shell parsers to be worth avoiding.
"""

from __future__ import annotations

import glob
import json
import os
import sys


def main() -> int:
    camp = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CAMPAIGNS_DIR", "")
    if not camp:
        print("usage: sum_openai_spend.py <campaigns_dir>")
        return 0
    total = 0.0
    for f in sorted(glob.glob(os.path.join(camp, "cost__*.json"))):
        name = os.path.basename(f)
        if "gpt" not in name:
            continue
        try:
            d = json.loads(open(f).read())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  {name:64s} UNREADABLE: {exc}")
            continue
        total += d.get("spent_usd", 0.0)
        print(f"  {name:64s} ${d.get('spent_usd', 0):.4f} "
              f"over {d.get('calls', 0)} calls"
              + (f"  [{d['stopped_reason']}]" if d.get("stopped_reason") else ""))
    print(f"  {'TOTAL (openai only)':64s} ${total:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
