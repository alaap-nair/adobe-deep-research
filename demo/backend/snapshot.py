"""
snapshot.py -- Freeze the live /ask result into snapshot.json.

After seeding Maya's episodes and confirming the backend works, run this to
capture the live-generated Generalized/Personalized answers into
demo/web/public/snapshot.json. The deployed (Vercel) app then replays this exact
real run with no backend connected.

Usage:  python -m demo.backend.snapshot
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from demo.backend.server import SNAPSHOT_PATH, ask


async def main() -> None:
    result = await ask()
    if result.get("_note"):
        print(f"WARNING: /ask returned a fallback ({result['_note']}). "
              "Seed Maya and check the LLM credential before snapshotting.")
        return

    snap = json.loads(Path(SNAPSHOT_PATH).read_text(encoding="utf-8"))
    snap["ask"] = result
    snap["meta"]["generated"] = "frozen from live /ask"
    Path(SNAPSHOT_PATH).write_text(
        json.dumps(snap, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote live /ask result into {SNAPSHOT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
