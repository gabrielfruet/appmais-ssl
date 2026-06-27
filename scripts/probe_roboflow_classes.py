"""Probe each Roboflow Universe dataset and print its class list.

Reads ROBOFLOW_API_KEY from the project .env (or env var) and prints, for each
`(workspace, project)` pair given on the CLI, the project's class names plus a
handful of useful metadata (version, image count, split sizes). This is a
one-shot dry-run to inform the class-mapping table before we download anything.

Usage:
    uv run python scripts/probe_roboflow_classes.py WORKSPACE/PROJECT ...
    uv run python scripts/probe_roboflow_classes.py -            # read from stdin
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def _load_api_key() -> str:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not key:
        sys.exit("ROBOFLOW_API_KEY missing from .env")
    return key


def _parse_pairs(args: list[str]) -> list[tuple[str, str]]:
    if args == ["-"]:
        pairs: list[tuple[str, str]] = []
        for line in sys.stdin:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ws, _, proj = line.partition("/")
            if not proj:
                sys.exit(f"bad line (expected WORKSPACE/PROJECT): {line!r}")
            pairs.append((ws.strip(), proj.strip()))
        return pairs
    pairs = []
    for arg in args:
        ws, _, proj = arg.partition("/")
        if not proj:
            sys.exit(f"bad arg (expected WORKSPACE/PROJECT): {arg!r}")
        pairs.append((ws.strip(), proj.strip()))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pairs", nargs="+", help="WORKSPACE/PROJECT pairs (use - to read from stdin)"
    )
    parser.add_argument(
        "--json", action="store_true", help="emit JSON instead of human-readable output"
    )
    args = parser.parse_args()

    api_key = _load_api_key()
    pairs = _parse_pairs(args.pairs)

    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    results = []
    for ws, proj in pairs:
        try:
            project = rf.workspace(ws).project(proj)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {ws}/{proj}: {exc}", file=sys.stderr)
            results.append({"workspace": ws, "project": proj, "error": str(exc)})
            continue
        versions = []
        try:
            for v in project.versions():
                versions.append(
                    {
                        "id": getattr(v, "id", None) or getattr(v, "version", None),
                        "name": getattr(v, "name", None),
                    }
                )
        except Exception:  # noqa: BLE001
            pass
        results.append(
            {
                "workspace": ws,
                "project": proj,
                "type": getattr(project, "type", None),
                "classes": list(getattr(project, "classes", []) or []),
                "versions": versions,
            }
        )

    if args.json:
        json.dump(results, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    for r in results:
        header = f"{r['workspace']}/{r['project']}"
        if "error" in r:
            print(f"=== {header} ===\n  ERROR: {r['error']}\n")
            continue
        print(f"=== {header} ===")
        print(f"  type:    {r.get('type')}")
        print(f"  classes: {r.get('classes')}")
        print(f"  versions: {r.get('versions')}")
        print()


if __name__ == "__main__":
    main()
