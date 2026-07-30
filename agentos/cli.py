"""Command-line entry point for the Agent OS.

    python -m agentos info
    python -m agentos run "Design a backup strategy for an airgapped server"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from agentos.bootstrap import build_default_os


def _cmd_info(args) -> int:
    os_ = build_default_os(workspace=args.workspace, state_dir=args.state, config_path=args.config)
    print(json.dumps(os_.describe(), indent=2))
    return 0


def _cmd_run(args) -> int:
    os_ = build_default_os(workspace=args.workspace, state_dir=args.state, config_path=args.config)
    report = asyncio.run(os_.run(args.goal))
    print(report.trace())
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agentos", description="Airgapped Agent OS")
    p.add_argument("--workspace", default=".", help="sandbox root for file tools")
    p.add_argument("--state", default=None, help="dir to persist memory/audit/blackboard")
    p.add_argument("--config", default=None, help="optional JSON config overriding defaults")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info", help="print the OS configuration").set_defaults(func=_cmd_info)

    run = sub.add_parser("run", help="run a goal end-to-end")
    run.add_argument("goal", help="the goal to accomplish")
    run.set_defaults(func=_cmd_run)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
