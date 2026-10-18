from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from pwgp.data.manager import DatasetManager


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Acquire and inspect PWGP datasets")
    parser.add_argument("--root", default="data", help="dataset cache directory")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list registered datasets")
    download = commands.add_parser("download", help="download a registered dataset")
    download.add_argument("dataset")
    download.add_argument("--date", type=date.fromisoformat)
    imported = commands.add_parser("import", help="import and validate a local file")
    imported.add_argument("dataset")
    imported.add_argument("path", type=Path)
    prepare = commands.add_parser("prepare", help="prepare downloaded raw data")
    prepare.add_argument("dataset")
    inspect = commands.add_parser("inspect", help="show an acquisition manifest")
    inspect.add_argument("dataset")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manager = DatasetManager(args.root)
    if args.command == "list":
        for item in manager.list():
            print(f"{item.name:10} {item.acquisition:16} {item.description}")
    elif args.command == "download":
        for path in manager.download(args.dataset, day=args.date):
            print(path)
    elif args.command == "import":
        print(manager.import_file(args.dataset, args.path))
    elif args.command == "prepare":
        print(manager.prepare(args.dataset))
    elif args.command == "inspect":
        print(json.dumps(manager.manifest(args.dataset), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

