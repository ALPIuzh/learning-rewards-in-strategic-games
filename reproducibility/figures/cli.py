"""Small CLI for listing, generating, and checking publication figures."""

from __future__ import annotations

import argparse

from reproducibility.figures.manifest import load_manifest
from reproducibility.figures.render import render_figures, write_build_manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="list registered figures")
    for command, help_text in (
        ("generate", "generate deterministic TikZ fragments"),
        ("check", "fail if generated fragments do not match their sources"),
    ):
        child = subparsers.add_parser(command, help=help_text)
        child.add_argument("ids", nargs="*", help="optional figure ids; default: all")
    return parser


def main() -> None:
    args = _parser().parse_args()
    manifest = load_manifest()
    if args.command == "list":
        for figure in manifest.figures:
            print(f"{figure.id:32} {figure.output.as_posix()}")
        return
    figures = manifest.select(args.ids)
    check = args.command == "check"
    records = render_figures(figures, check=check)
    if not args.ids:
        write_build_manifest(records, check=check)
    verb = "checked" if check else "generated"
    print(f"{verb} {len(records)} figure(s)")
