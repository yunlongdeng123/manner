#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from scenariotopo.data.validate import validate_manifest


def main():
    parser = argparse.ArgumentParser(description="Validate a ScenarioTopo manifest and split isolation")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    report = validate_manifest(args.manifest, args.data_root)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    if not report["valid"]:
        sys.exit(2)


if __name__ == "__main__":
    main()

