#!/usr/bin/env python3
"""Verify polymathic-aion is installed (``import aion``)."""

from __future__ import annotations

import sys


def main() -> int:
    try:
        import aion
        from aion.codecs import CodecManager
    except ImportError as e:
        print("FAIL: cannot import aion (polymathic-aion not installed)", file=sys.stderr)
        print("  Fix: bash scripts/bootstrap_venv.sh", file=sys.stderr)
        print("    or: .venv/bin/pip install -e .", file=sys.stderr)
        print(f"  ({e})", file=sys.stderr)
        return 1
    print(f"ok: aion at {aion.__file__}")
    print(f"ok: CodecManager {CodecManager}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
