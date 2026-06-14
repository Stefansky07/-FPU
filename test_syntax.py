#!/usr/bin/env python
"""
Syntax check for FPU modules.

This script verifies that all modules can be imported correctly.
Run this before running actual benchmarks to catch syntax errors early.
"""

import ast
import sys
from pathlib import Path


def check_syntax(filepath: Path) -> bool:
    """Check Python file syntax."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
        ast.parse(source)
        print(f"  OK: {filepath.name}")
        return True
    except SyntaxError as e:
        print(f"  FAIL: {filepath.name}: {e}")
        return False


def main():
    print("Checking FPU module syntax...")
    print("="*40)

    fpu_dir = Path(__file__).parent / "fpu"
    all_passed = True

    # Check all Python files in fpu/
    for pyfile in sorted(fpu_dir.glob("*.py")):
        if not check_syntax(pyfile):
            all_passed = False

    # Check top-level scripts
    top_files = [
        "run_benchmark.py",
        "example_integration.py",
    ]

    print("\nChecking scripts...")
    print("="*40)
    for filename in top_files:
        filepath = Path(__file__).parent / filename
        if filepath.exists():
            if not check_syntax(filepath):
                all_passed = False

    print("\n" + "="*40)
    if all_passed:
        print("All syntax checks PASSED!")
        return 0
    else:
        print("Some syntax checks FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
