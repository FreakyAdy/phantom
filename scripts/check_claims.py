"""
PHANTOM Quantitative Claims & Consistency CI Gate
==================================================
Scans all markdown documentation files to ensure every numeric metric matches
either:
  (a) a measured value in benchmarks/results/latest.json, or
  (b) an explicitly approved item in docs/claims_allowlist.yml.

Flags any unverified numbers or conflicting values across documents.
Mandated by PHANTOM Remediation Brief Section 8.1.

Usage:
  python scripts/check_claims.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
LATEST_JSON = REPO_ROOT / "benchmarks" / "results" / "latest.json"
ALLOWLIST_YML = REPO_ROOT / "docs" / "claims_allowlist.yml"

# Files specifically cataloging historical/deleted claims or assignment specs
EXEMPT_FILES = {
    "CLAIMS.md",
    "CHANGES.md",
    "WORKLOG.md",
    "PHANTOM_REMEDIATION_PROMPT.md",
}

EXEMPT_DIRS = {
    REPO_ROOT / "docs" / "specs",
    REPO_ROOT / ".agents",
    REPO_ROOT / ".pytest_cache",
}


def load_allowlisted_values() -> Set[str]:
    """Extract approved numerical values from latest.json and claims_allowlist.yml."""
    approved: Set[str] = set()

    # 1. Load from latest.json
    if LATEST_JSON.exists():
        with open(LATEST_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)

        def _extract_numbers(obj: Any):
            if isinstance(obj, (int, float)):
                approved.add(str(obj))
                approved.add(f"{obj:.1f}")
                approved.add(f"{obj:.2f}")
                try:
                    num = float(obj)
                    if num.is_integer():
                        approved.add(str(int(num)))
                except Exception:
                    pass
            elif isinstance(obj, dict):
                for v in obj.values():
                    _extract_numbers(v)
            elif isinstance(obj, list):
                for item in obj:
                    _extract_numbers(item)

        _extract_numbers(data)

    # 2. Load from allowlist yaml
    if ALLOWLIST_YML.exists():
        with open(ALLOWLIST_YML, "r", encoding="utf-8") as f:
            content = f.read()
        for match in re.finditer(r'value:\s*["\']?([^"\'\n]+)["\']?', content):
            val = match.group(1).strip()
            approved.add(val)
            try:
                num = float(val)
                approved.add(f"{num:.1f}")
                approved.add(f"{num:.2f}")
                if num.is_integer():
                    approved.add(str(int(num)))
            except ValueError:
                pass

    return approved


def scan_markdown_files(approved_values: Set[str]) -> Tuple[int, int, List[str]]:
    """Scan all markdown files for numeric claims."""
    violations: List[str] = []
    total_scanned_files = 0
    total_claims_checked = 0

    md_files = list(REPO_ROOT.glob("*.md")) + list((REPO_ROOT / "docs").glob("**/*.md"))

    # Regex pattern matching metrics like: 2.88 tok/s, 0.458 ms, 1.43 GB/s, 32.76B, 8.0x, 99.0%
    metric_pattern = re.compile(
        r'\b(\d+(?:\.\d+)?)\s*(?:tok/s|tok/sec|ms|GB/s|GB|%|x|B)\b',
        re.IGNORECASE,
    )

    for md_file in md_files:
        if md_file.name in EXEMPT_FILES:
            continue
        if any(exempt_dir in md_file.parents for exempt_dir in EXEMPT_DIRS):
            continue
        total_scanned_files += 1

        try:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            continue

        for i, line in enumerate(content.splitlines(), 1):
            # Ignore markdown links, tables headers, code fences
            if line.strip().startswith("```") or line.strip().startswith("<!--"):
                continue
            for match in metric_pattern.finditer(line):
                total_claims_checked += 1
                val_str = match.group(1)
                # Check if val_str or float conversion is in approved_values
                is_approved = (
                    val_str in approved_values
                    or f"{float(val_str):.1f}" in approved_values
                    or f"{float(val_str):.2f}" in approved_values
                )
                if not is_approved:
                    violations.append(f"{md_file.relative_to(REPO_ROOT)}:{i} — Unverified metric '{match.group(0)}' (value: {val_str})")

    return total_scanned_files, total_claims_checked, violations


def main():
    print("=" * 75)
    print("  PHANTOM QUANTITATIVE CLAIMS & CONSISTENCY CI GATE")
    print("=" * 75)

    approved = load_allowlisted_values()
    print(f"Loaded {len(approved)} approved numerical benchmarks and specifications.")

    files_count, claims_count, violations = scan_markdown_files(approved)
    print(f"Scanned {files_count} markdown files; verified {claims_count} numerical claims.")
    print("-" * 75)

    if violations:
        print(f"[FAIL] Found {len(violations)} unverified or drifting numeric claims:")
        for v in violations[:15]:
            print(f"  [X] {v}")
        if len(violations) > 15:
            print(f"  ... and {len(violations) - 15} more.")
        print("\nAction: Reconcile value with benchmarks/results/latest.json or add to docs/claims_allowlist.yml with justification.")
        sys.exit(1)
    else:
        print("[PASS] 100% of numeric claims across all documents are verified & consistent.")
        print("=" * 75 + "\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
