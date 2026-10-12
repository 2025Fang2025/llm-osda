"""Regenerate SYNC_FILES.txt / SYNC_MANIFEST.md5 for pushing this revision to the server.
(NEW FILE: tools/make_sync_manifest.py)

Lists every file that differs from the pristine llm_osda_public checkout plus the
new revision-only files, so nothing silently stays stale on the GPU box.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Always ship these, tracked or not.
EXTRA = [
    "run_revision.sh", "run_revision.local.sh.example", "run_revision_T2.sh", "run_strata.sh",
    "tools/make_t3_subset.py", "tools/make_strata_subset.py", "tools/pool_strata.py",
    "tools/make_sync_manifest.py",
    "tests/conftest.py", "tests/test_strata_pipeline.py", "tests/test_horizon_strata.py",
    "tests/test_no_horizon_leak.py", "tests/test_revision_correctness.py",
    "tests/test_export_envelope_paper_results.py",
    "docs/plans/2026-10-02-revision-correctness.md", ".gitignore",
]
SKIP_PREFIX = ("outputs/", "data/", "SYNC_", ".git/", ".pytest_cache", "__pycache__")


def tracked_changes():
    out = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                         capture_output=True, text=True).stdout
    files = []
    for line in out.splitlines():
        p = line[3:].strip().strip('"')
        if p.endswith("/") or p.startswith(SKIP_PREFIX):
            continue
        files.append(p)
    return files


def main():
    files = sorted({f for f in tracked_changes() + EXTRA if (ROOT / f).is_file()})
    (ROOT / "SYNC_FILES.txt").write_text("\n".join(files) + "\n")
    lines = []
    for f in files:
        h = hashlib.md5((ROOT / f).read_bytes()).hexdigest()
        lines.append(f"{h}  {f}")
    (ROOT / "SYNC_MANIFEST.md5").write_text("\n".join(lines) + "\n")
    print(f"SYNC_FILES.txt: {len(files)} files")
    for f in files:
        print("  ", f)


if __name__ == "__main__":
    sys.exit(main())
