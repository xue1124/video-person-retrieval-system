"""Offline source syntax and common publication checks; does not import the application."""
from __future__ import annotations

import ast
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", "node_modules", ".venv", "venv", "__pycache__", "dist"}
RUNTIME = {"video_archives", "video_crops", "query_storage", "temp_clips", "tracking_snapshots", "tracking_faiss", "outputs", "logs", "run", "siglip_v1"}
FORBIDDEN = {".mp4", ".avi", ".mov", ".mkv", ".onnx", ".pt", ".pth", ".engine", ".rknn", ".npz", ".npy", ".safetensors", ".whl", ".pem", ".key", ".faiss", ".index"}
TEXT = {".py", ".ts", ".vue", ".json", ".sql", ".md", ".sh", ".example", ".service"}
SECRET_PATTERNS = (
    re.compile(r"\b(?:app-|sk-)[A-Za-z0-9_-]{24,}"),
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
)


def main() -> int:
    failures = []
    checked = 0
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    candidates = [Path(item.decode("utf-8")) for item in result.stdout.split(b"\0") if item]
    for rel in candidates:
        path = ROOT / rel
        if not path.is_file() or any(part in SKIP for part in rel.parts):
            continue
        if (path.suffix in FORBIDDEN or any(part in RUNTIME for part in rel.parts)
                or path.suffix == ".env" or (path.name.startswith(".env") and not path.name.endswith("example"))):
            failures.append(f"Private/generated file present: {rel}")
            continue
        if path.stat().st_size > 10 * 1024 * 1024:
            failures.append(f"Unexpected large file: {rel}")
        if path.suffix not in TEXT:
            continue
        source = path.read_text(encoding="utf-8-sig")
        if path.suffix == ".py":
            try:
                ast.parse(source, filename=str(rel))
                checked += 1
            except SyntaxError as exc:
                failures.append(f"Syntax error: {rel}:{exc.lineno}")
        # TypeScript configuration supports JSON-with-comments; tsc validates it.
        if path.suffix == ".json" and not path.name.startswith("tsconfig"):
            json.loads(source)
        for pattern in SECRET_PATTERNS:
            if pattern.search(source):
                failures.append(f"Possible credential: {rel} (value not printed)")
    for failure in failures:
        print(failure)
    print(f"Python files checked: {checked}; findings: {len(failures)}")
    print("This check does not verify deployment, model accuracy, or publication rights.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
