from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

SUPPORTED = {".jsonl", ".ndjson", ".json", ".csv", ".parquet", ".pq"}
EVAL_PATH = re.compile(r"(?:^|[/_.-])(test|tests|eval|evaluation|valid|validation|dev)(?:[/_.-]|$)", re.I)


def role_name(value: Any) -> str | None:
    role = str(value or "").strip().lower()
    return {
        "human": "user", "user": "user", "prompt": "user", "instruction": "user",
        "gpt": "assistant", "assistant": "assistant", "bot": "assistant",
        "completion": "assistant", "system": "system", "tool": "tool", "function": "tool",
    }.get(role)


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def normalize_record(record: Any) -> list[dict[str, str]] | None:
    if not isinstance(record, dict):
        return None
    for key, role_key, content_keys in (
        ("messages", ("role", "from", "speaker"), ("content", "value", "text")),
        ("conversations", ("from", "role", "speaker"), ("value", "content", "text")),
    ):
        items = record.get(key)
        if isinstance(items, list):
            messages = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                role = next((role_name(item.get(k)) for k in role_key if item.get(k)), None)
                content = next((as_text(item.get(k)) for k in content_keys if item.get(k) is not None), "")
                if role and content:
                    messages.append({"role": role, "content": content})
            if len(messages) >= 2 and any(m["role"] == "assistant" for m in messages):
                return messages

    prompt = next((record.get(k) for k in ("prompt", "instruction", "question", "query") if record.get(k) is not None), None)
    answer = next((record.get(k) for k in ("completion", "response", "output", "answer") if record.get(k) is not None), None)
    if prompt is not None and answer is not None and as_text(prompt) and as_text(answer):
        extra = as_text(record.get("input"))
        user_text = as_text(prompt) + (f"\n\n{extra}" if extra else "")
        return [{"role": "user", "content": user_text}, {"role": "assistant", "content": as_text(answer)}]
    return None


def records_from(path: Path, stats: dict[str, int] | None = None) -> Iterable[Any]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line_number, line in enumerate(stream, start=1):
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        if stats is not None:
                            stats["malformed_jsonl_lines"] = stats.get("malformed_jsonl_lines", 0) + 1
                        print(f"INVALID_JSONL {path}:{line_number}")
                        continue
    elif suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        if isinstance(data, list):
            yield from data
        elif isinstance(data, dict):
            nested = next((data[k] for k in ("data", "records", "examples") if isinstance(data.get(k), list)), None)
            yield from (nested if nested is not None else [data])
    elif suffix == ".csv":
        with path.open(encoding="utf-8", errors="replace", newline="") as stream:
            yield from csv.DictReader(stream)
    elif suffix in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Parquet input requires optional dependency: pip install pyarrow") from exc
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=1000):
            yield from batch.to_pylist()


def prepare(input_root: Path, output_root: Path) -> dict[str, Any]:
    if not input_root.is_dir():
        raise FileNotFoundError(f"Dataset input directory does not exist: {input_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    paths = [p for p in input_root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED]
    # Process held-out files first so duplicate records never leak from held-out data into training.
    paths.sort(key=lambda p: (not bool(EVAL_PATH.search(p.relative_to(input_root).as_posix())), p.as_posix()))
    seen: set[str] = set()
    counts = {"files_found": len(paths), "records_seen": 0, "normalized": 0, "duplicates": 0, "skipped": 0, "malformed_jsonl_lines": 0, "file_errors": 0, "train": 0, "heldout": 0}
    file_errors: list[dict[str, str]] = []
    handles = {
        "train": (output_root / "train.jsonl").open("w", encoding="utf-8"),
        "heldout": (output_root / "heldout.jsonl").open("w", encoding="utf-8"),
    }
    try:
        for path in paths:
            try:
                relative = path.relative_to(input_root).as_posix()
                heldout = bool(EVAL_PATH.search(relative))
                for raw in records_from(path, counts):
                    counts["records_seen"] += 1
                    messages = normalize_record(raw)
                    if not messages:
                        counts["skipped"] += 1
                        continue
                    canonical = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    digest = hashlib.sha256(canonical.encode()).hexdigest()
                    if digest in seen:
                        counts["duplicates"] += 1
                        continue
                    seen.add(digest)
                    row = {"messages": messages, "source_file": relative}
                    bucket = "heldout" if heldout else "train"
                    handles[bucket].write(json.dumps(row, ensure_ascii=False) + "\n")
                    counts["normalized"] += 1
                    counts[bucket] += 1
            except Exception as exc:
                counts["file_errors"] += 1
                file_errors.append({"file": path.relative_to(input_root).as_posix(), "error": f"{type(exc).__name__}: {exc}"})
                print(f"FILE_ERROR {path}: {type(exc).__name__}: {exc}")
    finally:
        for handle in handles.values():
            handle.close()

    manifest = {
        "engine": "BELLA.01 Hybrid Dataset Engine",
        "format": "chat-messages-jsonl",
        "counts": counts,
        "file_errors": file_errors,
        "outputs": {name: f"{name}.jsonl" for name in handles},
        "notes": [
            "This stage prepares data only; it does not train or publish a model.",
            "Review source licenses, provenance, quality, and held-out split policy before training.",
            "Training and held-out records are deduplicated together to prevent exact overlap."
        ],
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="BELLA.01 dataset engine worker")
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("prepare")
    command.add_argument("--input", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)

    kaggle = sub.add_parser("download-kaggle")
    kaggle.add_argument("--dataset", required=True)
    kaggle.add_argument("--output", type=Path, required=True)

    mirror = sub.add_parser("mirror-huggingface")
    mirror.add_argument("--input", type=Path, required=True)
    mirror.add_argument("--repo", default=None)

    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.input, args.output)
    elif args.command == "download-kaggle":
        from integrations import download_kaggle_dataset
        print(json.dumps(download_kaggle_dataset(args.dataset, args.output), indent=2))
    elif args.command == "mirror-huggingface":
        from integrations import mirror_to_huggingface
        print(json.dumps(mirror_to_huggingface(args.input, args.repo), indent=2))


if __name__ == "__main__":
    main()
