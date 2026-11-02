# BELLA.01 Hybrid Dataset Engine

Node.js/TypeScript owns orchestration and the local HTTP API. Python owns dataset parsing and normalization. Both layers communicate through a controlled child-process interface and a JSONL/JSON manifest contract.

## Requirements
- Node.js 20+
- Python 3.10+
- Optional: `pyarrow` only when Parquet files are among the inputs

## Install
```bash
cd engine
npm install
# Only if your inputs contain Parquet:
python3 -m pip install -r python/requirements.txt
```

## Configure and run
```bash
export BELLA_DATASET_INPUT="/absolute/path/to/datasets"
export BELLA_DATASET_OUTPUT="/absolute/path/to/prepared"
npm run dev
```

The API binds to localhost:
- `GET /health`
- `POST /api/datasets/prepare`

The preparation endpoint scans supported JSONL, NDJSON, JSON, CSV, and optional Parquet files. It writes `train.jsonl`, `heldout.jsonl`, and `manifest.json`. Explicit test/evaluation/validation/dev path names are held out. Exact duplicate conversations are removed across both splits to prevent overlap.

## Current boundaries
- No model training, tokenizer/model download, GPU requirement, Hugging Face upload, or dataset deletion.
- No claims that source datasets are licensed or suitable for training; review provenance and licenses before use.
- This first milestone is a local orchestration service. Authentication, job queueing, persistent job history, richer schema adapters, validation/quality scoring, and training adapters are later milestones.
