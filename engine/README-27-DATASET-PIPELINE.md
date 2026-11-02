# BELLA.01 27-dataset pipeline

## Data layout
Place each downloaded source in its own directory under `engine/data/raw-27/`. The folder name must match one of the 27 registry names or its generated ID. Nested directories are allowed. The pipeline does not download datasets from Kaggle or HF by guessing identifiers; keep source downloads separate and preserve originals.

## Install
From `engine/`:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r python/requirements.txt -r python/requirements-pipeline.txt
```

## Run
From `engine/`:
```bash
# Extract archives, parse data and write one JSONL per training lane.
PYTHONPATH=python python -m bella_dataset.pipeline ingest --input data/raw-27 --output data/lanes

# Train the custom ByteLevel BPE vocabulary.
PYTHONPATH=python python -m bella_dataset.pipeline tokenizer --input data/lanes --output data/tokenizer --vocab-size 32000

# Stream causal-LM examples. Security lane is excluded unless explicitly enabled.
PYTHONPATH=python python -m bella_dataset.pipeline stream --input data/lanes --tokenizer data/tokenizer/bella_bpe.json --seq-length 2048 --limit 10
```

## Output
- `data/lanes/architectural_reasoning.jsonl`
- `data/lanes/full_stack_compilation.jsonl`
- `data/lanes/agentic_self_learning.jsonl`
- `data/lanes/secured_sandbox_guard.jsonl`
- `data/lanes/ingestion_manifest.json`
- `data/tokenizer/bella_bpe.json` and `tokenizer_manifest.json`

Ingestion supports JSON, JSONL, NDJSON, CSV, Parquet, plain text, Markdown, and common code files; ZIP, TAR, GZ, BZ2 and XZ archives are extracted with path and symlink checks. Parquet and large top-level JSON arrays are streamed in batches (large JSON arrays use ijson). Deduplication hashes are stored in temporary SQLite on disk rather than an in-memory set. The manifest reports discovered and missing datasets, counts, skipped binary samples, and file errors.

The security lane excludes executable/binary malware artifacts; only text metadata and feature records are ingested. It is excluded from model token streaming by default. Never execute dataset contents. Dataset licensing, redistribution rights, provenance, and data quality must be reviewed individually before training. A registry entry is not proof that a dataset has been downloaded or is licensed.

The tokenizer is a vocabulary, not a pretrained language model. Streamed records contain shifted next-token labels and attention masks, but do not train/update model weights or implement RL. Model loss, optimizer, security reward feedback, and model-specific adapters are separate stages.
