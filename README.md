# BELLA.01

BELLA.01 is the dataset preparation and orchestration project for AI WONDERLAND. It uses **Node.js/TypeScript for orchestration** and **Python for data processing**.

The initial pipeline is designed to take source datasets from Kaggle, normalize supported formats into chat-training JSONL, and optionally mirror the prepared artifacts to the BELLA.01 Hugging Face dataset repository.

## Repositories

- **Model:** [ai-wonderland-innovation/BELLA.01](https://huggingface.co/ai-wonderland-innovation/BELLA.01)
- **Dataset mirror:** [ai-wonderland-innovation/Bella-Training-Datasets](https://huggingface.co/datasets/ai-wonderland-innovation/Bella-Training-Datasets)

The model repository and dataset repository are separate. This codebase does not upload model weights or start model training.

## Architecture

- **Node.js + TypeScript:** local HTTP API, orchestration, and Python worker execution.
- **Python:** dataset download integration, parsing, normalization, deduplication, and manifest generation.
- **Kaggle:** intended primary source for the configured dataset.
- **Hugging Face:** optional mirror for prepared dataset artifacts.

```text
Kaggle dataset
    |
    v
engine/data/kaggle/       original downloaded files
    |
    v
Python preparation
    |
    +--> engine/data/prepared/train.jsonl
    +--> engine/data/prepared/heldout.jsonl
    +--> engine/data/prepared/manifest.json
    |
    v
Hugging Face dataset repository (optional mirror)
```

## Current capabilities

- Reads JSONL, NDJSON, JSON, CSV, and Parquet files (Parquet requires `pyarrow`).
- Normalizes common chat, ShareGPT, and instruction/prompt-completion record shapes.
- Deduplicates normalized message sequences.
- Writes training and held-out JSONL files plus a processing manifest.
- Downloads a Kaggle dataset when its exact `owner/dataset-slug` is configured.
- Mirrors prepared output to the existing Hugging Face dataset repository when an authorized token is configured.
- Exposes a localhost-only Node API for health checks and pipeline actions.

## Quick start

Requirements:

- Node.js 20 or newer
- Python 3.10 or newer
- Kaggle API credentials for Kaggle downloads
- Hugging Face write token for mirroring

Install:

```bash
cd engine
npm install
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r python/requirements.txt
```

Configure credentials in your shell or secret manager. The repository includes `engine/.env.example` as a template, but the Node service does not automatically load `.env`; export the variables before starting it. Never commit real credentials.

Required variables for the integrations:

- `KAGGLE_USERNAME`
- `KAGGLE_KEY`
- `KAGGLE_DATASET` — exact Kaggle dataset identifier in `owner/dataset-slug` form
- `HF_TOKEN` — Hugging Face write token
- `HF_DATASET_REPO` — defaults to `ai-wonderland-innovation/Bella-Training-Datasets`

The exact Kaggle dataset identifier has not been configured in this repository. Do not substitute a Kaggle Notebook URL for a Kaggle Dataset identifier.

Run each stage from the `engine` directory:

```bash
# Download the configured Kaggle dataset
python python/bella_dataset/cli.py download-kaggle \
  --dataset "$KAGGLE_DATASET" \
  --output data/kaggle

# Normalize source files
python python/bella_dataset/cli.py prepare \
  --input data/kaggle \
  --output data/prepared

# Optional: mirror prepared files to Hugging Face
python python/bella_dataset/cli.py mirror-huggingface \
  --input data/prepared
```

Start the local API:

```bash
npm run dev
```

Routes:

- `GET /health`
- `POST /api/datasets/download-kaggle`
- `POST /api/datasets/prepare`
- `POST /api/datasets/mirror-huggingface`

The API binds to `127.0.0.1`. It currently has no authentication, job queue, persistent job history, or concurrency lock. Do not expose it to a public network until those controls are implemented.

## Data, licensing, and safety

The repository's **software source code** is licensed under the [Do What the F*ck You Want to Public License (WTFPL), Version 2](LICENSE).

That software license does **not** automatically license the datasets, model weights, or other third-party materials processed by this project. Each source dataset retains its own license and redistribution conditions. Verify rights, attribution requirements, and provenance before training on or redistributing processed data. Do not apply a blanket license to the combined dataset unless you have the rights to do so.

The pipeline does not delete source files, delete the Hugging Face dataset repository, upload model weights, or train a model. The mirror operation uploads the prepared output folder; inspect the remote repository after an upload because files not present locally may remain remotely.

## Known limitations

- The Kaggle dataset slug is intentionally unset until the exact dataset is confirmed.
- Validation and test files are currently combined into a single held-out split; separate validation and test outputs are not implemented yet.
- The first preparation pass holds deduplication hashes in memory, which may be unsuitable for very large corpora.
- Malformed JSONL lines are skipped; file-level errors are printed and processing continues, so inspect logs and manifest counts before trusting an output.
- The local API has no authentication, queue, or concurrency lock.
- Kaggle downloads and Hugging Face uploads have not been verified end-to-end with live credentials in this repository session.
- No model training, tokenizer integration, evaluation harness, or training adapter is included yet.

## Tests and checks

From `engine/`:

```bash
npm run typecheck
npm run build
python -m unittest discover -s python/tests -v
```

Run these checks before treating the current implementation as production-ready. Live provider integration tests require valid credentials and a confirmed Kaggle dataset ID.

## Project layout

```text
engine/
  src/                 Node.js/TypeScript HTTP API
  python/
    bella_dataset/     Python CLI, normalization, provider integrations
    tests/             Python unit tests
  shared/              Shared JSON Schema
  data/                Local source and prepared data (git-ignored)
  .env.example         Environment variable template
  README.md            Engine-specific setup and API details
```
