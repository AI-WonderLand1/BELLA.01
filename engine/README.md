# BELLA.01 Hybrid Dataset Engine

Node.js/TypeScript owns orchestration and the local HTTP API. Python owns dataset download adapters, parsing, normalization, deduplication, and publishing. Kaggle is the primary dataset source; Hugging Face is an optional mirror for prepared dataset artifacts.

## Data flow

```text
Kaggle dataset
    |  Kaggle API credentials
    v
engine/data/kaggle/       (downloaded source files)
    |
    |  Python prepare
    v
engine/data/prepared/     (train.jsonl, heldout.jsonl, manifest.json)
    |
    |  Hugging Face upload_folder
    v
ai-wonderland-innovation/Bella-Training-Datasets
```

The model repository remains separate: `ai-wonderland-innovation/BELLA.01`. This pipeline does not upload model weights or start model training.

## Requirements

- Node.js 20+
- Python 3.10+
- Kaggle account/API token
- Hugging Face write token with access to the dataset repository
- Optional: `pyarrow` for Parquet files

## Install

```bash
cd engine
npm install
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r python/requirements.txt
```

## Credentials and configuration

Copy `.env.example` to `.env` and set the values locally. Do not commit `.env`, Kaggle API keys, or Hugging Face tokens.

- `KAGGLE_USERNAME` and `KAGGLE_KEY`: Kaggle API credentials.
- `KAGGLE_DATASET`: Kaggle dataset identifier in `owner/dataset-slug` form. This must be supplied for the actual dataset; it is intentionally not guessed.
- `HF_TOKEN`: Hugging Face token with write access to the dataset repo.
- `HF_DATASET_REPO`: defaults to `ai-wonderland-innovation/Bella-Training-Datasets`.
- `BELLA_KAGGLE_DOWNLOAD_DIR`: defaults to `engine/data/kaggle`.
- `BELLA_DATASET_INPUT`: defaults to no value; set this to the downloaded source folder before preparing.
- `BELLA_DATASET_OUTPUT`: defaults to `engine/data/prepared`.
- `BELLA_PYTHON`: Python executable; defaults to `python3`.
- `PORT`: API port; defaults to `8787`.

The `.env` file is not automatically loaded by Node. Export its variables in the shell or use your preferred local environment loader.

## Run the pipeline

Download from Kaggle:

```bash
python python/bella_dataset/cli.py download-kaggle --dataset "$KAGGLE_DATASET" --output data/kaggle
```

Prepare the downloaded files:

```bash
python python/bella_dataset/cli.py prepare --input data/kaggle --output data/prepared
```

Mirror prepared outputs to the existing Hugging Face dataset repository:

```bash
python python/bella_dataset/cli.py mirror-huggingface --input data/prepared
```

The mirror command checks that the target dataset repository exists and that the supplied token can access it. It uploads the prepared folder; it does not delete the repository or source files. Review the target repo contents and commit diff after upload.

## Local API

Start from the `engine` directory after exporting the variables:

```bash
npm run dev
```

Routes (localhost only):

- `GET /health`
- `POST /api/datasets/download-kaggle` — downloads the configured `KAGGLE_DATASET`.
- `POST /api/datasets/prepare` — prepares the configured `BELLA_DATASET_INPUT`.
- `POST /api/datasets/mirror-huggingface` — mirrors `BELLA_DATASET_OUTPUT` to `HF_DATASET_REPO`.

## Data safety and current boundaries

- Kaggle is the source; Hugging Face is the mirror. Do not manually edit the two copies independently.
- No repository deletion, dataset deletion, model upload, or training is performed.
- The Hugging Face mirror uploads the prepared output directory. Depending on the Hub upload behavior and repo state, files not present in the uploaded folder may remain in the remote repo; inspect the repo after publishing.
- Verify dataset licensing, provenance, quality, and split policy before training or redistribution.
- The normalizer writes `train.jsonl` and `heldout.jsonl`. It processes held-out paths first before deduplication to reduce train/held-out leakage. Validation and test path names are currently combined as held-out; separate validation/test outputs remain to be implemented.
- This is a local first milestone. The API has no authentication, job queue, persistent job history, or concurrency lock. It binds to `127.0.0.1`; do not expose it publicly without adding those controls.
- Malformed JSONL lines are skipped and counted; file-level failures are recorded in the manifest.
- Live Kaggle downloads and Hugging Face uploads require the user's credentials and have not been run from this repository change.


## Unit tests

From the `engine` directory:

```bash
PYTHONPATH=python python -m unittest discover -s python/tests -v
npm run typecheck
npm run build
```
