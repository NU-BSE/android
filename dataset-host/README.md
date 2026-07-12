# ARC Human-Played Games Dataset Host

## Complete installation and running guide

This guide covers:

- Local development with Docker Compose, PostgreSQL, and MinIO.
- Running the API directly in a Python virtual environment.
- Uploading and indexing the full `public-games-dataset` directory.
- Connecting a browser frontend or Python training process.
- Deploying the API with managed PostgreSQL and S3-compatible storage.
- Running optional Optuna studies.
- Updates, backups, security, and troubleshooting.

---

## 1. Architecture

The stack deliberately separates large raw recordings from searchable metadata:

```text
Dataset folder on ingestion machine
              |
              | python -m app.ingest
              v
   +----------------------+        +----------------------+
   | S3-compatible bucket |        | PostgreSQL           |
   | Raw .jsonl objects   |        | Recording metadata   |
   +----------+-----------+        +-----------+----------+
              |                                |
              +---------------+----------------+
                              |
                         +----v----+
                         | FastAPI |
                         +----+----+
                              |
                  Browser / trainer / Optuna
```

The original dataset directory is required only while ingestion runs. After upload and indexing, the deployed API reads from PostgreSQL and object storage.

Optuna is optional. It tracks hyperparameter studies and calls a model-specific objective; it is not used as the dataset database.

---

## 2. Prerequisites

### Docker installation

For the simplest setup, install:

- Docker Engine or Docker Desktop.
- Docker Compose v2, invoked as `docker compose`.
- Enough free disk space for the source dataset, the MinIO copy, PostgreSQL metadata, and Docker images.

Check the installation:

```bash
docker --version
docker compose version
```

### Dataset layout

The importer recursively finds every `.jsonl` file. Game codes are inferred from folders matching two letters followed by two digits, such as `aa00` or `ar25`.

```text
/arc-prize-2026-arc-agi-3/public-games-dataset/
├── aa00/
│   ├── first.recording.jsonl
│   ├── second.recording.jsonl
│   └── ...
├── aa01/
└── ...
```

Check the dataset before starting:

```bash
find /arc-prize-2026-arc-agi-3/public-games-dataset \
  -type f -name '*.jsonl' | wc -l

du -sh /arc-prize-2026-arc-agi-3/public-games-dataset
```

On Windows, run the Linux commands inside WSL2, or use the PowerShell commands shown below.

---

## 3. Unpack the project

From the directory containing `arc-dataset-host.zip`:

```bash
unzip arc-dataset-host.zip
cd arc-dataset-host
```

The important files are:

```text
arc-dataset-host/
├── app/                       FastAPI service and importer
├── clients/                   TypeScript and Python clients
├── experiments/               Optional Optuna runner
├── tests/                     Parser tests
├── .env.example               Production environment template
├── docker-compose.yml         Local PostgreSQL + MinIO + API
├── Dockerfile                 API/importer image
└── requirements*.txt          Python dependency layers
```

---

## 4. Local installation with Docker Compose

This starts three local services:

- FastAPI: `http://localhost:8000`
- MinIO S3 endpoint: `http://localhost:9000`
- MinIO console: `http://localhost:9001`
- PostgreSQL is internal to the Compose network and is not published to the host.

### 4.1 Set the dataset path

#### Linux, macOS, or WSL2

```bash
cd /path/to/arc-dataset-host
export DATASET_PATH=/arc-prize-2026-arc-agi-3/public-games-dataset
```

Confirm that Docker will see a real directory:

```bash
test -d "$DATASET_PATH" && echo "Dataset directory found"
```

#### Windows PowerShell

```powershell
Set-Location C:\path\to\arc-dataset-host
$env:DATASET_PATH = "C:\arc-prize-2026-arc-agi-3\public-games-dataset"
```

Docker Desktop must be allowed to mount the drive containing the dataset. WSL2 paths are usually more reliable for large datasets.

### 4.2 Optional local read API key

The local API is public by default. To require a shared key:

#### Linux, macOS, or WSL2

```bash
export READ_API_KEY='replace-with-a-long-random-value'
```

Generate one with Python:

```bash
python -c 'import secrets; print(secrets.token_urlsafe(48))'
```

#### Windows PowerShell

```powershell
$env:READ_API_KEY = "replace-with-a-long-random-value"
```

### 4.3 Build and start the services

```bash
docker compose up -d --build
```

Inspect service state:

```bash
docker compose ps
```

Inspect startup logs:

```bash
docker compose logs --tail=100 api
docker compose logs --tail=100 postgres
docker compose logs --tail=100 minio
```

The API startup creates the `recordings` metadata table. The `minio-init` service creates the local `arc-human-games` bucket.

### 4.4 Check the API before ingestion

Without an API key:

```bash
curl -s http://localhost:8000/healthz
curl -s http://localhost:8000/v1/stats
```

With an API key:

```bash
curl -s \
  -H "X-API-Key: $READ_API_KEY" \
  http://localhost:8000/v1/stats
```

Before ingestion, statistics should contain zero recordings.

Interactive documentation:

```text
http://localhost:8000/docs
```

### 4.5 Ingest the complete dataset

Run the importer inside the API container. The dataset is mounted read-only at `/data/public-games-dataset`.

```bash
docker compose exec api \
  python -m app.ingest /data/public-games-dataset
```

Equivalent Make target:

```bash
make ingest
```

For each file, ingestion:

1. Parses every non-empty JSONL line.
2. Computes SHA-256 over the exact file bytes.
3. Extracts metadata such as timestamps, frame shapes, available actions, transition count, and final summary.
4. Uploads the original file to object storage.
5. Inserts one PostgreSQL metadata row.

Typical output:

```text
OK   aa00/example.recording.jsonl -> raw/aa00/... (...uuid...)
SKIP aa00/already-loaded.recording.jsonl (...uuid...)
Processed 300 files with 0 failure(s).
```

A nonzero failure count makes the command exit with status `1`, but independent files continue processing. Search the output for `FAIL` and correct those files or credentials before rerunning.

### 4.6 Verify ingestion

```bash
curl -s http://localhost:8000/v1/stats
curl -s http://localhost:8000/v1/games
curl -s 'http://localhost:8000/v1/recordings?game_code=aa00&limit=5'
```

Add `-H "X-API-Key: $READ_API_KEY"` when authentication is enabled.

A readable formatted check using Python, without requiring `jq`:

```bash
python - <<'PY'
import json
import urllib.request

with urllib.request.urlopen("http://localhost:8000/v1/stats") as response:
    print(json.dumps(json.load(response), indent=2))
PY
```

For a keyed API:

```bash
python - <<'PY'
import json
import os
import urllib.request

request = urllib.request.Request(
    "http://localhost:8000/v1/stats",
    headers={"X-API-Key": os.environ["READ_API_KEY"]},
)
with urllib.request.urlopen(request) as response:
    print(json.dumps(json.load(response), indent=2))
PY
```

### 4.7 Inspect MinIO

Open:

```text
http://localhost:9001
```

Local credentials from `docker-compose.yml`:

```text
Username: arcminio
Password: arcminio-secret
```

The `arc-human-games` bucket should contain keys shaped like:

```text
raw/aa00/<sha-prefix>/aa00__filename.recording.jsonl
```

These local credentials are only for local development. Change them before exposing MinIO outside your machine.

### 4.8 Stop, restart, or reset

Stop containers while preserving data:

```bash
docker compose down
```

Restart later:

```bash
export DATASET_PATH=/arc-prize-2026-arc-agi-3/public-games-dataset
docker compose up -d
```

Remove containers and all local PostgreSQL/MinIO volumes:

```bash
docker compose down -v
```

`down -v` permanently deletes the local indexed metadata and uploaded MinIO copy. The source dataset directory is read-only and is not deleted.

---

## 5. Ingestion behavior and dataset updates

### Idempotent reruns

The importer identifies an existing item by `(source relative path, SHA-256)`.

Rerunning the same command safely produces `SKIP` lines:

```bash
docker compose exec api \
  python -m app.ingest /data/public-games-dataset
```

### Force an object re-upload

```bash
docker compose exec api \
  python -m app.ingest /data/public-games-dataset --force
```

`--force` re-uploads matching objects. It is not normally required.

### Changed files

When the content of a file changes, its SHA-256 changes, so the importer creates a new indexed recording and object. This preserves reproducibility.

### New files

Add them to the game folders and rerun ingestion. Only new or changed content is uploaded.

### Deleted source files

Deleting a local source file does not remove its existing object or database row. This version intentionally has no automatic prune command. Delete hosted data manually only after confirming that no training manifest depends on it.

---

## 6. API usage

All metadata endpoints return JSON. Recording downloads return `application/x-ndjson`.

### 6.1 Dataset totals

```bash
curl -s http://localhost:8000/v1/stats
```

Example shape:

```json
{
  "games": 25,
  "recordings": 300,
  "total_bytes": 4000000000,
  "total_transitions": 250000
}
```

The numbers above are illustrative; your API returns actual totals.

### 6.2 List games

```bash
curl -s http://localhost:8000/v1/games
```

### 6.3 List recordings

```bash
curl -s \
  'http://localhost:8000/v1/recordings?game_code=aa00&limit=100&offset=0'
```

`limit` is capped by `MAX_PAGE_SIZE`, which defaults to `500`.

### 6.4 Get one recording

First copy an `id` from the list response:

```bash
curl -s \
  http://localhost:8000/v1/recordings/RECORDING_ID
```

### 6.5 Download one recording through FastAPI

```bash
curl -fL \
  -o recording.jsonl \
  http://localhost:8000/v1/recordings/RECORDING_ID/download
```

Authenticated version:

```bash
curl -fL \
  -H "X-API-Key: $READ_API_KEY" \
  -o recording.jsonl \
  http://localhost:8000/v1/recordings/RECORDING_ID/download
```

### 6.6 Check size and range support

```bash
curl -I \
  http://localhost:8000/v1/recordings/RECORDING_ID/download
```

The API exposes `Accept-Ranges: bytes`.

Download only the first 1,024 bytes:

```bash
curl -fL \
  -H 'Range: bytes=0-1023' \
  http://localhost:8000/v1/recordings/RECORDING_ID/download
```

Resume an interrupted download:

```bash
curl -fL -C - \
  -o recording.jsonl \
  http://localhost:8000/v1/recordings/RECORDING_ID/download
```

### 6.7 Obtain a direct signed object-store URL

```bash
curl -s \
  http://localhost:8000/v1/recordings/RECORDING_ID/signed-url
```

The response contains `url` and `expires_in`. The default expiration is 900 seconds.

Signed URLs reduce FastAPI bandwidth because the client downloads directly from object storage. Browser use requires object-store CORS configuration and a publicly reachable `S3_PUBLIC_ENDPOINT_URL`.

### 6.8 Full manifest

```bash
curl -s \
  -o manifest.json \
  http://localhost:8000/v1/dataset/manifest
```

Pin this manifest with training runs. Each recording includes a SHA-256 checksum for byte-level verification.

---

## 7. Python training client

### 7.1 Create a virtual environment

From the project root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-client.txt
```

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-client.txt
```

### 7.2 Stream events without loading a complete file

```python
from clients.python_client import ArcDatasetClient

with ArcDatasetClient(
    "http://localhost:8000",
    api_key=None,  # or your READ_API_KEY
) as dataset:
    print(dataset.games())

    for recording in dataset.recordings("aa00"):
        print(recording["id"], recording["sha256"])

        for event in dataset.events(recording["id"]):
            data = event.get("data", {})
            frame = data.get("frame")
            action = data.get("action_input")
            if frame is not None:
                # train_step(frame, action)
                pass
```

`events()` parses one JSON line at a time and does not keep the full recording in memory.

### 7.3 Download to a local training cache

```python
from pathlib import Path

from clients.python_client import ArcDatasetClient

cache = Path("training-cache")
cache.mkdir(exist_ok=True)

with ArcDatasetClient("http://localhost:8000") as dataset:
    for recording in dataset.recordings("aa00"):
        destination = cache / recording["file_name"]
        dataset.download(recording["id"], str(destination))
```

For reproducible training, verify each downloaded file against `recording["sha256"]`.

```python
import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
```

---

## 8. Browser frontend integration

Copy `clients/frontend.ts` into your frontend source tree or import it from this project.

```ts
import { ArcDatasetApi } from "./frontend";

const dataset = new ArcDatasetApi("http://localhost:8000");
const recordings = await dataset.listRecordings("aa00");

for await (const event of dataset.streamRecording(recordings[0].id)) {
  console.log(event);
}
```

The local Compose configuration allows these frontend origins:

```text
http://localhost:3000
http://localhost:5173
```

To use another origin, edit `ALLOWED_ORIGINS` in `docker-compose.yml`, then recreate the API container:

```bash
docker compose up -d --build --force-recreate api
```

### Signed URL download in the browser

```ts
const url = await dataset.signedDownloadUrl(recordings[0].id);
const response = await fetch(url);
if (!response.ok) throw new Error(`Object download failed: ${response.status}`);
```

The object-store bucket must permit CORS for your frontend origin when the browser accesses signed URLs directly. At minimum, allow:

- Origins: your exact frontend origins.
- Methods: `GET`, `HEAD`.
- Request headers: `Range` when resumable downloads are needed.
- Exposed response headers: `Content-Length`, `Content-Range`, `ETag`, `Accept-Ranges`.

### API keys in frontend code

Do not ship a permanent `READ_API_KEY` in public JavaScript. Users can inspect bundled code and network requests. For a private application, use one of these designs:

- Route dataset calls through your authenticated backend-for-frontend.
- Put the API behind an identity-aware reverse proxy or gateway.
- Exchange authenticated user sessions for short-lived gateway tokens.

---

## 9. Native API installation without Docker

Use this method when PostgreSQL and S3-compatible storage already exist.

### 9.1 Install Python

Python 3.12 is recommended because the supplied Docker image uses it.

```bash
cd /path/to/arc-dataset-host
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 9.2 Configure environment

```bash
cp .env.example .env
```

Edit `.env` with real values. Minimum required settings are:

```dotenv
DATABASE_URL=postgresql+psycopg://arc:password@postgres.example:5432/arc_dataset
S3_BUCKET=arc-human-games
S3_REGION=us-east-1
S3_ENDPOINT_URL=https://s3-compatible-endpoint.example
S3_ACCESS_KEY_ID=...
S3_SECRET_ACCESS_KEY=...
ALLOWED_ORIGINS=http://localhost:3000
```

For AWS S3, leave `S3_ENDPOINT_URL` empty.

### 9.3 Start the API

Development mode:

```bash
uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --reload
```

Production-like single process:

```bash
uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --proxy-headers
```

Health check:

```bash
curl -s http://localhost:8000/healthz
```

### 9.4 Run native ingestion

```bash
python -m app.ingest \
  /arc-prize-2026-arc-agi-3/public-games-dataset
```

The importer uses the same `.env` file as the API.

---

## 10. Production deployment

A production deployment needs:

1. Managed or self-hosted PostgreSQL.
2. A private S3-compatible bucket.
3. A container platform or VM for FastAPI.
4. HTTPS and an authentication/gateway strategy appropriate for the data.

### 10.1 Create PostgreSQL databases

Create one database for dataset metadata:

```sql
CREATE USER arc WITH PASSWORD 'replace-me';
CREATE DATABASE arc_dataset OWNER arc;
```

For Optuna, use a separate database or at least a separate PostgreSQL schema and credentials:

```sql
CREATE USER optuna WITH PASSWORD 'replace-me-too';
CREATE DATABASE optuna OWNER optuna;
```

The API currently runs SQLAlchemy `create_all()` at startup. This is sufficient for the supplied initial schema. Use Alembic or your organization’s migration system before evolving the schema in production.

### 10.2 Create the object-storage bucket

Create a private bucket, for example:

```text
arc-human-games
```

The ingestion identity needs enough access to:

- Check the bucket and object existence.
- Upload objects, including multipart uploads for larger JSONL files.
- Read objects.
- Abort incomplete multipart uploads.

The API runtime identity only needs read access when ingestion uses a separate identity. Because the API also supports bucket creation when `S3_CREATE_BUCKET=true`, leave that setting `false` in production unless bucket creation is intentionally delegated to the application.

### 10.3 Production environment file

Create `.env` from `.env.example`:

```dotenv
DATABASE_URL=postgresql+psycopg://arc:STRONG_PASSWORD@db.example.net:5432/arc_dataset

S3_BUCKET=arc-human-games
S3_REGION=us-east-1
S3_ENDPOINT_URL=
S3_PUBLIC_ENDPOINT_URL=
S3_ACCESS_KEY_ID=...
S3_SECRET_ACCESS_KEY=...
S3_FORCE_PATH_STYLE=false
S3_CREATE_BUCKET=false

READ_API_KEY=LONG_RANDOM_SECRET
ALLOWED_ORIGINS=https://frontend.example
SIGNED_URL_TTL_SECONDS=900
STREAM_CHUNK_BYTES=1048576
MAX_PAGE_SIZE=500
```

Protect this file:

```bash
chmod 600 .env
```

Prefer your deployment platform’s secret manager instead of copying `.env` into an image.

### 10.4 S3 provider settings

#### AWS S3

```dotenv
S3_REGION=us-east-1
S3_ENDPOINT_URL=
S3_PUBLIC_ENDPOINT_URL=
S3_FORCE_PATH_STYLE=false
```

#### Self-hosted or managed MinIO

```dotenv
S3_REGION=us-east-1
S3_ENDPOINT_URL=https://minio-internal.example
S3_PUBLIC_ENDPOINT_URL=https://objects.example
S3_FORCE_PATH_STYLE=true
```

`S3_ENDPOINT_URL` must be reachable from the API/importer. `S3_PUBLIC_ENDPOINT_URL` is used only to generate signed URLs and must be reachable from clients. Omit the public endpoint when clients will only use the FastAPI `/download` proxy.

#### Other S3-compatible providers

Use the provider’s S3 API endpoint, region convention, credentials, and addressing style. Path-style addressing is common for MinIO; hosted providers may require virtual-hosted addressing.

### 10.5 Build the image

```bash
docker build -t arc-dataset-host:1.0 .
```

### 10.6 Run the production API on a VM

```bash
docker run -d \
  --name arc-dataset-api \
  --restart unless-stopped \
  --env-file .env \
  -p 127.0.0.1:8000:8000 \
  arc-dataset-host:1.0
```

Binding to `127.0.0.1` assumes Nginx, Caddy, Traefik, or another TLS reverse proxy runs on the same machine.

Inspect logs:

```bash
docker logs -f --tail=100 arc-dataset-api
```

Check health:

```bash
curl -s http://127.0.0.1:8000/healthz
```

### 10.7 Ingest from a trusted machine

The ingestion machine must be able to read the dataset folder and connect to PostgreSQL and object storage.

```bash
export DATASET_PATH=/arc-prize-2026-arc-agi-3/public-games-dataset

docker run --rm \
  --env-file .env \
  -v "$DATASET_PATH:/dataset:ro" \
  arc-dataset-host:1.0 \
  python -m app.ingest /dataset
```

The deployed API container does not need the dataset volume after this command finishes.

### 10.8 Reverse proxy requirements

Your reverse proxy should:

- Terminate HTTPS.
- Preserve the `Range` request header.
- Preserve `Content-Range`, `Accept-Ranges`, `Content-Length`, and `ETag` response headers.
- Disable response buffering for large streaming downloads when practical.
- Use sufficiently long upstream read timeouts for slower downloads.
- Forward the original scheme and client information; Uvicorn already starts with `--proxy-headers` in the Dockerfile.

A `/healthz` probe does not require API authentication.

### 10.9 Horizontal scaling

The API is stateless. Multiple replicas can share the same PostgreSQL database and S3 bucket. Place them behind a load balancer. No sticky sessions are required.

Ingestion should normally run as a separate one-off job, not in every API replica.

---

## 11. Configuration reference

| Variable | Meaning | Default |
|---|---|---|
| `DATABASE_URL` | SQLAlchemy PostgreSQL connection URL | Local Compose URL |
| `S3_BUCKET` | Bucket containing raw JSONL objects | `arc-human-games` |
| `S3_REGION` | S3 signing/region value | `us-east-1` |
| `S3_ENDPOINT_URL` | Internal S3-compatible API endpoint | Empty/AWS default |
| `S3_PUBLIC_ENDPOINT_URL` | Client-reachable endpoint used to sign direct URLs | Uses internal endpoint |
| `S3_ACCESS_KEY_ID` | Object-store access key | Empty/provider chain |
| `S3_SECRET_ACCESS_KEY` | Object-store secret key | Empty/provider chain |
| `S3_FORCE_PATH_STYLE` | Use `/bucket/key` addressing | `false` |
| `S3_CREATE_BUCKET` | Let the app create a missing bucket | `false` |
| `READ_API_KEY` | Optional shared API read key | Empty/public |
| `ALLOWED_ORIGINS` | Comma-separated or JSON-array CORS origins | `http://localhost:3000` |
| `SIGNED_URL_TTL_SECONDS` | Signed URL lifetime, 60–86400 seconds | `900` |
| `STREAM_CHUNK_BYTES` | API proxy stream chunk size | `1048576` |
| `MAX_PAGE_SIZE` | Maximum recordings returned per page | `500` |
| `OPTUNA_STORAGE_URL` | Optuna RDB storage URL | None |
| `ARC_DATASET_API_URL` | Dataset API used by Optuna worker | `http://localhost:8000` |
| `ARC_DATASET_API_KEY` | Dataset key used by Optuna worker | Empty |

`ALLOWED_ORIGINS` accepts either:

```dotenv
ALLOWED_ORIGINS=https://app.example,https://admin.example
```

or:

```dotenv
ALLOWED_ORIGINS=["https://app.example","https://admin.example"]
```

---

## 12. Optional Optuna integration

Optuna workers are training processes that consume the hosted dataset API. They do not need direct bucket access.

### 12.1 Install dependencies

```bash
python3.12 -m venv .venv-optuna
source .venv-optuna/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-optuna.txt
```

### 12.2 Configure the study

```bash
export OPTUNA_STORAGE_URL='postgresql+psycopg://optuna:PASSWORD@db.example.net:5432/optuna'
export ARC_DATASET_API_URL='https://dataset-api.example'
export ARC_DATASET_API_KEY='YOUR_READ_API_KEY'
```

For the public local API:

```bash
export OPTUNA_STORAGE_URL='postgresql+psycopg://optuna:PASSWORD@localhost:5432/optuna'
export ARC_DATASET_API_URL='http://localhost:8000'
unset ARC_DATASET_API_KEY
```

### 12.3 Run the included connectivity objective

```bash
python -m experiments.optuna_runner \
  --objective experiments.example_objective:objective \
  --study-name arc-smoke-test \
  --trials 10 \
  --direction maximize
```

The included objective is intentionally cheap. It samples recordings and maximizes sampled action-distribution entropy. It confirms that Optuna, the API, streaming JSONL parser, and PostgreSQL study storage work together.

### 12.4 Replace it with model training

Create a module such as `experiments/train_model.py`:

```python
import optuna

from clients.python_client import ArcDatasetClient


def objective(trial: optuna.Trial, dataset: ArcDatasetClient) -> float:
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    history = trial.suggest_int("history", 1, 32)
    hidden_size = trial.suggest_categorical("hidden_size", [128, 256, 512])

    # Build streaming/cached loaders using dataset.recordings() and dataset.events().
    # model = build_model(hidden_size=hidden_size, history=history)
    # train(model, dataset, learning_rate=learning_rate)
    # validation_score = evaluate(model)

    validation_score = 0.0  # Replace with the real metric.
    return validation_score
```

Run it:

```bash
python -m experiments.optuna_runner \
  --objective experiments.train_model:objective \
  --study-name arc-model-v1 \
  --trials 100 \
  --direction maximize
```

### 12.5 Parallel Optuna workers

Start the same command on multiple machines with the same `--study-name` and `OPTUNA_STORAGE_URL`. Optuna coordinates trials through its database. Each worker independently downloads or streams training data.

For expensive training, cache recordings locally by SHA-256 so repeated trials do not repeatedly transfer identical files.

---

## 13. Tests

Install development dependencies:

```bash
python3.12 -m venv .venv-dev
source .venv-dev/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

Run tests:

```bash
pytest -q
```

Inspect an individual recording:

```bash
python - <<'PY'
from pprint import pprint

from app.parser import inspect_jsonl

pprint(inspect_jsonl("/path/to/example.recording.jsonl"))
PY
```

Validate all files without uploading them:

```bash
python - <<'PY'
from pathlib import Path

from app.parser import inspect_jsonl

root = Path("/arc-prize-2026-arc-agi-3/public-games-dataset")
failures = 0
for path in sorted(root.rglob("*.jsonl")):
    try:
        metadata = inspect_jsonl(path)
        print("OK", path.relative_to(root), metadata.record_count)
    except Exception as exc:
        failures += 1
        print("FAIL", path.relative_to(root), exc)

raise SystemExit(1 if failures else 0)
PY
```

---

## 14. Security checklist

Before exposing the service publicly:

- Use HTTPS.
- Keep the object bucket private.
- Set a long random `READ_API_KEY`, or use a proper identity-aware gateway.
- Do not embed a permanent private key in browser JavaScript.
- Restrict `ALLOWED_ORIGINS` to exact frontend origins.
- Use separate ingestion and API object-store identities when possible.
- Give the API identity read-only bucket permissions.
- Store database and object-store credentials in a secret manager.
- Restrict PostgreSQL network access to API/importer/Optuna hosts.
- Enable PostgreSQL backups and bucket versioning or replication.
- Rotate credentials and restart/redeploy services after rotation.
- Add rate limits at the reverse proxy or API gateway for public deployments.

`READ_API_KEY` is intentionally simple shared-key protection. It is not user-level authorization, quota management, or identity federation.

---

## 15. Backups and reproducibility

Back up both layers:

### PostgreSQL metadata

```bash
pg_dump "$DATABASE_URL" > arc_dataset_metadata.sql
```

Restore into an empty database:

```bash
psql "$DATABASE_URL" < arc_dataset_metadata.sql
```

### Object storage

Use the provider’s bucket replication, versioning, lifecycle, and backup tooling. PostgreSQL backup alone is not sufficient because it stores object keys and metadata, not the raw recording bytes.

### Training manifests

Save `/v1/dataset/manifest` with each important training run. It records exact SHA-256 values and object metadata, allowing the run to pin a stable dataset version even after new recordings are ingested.

---

## 16. Troubleshooting

### `No .jsonl files found`

The path inside the container is wrong or the host path was not mounted.

```bash
echo "$DATASET_PATH"
docker compose exec api \
  find /data/public-games-dataset -maxdepth 3 -type f -name '*.jsonl' | head
```

On Windows, verify Docker Desktop drive sharing or use WSL2.

### API statistics stay at zero

The API started correctly, but ingestion has not completed against the same `DATABASE_URL` and bucket.

```bash
docker compose exec api \
  python -m app.ingest /data/public-games-dataset
```

### PostgreSQL connection refused

```bash
docker compose ps
docker compose logs postgres
docker compose logs api
```

For managed PostgreSQL, check hostname, port, firewall rules, TLS requirements, user permissions, and URL escaping for special password characters.

### S3 `AccessDenied`

Confirm:

- Bucket name.
- Access key and secret.
- Region and endpoint.
- Bucket policy/IAM policy.
- Multipart upload permissions.
- Whether path-style addressing is required.

### Signed URL points to an internal hostname

Set a client-reachable endpoint:

```dotenv
S3_PUBLIC_ENDPOINT_URL=https://objects.example
```

Then restart the API. The public endpoint must route to the same object store and use credentials/signature behavior compatible with the internal endpoint.

### Signed URL works in `curl` but not in the browser

This is usually bucket CORS. Allow the frontend origin, `GET`/`HEAD`, and the headers required for range downloads.

### Browser receives a CORS error from FastAPI

Add the exact browser origin to `ALLOWED_ORIGINS`. Origin matching includes scheme and port.

Correct:

```dotenv
ALLOWED_ORIGINS=https://app.example,http://localhost:5173
```

A path such as `https://app.example/page` is not a valid CORS origin.

### API returns `401 Missing or invalid API key`

Send:

```text
X-API-Key: configured-value
```

The health endpoint `/healthz` remains public.

### Invalid JSON during ingestion

The importer reports the exact file and line. Repair or replace the source file, then rerun ingestion.

### An interrupted ingestion run

Rerun the same command. Completed files are skipped; incomplete or missing objects are processed again.

### `Requested byte range is outside the object`

The client requested bytes beyond the recording size. Check `Content-Length` with a `HEAD` request and restart the download with a valid offset.

### Docker containers start but API cannot reach MinIO

Inside Compose, the API endpoint must remain:

```text
http://minio:9000
```

`localhost:9000` from inside the API container would point back to the API container, not MinIO. The separate `S3_PUBLIC_ENDPOINT_URL=http://localhost:9000` is only for URLs returned to the host browser.

### Configuration changes are ignored

The settings object is loaded at process startup. Recreate or restart the API container after changing environment variables:

```bash
docker compose up -d --force-recreate api
```

---

## 17. Recommended first run

For a clean local validation, execute these commands in order:

```bash
unzip arc-dataset-host.zip
cd arc-dataset-host

export DATASET_PATH=/arc-prize-2026-arc-agi-3/public-games-dataset

docker compose up -d --build
docker compose ps

curl -s http://localhost:8000/healthz

docker compose exec api \
  python -m app.ingest /data/public-games-dataset

curl -s http://localhost:8000/v1/stats
curl -s http://localhost:8000/v1/games

# Open API documentation:
# http://localhost:8000/docs
# Open MinIO console:
# http://localhost:9001
```

When these steps succeed, point your frontend at `http://localhost:8000`, or deploy the same image with managed PostgreSQL and S3-compatible storage using the production steps above.
