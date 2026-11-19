# Multi-Model Kidney Tumor AI

Monorepo for a Next.js web app and a Python API.

## Repository layout

```text
apps/
  api/   Python API (uv)
  web/   Next.js app (pnpm)
```

Use **pnpm** for the web app and **uv** for the API. They have separate dependency manifests and lockfiles.

## Requirements

- Node.js and pnpm 11.22.0
- Python 3.14 and uv (the API's `.python-version` specifies 3.14; its project supports Python 3.11+)

## Web app

From the repository root:

```sh
pnpm install
pnpm dev:web
```

Open <http://localhost:3000>. To build or lint:

```sh
pnpm build:web
pnpm lint:web
```

## API

Install the API dependencies from the repository root:

```sh
uv sync --project apps/api
uv run --project apps/api uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

The API exposes `POST /analyze` and loads the CT and MRI model weights from `apps/api/src/models/`.

## Current status and next steps

- Set `NEXT_PUBLIC_API_URL` in the web app environment when the API is not available at `http://localhost:8000`.
- Add Dockerfiles and compose configuration if containerized local development or deployment is required.
- Add tests and CI checks for the API and web app.

## Dependency files

- `apps/web/package.json`: web dependencies and scripts
- `apps/api/pyproject.toml`: API dependencies and Python project configuration
- `apps/api/uv.lock`: locked API dependencies
- Root `package.json` and `pnpm-workspace.yaml`: pnpm workspace and web scripts
