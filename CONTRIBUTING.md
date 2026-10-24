# Contributing to agent-model-router

## TL;DR

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install poetry
poetry install
poetry run pytest
poetry run ruff check agent_model_router tests examples && poetry run ruff format --check agent_model_router tests examples
poetry run mypy agent_model_router
poetry run bandit -c pyproject.toml -q -r agent_model_router
poetry run pip-audit
poetry run python -m examples.local_demo
```

Open a PR against `main` from a `feat/…`, `fix/…`, `docs/…` or `chore/…` branch.

## What this package will not accept

- Tests that call a real model, a hosted API or the network (fake backends, respx and
  loopback servers only).
- Logging or storing prompt text, tool arguments, headers or keys, even at debug level.
- Backend URLs, hosts or headers taken from a request.
- Forwarding the client's credentials to a backend.
- Following redirects with a key, or reading proxy settings from the environment.
- Hidden network calls (telemetry, update checks, model downloads).

## What it does accept

- New backend dialects (behind `Backend.api`) with translation and streaming tests.
- New features for the scorer, with a regenerated dataset/weights and the eval numbers
  in the PR description.
- Better labelled prompts in `agent_model_router/data/eval.jsonl` (no real code or data).

## Changing the classifier

1. Edit `classifier/features.py` / `classifier/dataset.py`.
2. Regenerate: `python -c "from pathlib import Path; from agent_model_router.classifier.dataset import write_dataset; write_dataset(Path('agent_model_router/data/synthetic.jsonl'))"`
   and `agent-model-router train --out agent_model_router/data/weights.json`.
3. Update the numbers in the README's "Classifier quality" table.

## House rules

1. Tests alongside features; security-relevant changes get a hostile case.
2. README, ARCHITECTURE/SECURITY (when relevant) and CHANGELOG change with the code.
3. No secrets in commits; `.env.example` holds placeholders only.

## Releasing

1. Bump `version` in `pyproject.toml` and `__version__` in
   `agent_model_router/__init__.py`, and move `Unreleased` in `CHANGELOG.md` under the
   new version.
2. Merge, tag the merge commit `vX.Y.Z`, push the tag. `publish.yml` checks
   tag == version, runs the tests, scans the artefacts and publishes through
   PyPI Trusted Publishing (environment `pypi`), then creates the GitHub Release.
   The PyPI pending publisher must be registered first
   (owner `MartinCastroAlvarez`, repository `agent-model-router`, workflow `publish.yml`, environment `pypi`).
