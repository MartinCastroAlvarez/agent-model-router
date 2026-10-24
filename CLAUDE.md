# CLAUDE.md

Contract between this repository and any AI agent contributing to it.

## Required reading

1. This file. 2. [`README.md`](README.md). 3. [`ARCHITECTURE.md`](ARCHITECTURE.md).
4. [`SECURITY.md`](SECURITY.md).

## Mission

`agent-model-router` picks the model and reasoning effort for each coding-agent request,
locally, explainably and without leaking prompts or keys.

## The rules

1. **Prompts are never logged or stored.** Not in log lines, SQLite, exceptions or
   test fixtures that write to disk. Hash ids only.
2. **Backends come from the config.** Never let a request choose a URL, host or header
   value that reaches a backend; never forward the client's credentials.
3. **Routing output is advice.** Guards may only raise a route; the decision model may
   lower the heuristic by at most one bucket; secrets stay on trusted backends.
4. **Retry only before the first byte.** Once a stream started, it is committed.
5. **Offline tests.** Fake backends (`MockTransport`), respx and loopback servers only.
   Never a real model or API.
6. Every behaviour change gets a test (a hostile one when it touches security), plus
   README and CHANGELOG entries. Classifier changes regenerate the dataset and weights.
7. Never add `# noqa`, `# nosec` or `# type: ignore` to the package, skip a test, or
   loosen lint/type/coverage config. Fix the code.

## Done means

`poetry run pytest` (coverage >= 85%), `ruff check`, `ruff format --check`,
`mypy` (strict), `bandit`, `pip-audit` all green; `python -m examples.local_demo` runs;
docs and CHANGELOG updated.

## Git

Small commits, imperative subjects, never force-push `main`, never commit secrets,
`.env`, route logs (`*.sqlite3`) or the token file.
