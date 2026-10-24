# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### To do before the first PyPI release
- Register the PyPI pending publisher for `publish.yml` (environment `pypi`),
  then tag `v0.1.0`.

## [0.1.0] — unreleased

### Added
- Local proxy (Starlette + httpx, async) speaking Anthropic Messages (`/v1/messages`,
  `count_tokens` estimate) and OpenAI Chat Completions (`/v1/chat/completions`) inbound,
  and translating to the backend's dialect, streaming included (byte-for-byte
  passthrough or event-level SSE translation, tool calls both ways).
- Backends: ollama (default), any OpenAI-compatible server (llama.cpp, vLLM, LM Studio),
  and hosted Anthropic with the operator's own key (disabled by default).
- Heuristic classifier: task rules (chat/edit/plan/debug/review/tool_followup) plus an
  18-feature logistic scorer trained on a bundled, deterministic synthetic dataset, with
  fitted bucket cut points, sensitive-topic floors and secret detection.
- Optional decision-model mode: a small local model judges a scrubbed copy of the
  request; its answer is advice (fallback, floors, at most one bucket down).
- Policy: cheapest model that clears the bucket's quality bar; context, tools, budget and
  trusted-backend filters; ratchet stickiness per conversation; tool-loop inheritance;
  idle reset; pins; fallback chain; effort per bucket (`reasoning_effort`, adaptive
  thinking + `output_config.effort`, thinking budgets).
- Escalation on connection errors, timeouts, 408/409/425/429/5xx/529 and refusals.
- `/router/explain`, `/router/stats`, `x-router-*` response headers, SQLite route log
  and `agent-model-router stats` with savings against the baseline model.
- Security: loopback bind, local token, Host/Origin checks, header allowlists, SSRF-safe
  config-only backend URLs, no redirects, no prompt or key logging.
- CLI: `serve`, `init`, `token`, `explain`, `stats`, `train`, `eval`.
