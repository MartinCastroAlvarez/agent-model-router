# Security

agent-model-router sits between a coding agent and the models it talks to. It sees
every prompt, including code and anything a developer pastes. It holds API keys for
hosted backends, and other people install it. Design goal: nothing a client, a web page,
a prompt or a model sends can make the router send data to a host the operator did not
configure, leak a key or a prompt, or run code.

## Reporting a vulnerability

Open a private security advisory at
<https://github.com/MartinCastroAlvarez/agent-model-router/security/advisories/new>.
Do **not** open a public issue for a suspected vulnerability. We acknowledge within 5
business days and aim to ship a fix within 30 days. You are credited in the advisory
unless you ask otherwise.

Supported versions: the latest minor release.

## 1. Threat model

**Assets.** Prompts and code that pass through the router, the operator's provider keys,
the router token, the local machine and network (SSRF), and the operator's bill.

**Actors.**

| Actor | Controls | Must not be able to |
| --- | --- | --- |
| Another local process or user | HTTP requests to 127.0.0.1 | use the router (and the operator's keys) without the token |
| A web page in the developer's browser | requests to `localhost`, DNS rebinding | reach the API (Host/Origin checks), read responses |
| The prompt author / pasted content | message text, tool results | pick a backend URL, read keys, push a hard task onto a weak model unnoticed, send secrets to a hosted provider |
| The decision model | its JSON answer | override guards, lower a route by more than one bucket |
| A backend | its responses, redirects | receive the client's credentials, redirect the router's key elsewhere, inject headers into the client response |

**Threats and controls.**

| Threat | Control |
| --- | --- |
| Unauthorised local use | A random 256-bit token (`amr-…`) is created in a 0600 file, or read from `AGENT_ROUTER_TOKEN`. Every endpoint except `/healthz` requires it (`x-api-key` or `Bearer`), compared in constant time. `auth: false` is accepted only on a loopback bind. |
| Exposure to the network | Binds to `127.0.0.1`; any other host needs `server.allow_remote: true` (config and CLI both check). |
| DNS rebinding / browsers | On a loopback bind the `Host` header must be loopback, or the request gets 403. Any `Origin` header is refused unless allowlisted (none by default). No CORS headers are sent. |
| SSRF | Backend URLs come from the config file only. No request field, header or model name can add a URL. URLs must be `http(s)://host[:port][/path]` with no credentials, query or fragment. Plain HTTP is allowed only to loopback, or with `allow_insecure: true` per backend. Hosted kinds require https. Redirects are never followed (a key can't follow a 302), and proxy environment variables are ignored. Image URLs inside messages are passed on as data. The router never fetches them, and only `https://` image URLs are kept in translation. |
| Credential leakage | The client's `Authorization`/`x-api-key` (the router token) is never forwarded. Each backend gets the operator's key from `api_key_env`. Outgoing headers come from an allowlist (`anthropic-version`, `anthropic-beta` same dialect only, `accept-language`, `openai-organization`/`project` same dialect only). Static `headers` in the config can't set `Authorization`, `x-api-key`, `Cookie` or `Host`. Responses pass back only `retry-after`, `request-id` and `x-request-id` (no `set-cookie`). |
| Secrets in prompts | Secret-shaped text (provider keys, GitHub/Slack tokens, AWS key ids, JWTs, private keys, `password=`) marks the request `secret`. It is then routed only to `trusted` backends (loopback by default), and its fallback chain stays there. The decision model only ever sees a redacted copy. |
| Prompt and key leakage at rest | The router never logs or stores prompt text, tool arguments, headers or keys. Log lines and SQLite rows hold the hashed session id, task, bucket, model, effort, status, token counts, cost and latency. The SQLite file is created 0600. Upstream error bodies come back as at most 300 characters of the provider's `error.message`. |
| Classifier manipulation (prompt injection) | Routing output is advice. Sensitive-topic guards only raise the bucket. The decision model reads the request inside `<request>` tags with an instruction to treat it as data, and its answer can lower the heuristic's bucket by at most one step. The keyword heuristic itself can be nudged by words like "simple" or "quick". Pin hard work with `x-router-model` if you care, and remember the worst case is a cheaper model, not a different destination. |
| Model output | Responses are relayed to the client, which is the agent's own trust boundary. The router doesn't execute, parse into code, or act on model output. It only reads `stop_reason`/usage and checks for refusal phrases. |
| Resource exhaustion / bill | Bodies are limited to 32 MiB (`server.max_body_bytes`), counted while streaming. Each backend has connect and read timeouts. The fallback chain is capped (`max_attempts`). A daily USD budget routes to the cheapest model or rejects with 429. The session store is a bounded LRU. |
| Code execution | No `eval`, `exec`, `pickle`, `marshal`, `subprocess` or `shell=True`. YAML is read with `yaml.safe_load`, weights are JSON, and all SQL is parameterised. An AST test enforces this, and also that no `noqa`/`nosec`/`type: ignore` hides anything. |

## 2. Running it safely

- Keep the default loopback bind. If you must expose it, put it behind your own TLS and
  auth, and set `allow_remote: true` knowingly.
- Mark a LAN backend `trusted: true` only if prompts with secrets may go there.
- Treat `quality` values and the effort table as policy: tune them on your own
  prompts (`agent-model-router eval --data`), and pin critical work.
- Set `budgets.daily_usd` when a hosted backend is enabled.

## 3. What this project never does

It never calls a hosted model in its tests (fake backends only), ships no telemetry, and
contacts only the backends in your config.
