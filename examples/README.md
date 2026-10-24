# Examples

| File | What it shows |
| --- | --- |
| `local_demo.py` | `python -m examples.local_demo`: starts `fake_backend.py` and the router on loopback ports, sends Anthropic, OpenAI and streaming requests plus an explain call, prints the routes and the savings report. No models, no network. |
| `fake_backend.py` | A stand-in OpenAI-compatible server (JSON + SSE) that answers with the model name and effort it received. `uvicorn examples.fake_backend:app --port 11434` lets you try `agent-model-router serve` without ollama. |
| `claude-only.yaml` | A pool of hosted Claude models (your key) with adaptive effort: the fully supported setup for Claude Code. |
| `mixed.yaml` | Local ollama for trivial and easy turns, a LAN vLLM box for medium ones, hosted Claude for hard ones, a daily budget and a local decision model. |

With ollama installed:

```bash
ollama pull qwen3:1.7b && ollama pull qwen3:8b && ollama pull qwen3-coder:30b
agent-model-router serve
curl -s http://127.0.0.1:8787/router/explain -H "x-api-key: $(agent-model-router token)" \
  -d '{"messages":[{"role":"user","content":"rename foo to bar in a.py"}]}' | jq .model
```
