"""Command line: ``agent-model-router serve | init | token | explain | stats | train | eval``."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from collections.abc import Sequence
from importlib import resources
from pathlib import Path

from agent_model_router import __version__
from agent_model_router.config import Config
from agent_model_router.config import ConfigError
from agent_model_router.config import default_config_path
from agent_model_router.config import load_config

__all__ = ["main"]


def _config(path: str | None) -> Config:
    return load_config(path)


def _router_key(config: Config) -> str:
    from agent_model_router.security import load_or_create_token

    return load_or_create_token(config.server.key_env, Path(config.server.key_file))


def cmd_serve(args: argparse.Namespace) -> int:
    import dataclasses

    import uvicorn

    from agent_model_router.proxy import create_app
    from agent_model_router.security import is_loopback_host

    config = _config(args.config)
    server = config.server
    if args.host or args.port:
        server = dataclasses.replace(
            server, host=args.host or server.host, port=args.port or server.port
        )
        config = dataclasses.replace(config, server=server)
    if not is_loopback_host(server.host) and not server.allow_remote:
        print(
            "refusing to bind a non-loopback address without server.allow_remote: true",
            file=sys.stderr,
        )
        return 2
    token = _router_key(config) if server.auth else None
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    app = create_app(config, token=token)
    base = f"http://{server.host}:{server.port}"
    print(f"agent-model-router {__version__} on {base}", file=sys.stderr)
    print(
        f"  Claude Code:  ANTHROPIC_BASE_URL={base} "
        "ANTHROPIC_AUTH_TOKEN=$(agent-model-router token) claude",
        file=sys.stderr,
    )
    print(
        f"  OpenAI SDKs:  base_url={base}/v1, api_key=$(agent-model-router token)", file=sys.stderr
    )
    uvicorn.run(app, host=server.host, port=server.port, access_log=False, log_level="warning")
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    target = Path(args.path).expanduser() if args.path else default_config_path()
    if target.exists() and not args.force:
        print(f"{target} exists (use --force to overwrite)", file=sys.stderr)
        return 1
    target.parent.mkdir(parents=True, exist_ok=True)
    text = resources.files("agent_model_router").joinpath("data/default.yaml").read_text("utf-8")
    target.write_text(text, encoding="utf-8")
    print(str(target))
    return 0


def cmd_token(args: argparse.Namespace) -> int:
    print(_router_key(_config(args.config)))
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    from agent_model_router.backends import Upstream
    from agent_model_router.classifier.heuristic import HeuristicClassifier
    from agent_model_router.classifier.linear import LinearScorer
    from agent_model_router.policy import NoRoute
    from agent_model_router.policy import select
    from agent_model_router.proxy import _build_classifier
    from agent_model_router.view import view_request

    config = _config(args.config)
    raw = sys.stdin.read() if args.file == "-" else Path(args.file).read_text("utf-8")
    if not raw.lstrip().startswith("{"):
        raw = json.dumps({"messages": [{"role": "user", "content": raw}]})
    body = json.loads(raw)
    view = view_request(args.api, body)
    if args.offline or config.classifier.mode == "heuristic":
        classifier = HeuristicClassifier(LinearScorer.load(config.classifier.weights))
        decision = classifier.decide(view)
    else:  # pragma: no cover - needs a running decision model
        upstream = Upstream(config)
        decision = asyncio.run(_build_classifier(config, upstream).classify(view))
    try:
        route = select(config, decision, view, session_key=view.session_key({}), session=None)
    except NoRoute as exc:
        print(json.dumps({"error": str(exc), "decision": decision.as_dict()}, indent=2))
        return 1
    out = route.as_dict()
    out["baseline_model"] = config.baseline().name
    print(json.dumps(out, indent=2))
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    from agent_model_router.metrics import MetricsStore
    from agent_model_router.metrics import format_stats

    config = _config(args.config)
    db = args.db or config.metrics_db
    if db is None:
        print("metrics are disabled in the config", file=sys.stderr)
        return 1
    if not Path(db).expanduser().exists():
        print(f"no route log at {db} yet", file=sys.stderr)
        return 1
    store = MetricsStore(db)
    since = time.time() - args.hours * 3600 if args.hours else 0.0
    summary = store.summary(since)
    store.close()
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(format_stats(summary, config.baseline().name))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from agent_model_router.classifier.dataset import load_rows
    from agent_model_router.classifier.training import evaluate
    from agent_model_router.classifier.training import train_rows

    rows = load_rows(path=args.data) if args.data else load_rows()
    scorer = train_rows(rows)
    Path(args.out).write_text(scorer.to_json(), encoding="utf-8")
    report = {"train": evaluate(scorer, rows), "eval": evaluate(scorer, load_rows("eval.jsonl"))}
    print(json.dumps({"weights": args.out, **report}, indent=2))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from agent_model_router.classifier.dataset import load_rows
    from agent_model_router.classifier.linear import LinearScorer
    from agent_model_router.classifier.training import evaluate

    rows = load_rows(path=args.data) if args.data else load_rows("eval.jsonl")
    print(json.dumps(evaluate(LinearScorer.load(args.weights), rows), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-model-router",
        description="Local proxy that picks the model and reasoning effort per request.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--config",
        help="config YAML (default: $AGENT_ROUTER_CONFIG, "
        "~/.config/agent-model-router/config.yaml, then the bundled default)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("serve", help="run the proxy")
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("init", help="write the default config to edit")
    p.add_argument("path", nargs="?")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("token", help="print the local router token (created on first use)")
    p.set_defaults(func=cmd_token)

    p = sub.add_parser("explain", help="show the route for a request body or plain prompt")
    p.add_argument("file", help="JSON request body or plain text; '-' for stdin")
    p.add_argument("--api", choices=["anthropic", "openai"], default="anthropic")
    p.add_argument("--offline", action="store_true", help="heuristic only, no decision model")
    p.set_defaults(func=cmd_explain)

    p = sub.add_parser("stats", help="routes and estimated savings from the local log")
    p.add_argument("--db")
    p.add_argument("--hours", type=float, help="only the last N hours")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("train", help="retrain the difficulty scorer")
    p.add_argument("--data", help="labelled JSONL (default: the bundled synthetic set)")
    p.add_argument("--out", default="weights.json")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("eval", help="evaluate the scorer on a labelled JSONL")
    p.add_argument("--data", help="labelled JSONL (default: the bundled hand-written set)")
    p.add_argument("--weights")
    p.set_defaults(func=cmd_eval)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
