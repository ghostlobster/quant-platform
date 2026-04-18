"""
Daily ML signal execution cron job.

Loads the trained LightGBM alpha model, scores the configured ticker universe,
and translates the resulting scores into broker orders via the configured
``BROKER_PROVIDER`` (paper by default). Intended to run after the US market
close on weekdays.

Run manually:
    python -m cron.daily_ml_execute

Schedule (16:05 ET on weekdays):
    5 16 * * 1-5 cd /path/to/quant-platform && \\
        .venv/bin/python -m cron.daily_ml_execute

Optional circuit breaker — fail closed when ML knowledge is stale:
    python -m cron.daily_ml_execute --enforce-knowledge-gate
    KNOWLEDGE_GATE_ENFORCE=1 python -m cron.daily_ml_execute

    When the gate is on and ``KnowledgeAdaptionAgent`` returns a
    ``retrain`` verdict, the cron exits with code ``2`` before any order is
    placed. ``fresh`` / ``monitor`` verdicts proceed unchanged. The CLI flag
    wins over the env var when both are set.

Exit codes:
    0 — success (trades placed or no-op).
    1 — fatal error (lightgbm missing, no trained model, unexpected exception).
    2 — circuit breaker tripped on retrain verdict.

ENV vars
--------
    WF_TICKERS             Comma-separated tickers (default: SPY,QQQ,AAPL,MSFT,TSLA)
    ML_SCORE_THRESHOLD     Minimum |score| required to act (default: 0.3)
    ML_MAX_POSITIONS       Maximum simultaneous longs (default: 5)
    BROKER_PROVIDER        paper | alpaca | ibkr | schwab (default: paper)
    LGBM_ALPHA_MODEL_PATH  Path to the trained model checkpoint
    KNOWLEDGE_GATE_ENFORCE 1 → same as passing --enforce-knowledge-gate
"""
from __future__ import annotations

import argparse
import os
import sys

from strategies.ml_execution import execute_ml_signals
from strategies.ml_signal import _LGBM_AVAILABLE, MLSignal
from utils.logger import get_logger

log = get_logger("cron.daily_ml_execute")

DEFAULT_TICKERS = "SPY,QQQ,AAPL,MSFT,TSLA"

EXIT_OK = 0
EXIT_FATAL = 1
EXIT_KNOWLEDGE_GATE = 2


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cron.daily_ml_execute",
        description="Daily ML signal execution cron job.",
    )
    parser.add_argument(
        "--enforce-knowledge-gate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Refuse to place any order when KnowledgeAdaptionAgent returns a "
            "retrain verdict (exit code 2). When unset the job falls back to "
            "the KNOWLEDGE_GATE_ENFORCE env var; the flag wins on conflict."
        ),
    )
    return parser


def _resolve_gate_enforcement(flag: bool | None) -> bool:
    """CLI flag wins when explicitly set; otherwise consult env var."""
    if flag is not None:
        return flag
    return os.environ.get("KNOWLEDGE_GATE_ENFORCE", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _check_knowledge_gate() -> int | None:
    """Run KnowledgeAdaptionAgent and return an exit code when it blocks.

    Returns ``None`` when the verdict is fresh/monitor (proceed). Returns
    ``EXIT_KNOWLEDGE_GATE`` (2) when the verdict is ``retrain``.
    """
    try:
        from agents.knowledge_agent import KnowledgeAdaptionAgent
    except ImportError as exc:
        log.error("daily_ml_execute: knowledge agent unavailable", error=str(exc))
        return EXIT_KNOWLEDGE_GATE

    sig = KnowledgeAdaptionAgent().run({})
    recommendation = (sig.metadata or {}).get("recommendation", "fresh")
    log.info(
        "daily_ml_execute: knowledge-gate verdict",
        recommendation=recommendation,
        reasoning=sig.reasoning,
    )
    if recommendation == "retrain":
        log.error(
            "daily_ml_execute: knowledge gate tripped — refusing to trade",
            recommendation=recommendation,
            reasoning=sig.reasoning,
        )
        return EXIT_KNOWLEDGE_GATE
    return None


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    enforce_gate = _resolve_gate_enforcement(args.enforce_knowledge_gate)

    tickers_str = os.environ.get("WF_TICKERS", DEFAULT_TICKERS)
    tickers = [t.strip() for t in tickers_str.split(",") if t.strip()]
    threshold = float(os.environ.get("ML_SCORE_THRESHOLD", "0.3"))
    max_positions = int(os.environ.get("ML_MAX_POSITIONS", "5"))
    broker_name = os.environ.get("BROKER_PROVIDER", "paper")

    log.info(
        "daily_ml_execute: starting",
        n_tickers=len(tickers),
        threshold=threshold,
        max_positions=max_positions,
        broker=broker_name,
        enforce_knowledge_gate=enforce_gate,
    )

    if not _LGBM_AVAILABLE:
        log.error("daily_ml_execute: lightgbm is not installed — aborting")
        sys.exit(EXIT_FATAL)

    if enforce_gate:
        gate_exit = _check_knowledge_gate()
        if gate_exit is not None:
            sys.exit(gate_exit)

    try:
        model = MLSignal()
        if model._model is None:
            log.error(
                "daily_ml_execute: no trained baseline model found — "
                "run cron.monthly_ml_retrain first"
            )
            sys.exit(EXIT_FATAL)

        scores = model.predict(tickers, period="6mo")
        log.info("daily_ml_execute: scored", n_scores=len(scores))

        # Persist scored rows before routing — covers the case where the
        # circuit breaker (#120) or an execute_ml_signals exception skips
        # the writer inside strategies/ml_execution.py.
        try:
            from analysis.live_ic import record_predictions

            record_predictions(scores, model_name="lgbm_alpha", horizon_d=5)
        except Exception as exc:
            log.warning(
                "daily_ml_execute: record_predictions failed", error=str(exc),
            )

        actions = execute_ml_signals(
            scores,
            threshold=threshold,
            max_positions=max_positions,
        )

        log.info(
            "daily_ml_execute: complete",
            n_actions=len(actions),
            actions=actions,
        )

    except Exception as exc:
        log.error("daily_ml_execute: unexpected failure", error=str(exc))
        sys.exit(EXIT_FATAL)


if __name__ == "__main__":
    main()
