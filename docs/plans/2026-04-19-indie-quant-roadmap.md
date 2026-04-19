# Comparison: quant-platform vs. Commercial Platforms & Quant Trading Firms

## Context

User asked for a full comparison between this repo (`quant-platform`, branch
`claude/quant-platform-comparison-Rta76`) and **(a)** commercial/retail quant
platforms, and **(b)** proprietary quant-trading firm stacks, to surface gaps.
This is an analytical deliverable, not a code change — no files will be
modified. The plan file captures the comparison itself.

---

## Quick Positioning

`quant-platform` is a **research → paper → small-scale live** Python/Streamlit
stack. Strongest on ML-alpha research (López de Prado + Stefan Jansen
playbook), multi-broker paper trading, and an agent-based governance layer
(RegimeAgent, RiskAgent, SentimentAgent, ScreenerAgent, ExecutionAgent,
KnowledgeAdaptionAgent, MetaAgent). Weakest on institutional-grade
infrastructure: no FIX, no L2/tick data, no co-location, no compliance layer,
no distributed compute.

It sits between **QuantConnect/Alpaca-SDK** (retail) and **open-source
Backtrader/Zipline** — closer to a self-hosted "QuantConnect-lite with
agents + model zoo".

---

## Table 1 — vs. Commercial / Retail Quant Platforms

| Dimension | quant-platform (this repo) | QuantConnect / Lean | Numerai | WorldQuant Brain | Alpaca SDK | TradingView Pine | Bloomberg Terminal |
|---|---|---|---|---|---|---|---|
| **Pricing** | Free / self-host | Free tier + paid cloud compute | Free (stake in NMR) | Free (paid in BRAIN tokens) | Free API + commissions | $15-60/mo | $24k+/yr/seat |
| **Asset classes** | Equities, options (Greeks), crypto (CCXT); no native FX/futures | Equities, options, futures, FX, crypto | Global equities (pre-processed only) | Global equities (pre-processed) | US equities, crypto, options | All via charts, no native exec | All asset classes |
| **Market data** | yfinance, Alpaca WS, FRED, ThetaData, UnusualWhales, StockTwits; **no L2/tick** | Minute + tick (US equities), options chains, crypto | Pre-engineered obfuscated features | Pre-engineered alpha dataset | Minute + L2 (SIP) | EOD + intraday | Full L1/L2/L3, news, refs |
| **Backtesting** | Event-driven + walk-forward + Monte Carlo + purged-CV + combinatorial CV | Event-driven, tick-level, cloud-parallel | No backtester (submit predictions) | No backtester (submit alphas) | User-assembled (vectorbt etc.) | Strategy tester (basic) | No backtester |
| **ML / AI stack** | LightGBM, Ridge, Bayesian, MLP, CNN, GNN, DRL, sentiment NN; **Optuna+purged-CV**; triple-barrier, meta-labels, fractional diff | Ships ML but DIY; no opinionated zoo | Model zoo is the product (you submit predictions) | Operator-based feature DSL | BYO | None native | BQuant (prem. Python notebooks) |
| **Feature store** | In-memory + Redis adapter | None native | Dataset is the feature store | Operator expressions | None | None | BQL |
| **Agent / LLM layer** | 7 specialist agents + MetaAgent + LLM arbitration (Claude/OpenAI/Ollama) | None | None | None | None | None | AI Assist (prem.) |
| **Regime detection** | 4-state (SMA/VIX) + GARCH + structural-break + anomaly detector | User-coded | N/A | N/A | User-coded | User-coded | None packaged |
| **Portfolio / risk** | VaR, CVaR, Kelly, Markowitz, HRP, Deflated Sharpe, stress test | Basic PF, user-coded risk | None (submit predictions) | None | None native | None | PORT, MARS |
| **Execution venues** | Alpaca, IBKR, Schwab, Tradier, CCXT, paper | Alpaca, IBKR, Tradier, Bitfinex, Binance, Oanda | None (no trading) | None | Alpaca only | Broker-agnostic via webhook | EMSX / SSEOMS |
| **Execution algos** | Market, TWAP, VWAP | Market, Limit, bracket, TWAP/VWAP (BYO) | — | — | Market, Limit, bracket | Webhook only | TWAP/VWAP/POV/IS |
| **Scheduler / cron** | APScheduler + daily/monthly jobs | Cloud-scheduled | Weekly round | Weekly round | User-wired | Server-side alerts | MAPS |
| **Deployment** | Docker Compose (Streamlit + alerts + Prometheus + Grafana) | Managed cloud + local Lean CLI | Managed | Managed | BYO | SaaS | Thick client |
| **Storage** | SQLite (WAL) + DuckDB/Timescale adapters | Proprietary | — | — | BYO | — | Proprietary |
| **Observability** | structlog JSON + Prometheus + Grafana | Cloud UI | Leaderboard | Leaderboard | BYO | Basic | PORT/FXIP |
| **Compliance / audit** | **Missing** | Basic order log | N/A | N/A | Basic | N/A | Enterprise-grade |
| **Target user** | Indie quant / prop / bootcamp | Retail+pro quant | Crowd-sourced quant | Crowd-sourced quant | Retail dev | Retail trader | Sell-side / buy-side desks |

---

## Table 2 — vs. Proprietary Quant Firm Stacks

Firms: Two Sigma, Citadel Securities, Renaissance Technologies, Jane Street,
Hudson River Trading (HRT), Jump Trading. Values are industry-public approximations.

| Dimension | quant-platform | Prop / HFT firm stack | Gap |
|---|---|---|---|
| **Latency (order)** | ~50-500 ms (HTTPS/WS) | Sub-µs (kernel-bypass, FPGA, co-lo) | 6-9 orders of magnitude |
| **Market data** | L1 minute bars + yfinance | Direct exchange feeds (ITCH, OUCH), consolidated tape, full-depth L3 | No microstructure |
| **Connectivity** | REST / WebSocket | FIX 4.2/4.4/5.0, OUCH, native binary | No FIX session manager |
| **Co-location** | None | NY4/NY5/LD4/TY3 cross-connects to matching engines | Structural |
| **Execution venues** | 4 US brokers + CCXT | Direct market access to 100+ venues, dark pools, ATS | Structural |
| **Smart order routing** | Basic TWAP/VWAP | Proprietary SOR, liquidity-seeking, IS, PoV, dark aggregation | Major |
| **Risk engine** | On-demand VaR/CVaR/Kelly | Real-time pre-trade checks, portfolio Greeks, firm-wide limits, kill-switch | Major |
| **Research compute** | Local CPU; optional Ray | On-prem K8s/Slurm, GPU farms, petabyte tick archives | Scale |
| **Data archive** | SQLite (GB scale) | kdb+/Shakti columnar tick store (PB scale) | 6 orders of magnitude |
| **Alpha research** | Optuna + walk-forward + purged CV + meta-label | Same methodology + 20+ yrs proprietary tick, alt data (satellite, credit card, shipping) | Data moat, not method |
| **Model families** | LightGBM, Ridge, MLP/CNN/GNN, DRL | Same + proprietary deep learning (transformer on order book), genetic programming | Comparable toolbox |
| **Portfolio opt.** | Markowitz + HRP | Multi-period Bayesian / robust / transaction-cost-aware, factor-attributed | Incremental |
| **Asset classes** | Eq + options + crypto | All (eq, fixed income, FX, futures, options, swaps, credit, commodities, crypto) | Major |
| **Compliance** | None | MIFID II, Reg NMS, Reg SCI, Dodd-Frank, audit trail, archived order lifecycle | Major |
| **Infrastructure** | Docker Compose | Kubernetes, Kafka, Airflow, Spark, MLflow-tier model registry, DR sites | Major |
| **Headcount investment** | 1 dev | 100-1000+ engineers, quants, infra, compliance | — |
| **Annual tech spend** | $0-100 | $100M-$1B+ | — |

---

## Consolidated Gap List

| # | Gap | Severity | Notes |
|---|---|---|---|
| 1 | No FIX protocol | High | Blocks institutional broker / DMA |
| 2 | No tick / L2 / order-book data | High | Closes off microstructure alpha |
| 3 | No co-location / low-latency path | High | Can't compete in HFT / market-making |
| 4 | SQLite primary store | Medium | DuckDB/Timescale adapters exist but unused in hot path |
| 5 | No real-time streaming risk dashboard | Medium | Risk is on-demand, not pub/sub |
| 6 | No compliance / audit trail | Medium | Pre-trade limits, MIFID II, SEC reporting absent |
| 7 | No message bus (Kafka / NATS) | Medium | Cron-driven, silent on failure |
| 8 | No GPU training path | Medium | DRL / deep models CPU-only |
| 9 | No multi-leg / bracket / OCO natively | Medium | Options spreads require manual legs |
| 10 | Equities-only Greeks (Black-Scholes) | Low | No rates / commodity / FX vol surfaces |
| 11 | No native FX / futures asset class | Low | CCXT partially covers crypto perps |
| 12 | No distributed backtest orchestrator | Low | Ray wired for parallel WF only |
| 13 | No sell-side data (refs, corporate actions, earnings consensus) | Low | FRED macro + yfinance corporate events only |
| 14 | No mobile / alerting UX beyond Telegram/Slack | Low | Adequate for small scale |

---

## Comparative Strengths (what this repo does *better* than most retail tools)

1. **Agent-based governance** (KnowledgeAdaptionAgent auditing staleness + IC
   degradation + regime coverage, MetaAgent with LLM arbitration) — rare
   outside bespoke buy-side stacks.
2. **López de Prado methodology baked in** — triple-barrier, meta-labeling,
   fractional differentiation, purged + combinatorial CV, deflated Sharpe.
   Most retail platforms ship a naive train/test split.
3. **Multi-broker + provider DI** via clean Protocol + factory — swap
   Alpaca↔IBKR↔Schwab↔Tradier↔CCXT with an env var.
4. **Regime-conditioned Kelly sizing** (0.5× multiplier in high-vol) — simple
   but principled, and missing from Alpaca/QC defaults.
5. **Self-host + full source** — no vendor lock, auditable alpha pipeline,
   testable to 76%+ coverage.

---

## Verification

This is a documentation-only deliverable. To verify the claims above:

1. Capability inventory: run `ls adapters/broker/ adapters/market_data/
   strategies/ agents/ analysis/ backtester/` against the list in the tables.
2. Agent layer: `ls agents/*.py` should show the 7 agents named in Table 1.
3. ML methods: `ls strategies/*_signal.py strategies/*_agent.py` should show
   ridge, bayesian, mlp, cnn, gnn, drl, ml, sentiment.
4. López-de-Prado features: `ls analysis/{triple_barrier,meta_label,frac_diff,deflated_sharpe,structural_breaks}.py`.
5. No FIX / no tick confirmation: `grep -r "FIX\|ITCH\|OUCH" .` should return
   nothing substantive.

---

# ROADMAP

Two phases: **Phase 1 = indie-quant realism** (1 dev, $0–$5k/mo infra budget,
trading own capital $10k–$1M, optimising for robustness + alpha, *not*
latency). **Phase 2 = scale-up** (post-indie: small prop shop, friends-and-family
LP vehicle, or just future-proofing). Each item lists scope, files touched,
effort (S ≤ 1d, M ≤ 1w, L ≤ 1mo, XL > 1mo), and reuse of existing code.

---

## Phase 1 — Indie Quant Roadmap (close the gap that actually matters)

Ordered by **ROI for a solo operator**. Each item is self-contained.

### P1.1 — Pre-trade risk guard + kill-switch (M)
**Why**: Today order dispatch in `cron/daily_ml_execute.py` and
`strategies/ml_execution.py` has no hard limits. One bad signal → account
blown. This is the single highest-ROI indie-only gap.
**Scope**:
- New `risk/pretrade_guard.py` with: max position % of equity, max daily
  loss %, max gross/net exposure, max orders/day, symbol block-list.
- Wire into `adapters/broker/*_adapter.py` `place_order()` as a decorator so
  every broker path goes through it.
- Kill-switch file (`.killswitch`) + SIGTERM hook; honored by
  `cron/daily_ml_execute.py` and `scheduler/alerts.py`.
- Config via `.env`: `MAX_POSITION_PCT`, `MAX_DAILY_LOSS_PCT`, etc.
**Reuse**: `analysis/risk_metrics.py` for exposure math,
`journal/trading_journal.py` for daily-loss calc.
**Tests**: `tests/test_pretrade_guard.py` with fixture portfolios.

### P1.2 — Real-time streaming risk dashboard (M)
**Why**: Prometheus + Grafana are already deployed; risk is computed
on-demand but never emitted.
**Scope**:
- `risk/metrics_exporter.py`: Prometheus gauges for equity, gross exposure,
  net exposure, daily P&L, open-position VaR, current drawdown.
- Tick loop in `scheduler/alerts.py` → update gauges every 60s.
- Grafana dashboard JSON under `deploy/grafana/dashboards/risk.json`.
- Drawdown breach → alert via existing `alerts/channels.py`.
**Reuse**: `analysis/risk_metrics.py`, `journal/trading_journal.py`,
existing `prometheus-client` dep.

### P1.3 — Bracket / OCO / trailing-stop orders (M)
**Why**: Single-leg orders only today. Alpaca + IBKR + Tradier all support
brackets natively; it's purely a wiring job.
**Scope**:
- Extend `providers/broker.py` `Broker` Protocol with `place_bracket()`.
- Implement per-adapter in
  `adapters/broker/{alpaca,ibkr,tradier}_adapter.py`.
- Add `OrderIntent` dataclass with `take_profit`, `stop_loss`,
  `trail_percent`.
- `broker/paper_trader.py` — simulate brackets locally.
**Reuse**: Existing broker bridges already expose these via vendor SDK.

### P1.4 — Upgrade data tier: Polygon.io or Databento (M)
**Why**: yfinance is unreliable (rate-limits, revisions, gaps). Polygon
Stocks Starter = $29/mo for minute bars + historical tick; Databento is
tick-native and pay-per-byte.
**Scope**:
- New `adapters/market_data/polygon_adapter.py` implementing
  `MarketData` protocol.
- Register in `providers/market_data.py` factory.
- Cache warmup script `cron/polygon_backfill.py`.
- Keep yfinance as fallback.
**Reuse**: Adapter pattern in `adapters/market_data/alpaca_adapter.py`;
SQLite cache in `data/fetcher.py`.

### P1.5 — Promote DuckDB to hot path for backtests (M)
**Why**: SQLite walk-forward across 100 tickers × 5 yrs minute bars is
slow. DuckDB adapter exists (`adapters/tsdb/duckdb_adapter.py`) but isn't
wired into `backtester/walk_forward.py`.
**Scope**:
- Teach `data/fetcher.py` to route bulk reads through DuckDB when
  `TSDB_PROVIDER=duckdb`.
- Benchmark script in `scripts/bench_backtest.py`.
- Expected 10–50× speedup on multi-ticker walk-forward.

### P1.6 — MLflow model registry wiring (S)
**Why**: `adapters/model_registry/mlflow_adapter.py` exists but
`cron/monthly_ml_retrain.py` still pickles to disk.
**Scope**:
- Route `knowledge_registry.py` save/load through MLflow.
- Docker-compose sidecar `mlflow:5000`.
- `KnowledgeAdaptionAgent` reads IC history from MLflow runs.

### P1.7 — Options multi-leg strategy engine (L)
**Why**: Greeks and Tradier are there; spreads/straddles require manual
leg assembly.
**Scope**:
- `strategies/options_legs.py`: vertical spread, iron condor, straddle,
  calendar builders returning `list[OrderIntent]`.
- Greeks-aware position sizer in `risk/options_sizing.py` (delta-neutral
  target, max vega).
- Tradier `place_multi_leg()` wrapper.
**Reuse**: `analysis/greeks.py`, `broker/tradier_bridge.py`.

### P1.8 — Expose IBKR FX + futures + global equities (M)
**Why**: IBKR bridge only trades US equities today; same SDK covers FX,
futures, global stocks.
**Scope**:
- Extend `broker/ibkr_bridge.py` contract factory for FX cash, CME futures,
  LSE/HKEX stocks.
- Asset-class routing in `adapters/broker/ibkr_adapter.py`.
- Symbol metadata table in `data/symbols.py`.

### P1.9 — Event bus (Redis Streams) (M)
**Why**: Cron-driven fire-and-forget means silent failure. Redis is
already an optional feature-store adapter dep.
**Scope**:
- `bus/event_bus.py` thin wrapper over Redis Streams.
- Publishers: `cron/daily_ml_execute.py`, `scheduler/alerts.py`,
  `agents/*.py` emit typed events (`signal.generated`, `order.placed`,
  `risk.breach`).
- Dead-letter queue + replay tool `scripts/replay_events.py`.
- Subscriber example: alert daemon listens for `risk.breach`.

### P1.10 — Structured audit log + trade blotter export (S)
**Why**: No paper trail. Any serious broker (IBKR, Schwab) will
eventually ask.
**Scope**:
- Append-only `audit/` JSONL log: decision → order → fill → P&L, keyed by
  `run_id`.
- Streamlit page `pages/audit.py` with CSV export.
- Retention rotation cron.

### P1.11 — Paper→Live promotion guard (S)
**Why**: Today `BROKER_PROVIDER=alpaca` flips live with one env var. Need
explicit 2-step confirmation + a minimum paper track record.
**Scope**:
- `providers/broker.py`: refuse `alpaca` live unless
  `LIVE_TRADING_CONFIRMED=true` *and* journal shows ≥30d paper Sharpe > X.
- Banner in sidebar (`pages/shared.py`) showing live/paper mode colored.

### P1.12 — Walk-forward parallelism via Ray (S→M)
**Why**: Ray is already in `requirements.txt` and referenced but not
default. Single laptop run of 200-ticker × 5-yr combinatorial CV should
drop from hours to minutes.
**Scope**: Flip default executor in `backtester/walk_forward.py` to Ray
Pool when available.

**Phase 1 total effort**: ~3–5 months wall-clock for one dev, assuming
part-time evenings.

---

## Phase 2 — Scale-Up Roadmap (post-indie)

Triggered when the indie setup graduates to: managing others' money,
running a legal entity, trading 7-figure AUM, or targeting lower-latency
strategies.

### P2.1 — FIX protocol adapter (L)
- `adapters/broker/fix_adapter.py` via QuickFIX/n or simplefix.
- FIX 4.4 session manager + cert mgmt.
- Enables institutional brokers (Pershing, Fidelity Prime, IBKR Prime).

### P2.2 — L2 / order-book data (L)
- Databento ITCH or Polygon Advanced (L2 NBBO + depth).
- New `data/orderbook.py` with ring-buffer snapshots.
- Microstructure features in `data/features.py` (imbalance, spread,
  VPIN).

### P2.3 — Compliance layer (XL)
- Pre-trade: wash-sale detection, restricted list, short-locate.
- Post-trade: MiFID II / CAT reporting exports, best-exec attestation.
- `compliance/` module with daily attestation script.

### P2.4 — Kubernetes deployment (L)
- Helm chart; separate pods for streamlit, scheduler, workers, redis,
  prometheus, mlflow, postgres.
- Secrets via sealed-secrets or external-secrets operator.
- Horizontal pod autoscaling for Ray workers.

### P2.5 — Proper time-series store (L)
- Timescale or Clickhouse for tick archive.
- Columnar backtest queries replace DuckDB for > 10B bars.
- Partitioning by asset-class / date.

### P2.6 — GPU training cluster (L)
- CUDA-enabled containers for CNN / GNN / DRL.
- MLflow experiment tracking with GPU-pinned runs.
- Consider Modal.com or vast.ai rental for spiky workloads.

### P2.7 — Smart Order Router (XL)
- `execution/smart_router.py`: VWAP/IS/PoV with liquidity-seeking across
  Alpaca, IBKR, and direct exchange venues.
- Child-order slicer with signal-decay model.
- Backtest SOR impact via `backtester/execution_replay.py`.

### P2.8 — Alt-data ingestion platform (XL)
- Pluggable `adapters/altdata/` for: news NLP (AlphaSense, RavenPack),
  earnings transcripts (S&P), satellite (RS Metrics), credit-card (Second
  Measure), shipping (Spire).
- Feature engineering pipeline with Airflow DAGs.

### P2.9 — Multi-user / PM attribution (L)
- Per-strategy P&L attribution, risk budgets.
- Role-based UI (researcher / trader / PM / risk).
- Auth via OIDC (Keycloak).

### P2.10 — Disaster recovery + HA (L)
- Hot/cold Postgres replica, Redis sentinel, dual-region deploys.
- Runbooks in `deploy/runbooks/`.
- Chaos drill cron.

### P2.11 — Co-location / low-latency (XL, only if strategy demands)
- Equinix NY4/NY5 cross-connect.
- Kernel-bypass via Solarflare Onload or DPDK.
- Rust/C++ rewrite of order path.
- Probably out of scope forever for a non-HFT quant — include only as a
  decision gate.

---

## Phase 1 Verification (how the user tests each item end-to-end)

| Item | Verification |
|---|---|
| P1.1 Pre-trade guard | `pytest tests/test_pretrade_guard.py` + manual: paper-trade with tiny limits, expect order rejection |
| P1.2 Risk dashboard | `docker-compose up`, hit `http://localhost:3000`, confirm gauges update every 60s |
| P1.3 Brackets | Paper-trade bracket order, check fills hit TP/SL in journal |
| P1.4 Polygon | `MARKET_DATA_PROVIDER=polygon` + `pytest -m integration tests/test_polygon.py` |
| P1.5 DuckDB | `scripts/bench_backtest.py` — expect ≥10× speedup |
| P1.6 MLflow | Retrain cron, confirm model lands in MLflow UI at `:5000` |
| P1.7 Options spreads | Paper-trade iron condor on SPY, confirm 4 legs via Tradier |
| P1.8 IBKR asset classes | Paper-trade EUR/USD via IBKR gateway, verify fill |
| P1.9 Event bus | `redis-cli XREAD` shows live stream; replay tool reproduces a day |
| P1.10 Audit log | Generate trades, `scripts/export_blotter.py` → CSV diff matches journal |
| P1.11 Live guard | Set `BROKER_PROVIDER=alpaca` without confirm env var, expect refusal |
| P1.12 Ray WF | Walk-forward wall-clock before/after; expect ≥5× on 8-core laptop |

Each should keep `/pre-push` green (ruff + pytest 76% + bandit HIGH +
pip-audit).
