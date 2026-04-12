# Quant Trading Platform — Features, Philosophy & Mind Flow

> *"The market is a device for transferring money from the impatient to the patient."*
> — Warren Buffett

---

## Table of Contents

1. [The Overarching Philosophy](#1-the-overarching-philosophy)
2. [Technical Indicators — What They Are & When to Use Them](#2-technical-indicators)
3. [Risk Metrics — Measuring What You Cannot Ignore](#3-risk-metrics)
4. [Strategies — Turning Signals Into Decisions](#4-strategies)
5. [Backtesting Suite — Learning From the Past](#5-backtesting-suite)
6. [Portfolio & Position Sizing](#6-portfolio--position-sizing)
7. [Using Everything Together — The Decision Stack](#7-using-everything-together)
8. [Mind Flow Chart — The Trading Philosophy](#8-mind-flow-chart)
9. [Backward Use Cases — Reading Signals in Reverse](#9-backward-use-cases)
10. [Common Anti-Patterns to Avoid](#10-common-anti-patterns-to-avoid)

---

## 1. The Overarching Philosophy

This platform is built on three pillars:

### Pillar 1 — Edge Identification
A trade is only worth taking if there is a *statistical edge*: a repeatable condition where the probability-weighted expected return is positive. Indicators and screeners exist to surface that edge systematically, not emotionally.

### Pillar 2 — Risk-First Thinking
Every position has two questions before it is placed:
- **How much can I make?** (expected return)
- **How much can I lose?** (VaR, max drawdown, ATR stop distance)

If you cannot answer both, the trade does not happen.

### Pillar 3 — Humility Through Testing
No strategy is trusted until it has been:
1. Backtested on historical data
2. Walk-forward tested across multiple out-of-sample windows
3. Stress-tested with Monte Carlo simulation
4. Paper-traded before any real money is committed

The platform enforces this sequence — every module feeds the next.

---

## 2. Technical Indicators

### 2.1 RSI — Relative Strength Index

**What it is:**
RSI measures the speed and magnitude of recent price changes on a 0–100 scale. It compares average gains to average losses over a rolling 14-period window.

```
RSI = 100 - (100 / (1 + RS))
RS  = Average Gain (14 periods) / Average Loss (14 periods)
```

**Standard interpretation:**
| RSI Value | Signal |
|-----------|--------|
| < 30 | Oversold — potential buy zone |
| 30–70 | Neutral — no strong signal |
| > 70 | Overbought — potential sell zone |

**Backward use case:**
If a stock *crashed* and you see RSI was at 85 before the drop, that confirms the move was driven by exhausted momentum, not new fundamental news. You can annotate past crashes to train your intuition for future extreme readings.

**Platform role:** Powers the RSI Mean Reversion strategy and the stock screener signal labels.

---

### 2.2 MACD — Moving Average Convergence Divergence

**What it is:**
MACD tracks the relationship between two exponential moving averages (12-period and 26-period EMA) and plots the difference. A 9-period signal line smooths the result.

```
MACD Line   = EMA(12) - EMA(26)
Signal Line = EMA(9) of MACD Line
Histogram   = MACD Line - Signal Line
```

**Standard interpretation:**
- **Bullish crossover**: MACD crosses above signal line → momentum turning up
- **Bearish crossover**: MACD crosses below signal line → momentum turning down
- **Histogram expanding**: Trend strengthening in current direction
- **Histogram compressing**: Trend weakening, potential reversal ahead

**Backward use case:**
Look back at any major trend reversal. Almost always, MACD histogram was compressing (shrinking) 3–10 bars *before* the price reversed. This divergence — price making new highs while MACD makes lower highs — is one of the most reliable early warning signals in technical analysis.

**Platform role:** Displayed in the chart overlay; used to confirm trend direction before entering a position.

---

### 2.3 Bollinger Bands

**What it is:**
Bollinger Bands place a channel around price based on a 20-period simple moving average ± 2 standard deviations. The bands expand during high volatility and contract during low volatility.

```
Middle Band = SMA(20)
Upper Band  = SMA(20) + 2 × StdDev(20)
Lower Band  = SMA(20) - 2 × StdDev(20)
```

**Standard interpretation:**
- Price touching the **lower band** in an uptrend → potential buy
- Price touching the **upper band** in a downtrend → potential sell
- **Band squeeze** (bands narrowing) → volatility compression, breakout likely soon
- **Band expansion** → volatility surge, trend underway

**Backward use case:**
Study any large earnings gap in your chart history. You'll find Bollinger Bands were tight (squeezed) in the 1–2 weeks before the gap. The squeeze was the market pricing in *uncertainty* before the *resolution*. Use squeeze detection ahead of known binary events (earnings, FOMC).

**Platform role:** Overlay on price chart; combined with RSI to confirm mean-reversion entries.

---

### 2.4 EMA — Exponential Moving Average (20 & 50)

**What it is:**
EMA gives more weight to recent prices than a simple average. The 20-EMA tracks short-term trend; the 50-EMA tracks medium-term trend.

```
EMA(today) = Price(today) × k + EMA(yesterday) × (1 - k)
k = 2 / (period + 1)
```

**Standard interpretation:**
- **Price above EMA-20**: Short-term uptrend
- **Price above EMA-50**: Medium-term uptrend
- **EMA-20 crosses above EMA-50**: Golden cross — bullish trend confirmation
- **EMA-20 crosses below EMA-50**: Death cross — bearish trend signal

**Backward use case:**
During any sustained rally you've witnessed, go back and count how many times price bounced off the 20-EMA during the uptrend. In strong trends, the 20-EMA acts as *dynamic support*. Missing this on the first pullback is the most common way new traders exit good trades too early.

**Platform role:** Drives the SMA/EMA Crossover strategy; displayed as chart overlay.

---

### 2.5 ATR — Average True Range

**What it is:**
ATR measures market volatility as the average of the "true range" over 14 periods. True Range is the greatest of: High−Low, |High−PrevClose|, |Low−PrevClose|.

**Standard interpretation:**
- **High ATR**: Market is volatile — wider stops needed, smaller position sizes
- **Low ATR**: Market is calm — tighter stops possible, can size up
- ATR does NOT indicate direction — only volatility magnitude

**Backward use case:**
Look at any stop-out that felt unfair ("the market stopped me out then reversed"). In most cases, the stop was placed without accounting for ATR — it was too close given prevailing volatility. Replaying the trade with a 2×ATR stop would often have survived the noise and captured the eventual move.

**Platform role:** Powers the dynamic ATR stop-loss in the backtester. On every BUY, the stop is set at `entry_price − 2 × ATR`, adapting to current volatility rather than using a fixed percentage.

---

## 3. Risk Metrics

### 3.1 Sharpe Ratio

**What it is:**
Sharpe measures return per unit of total risk (volatility).

```
Sharpe = (Annualised Return - Risk-Free Rate) / Annualised Volatility
```

**Interpretation:**
| Sharpe | Quality |
|--------|---------|
| < 0 | Worse than risk-free |
| 0–1 | Acceptable |
| 1–2 | Good |
| > 2 | Excellent (rare in live trading) |

**Backward use case:** Compare two strategies with the same total return. The one with a higher Sharpe got there with less gut-wrenching volatility — that is the one more likely to be *followed* during live drawdowns.

---

### 3.2 Sortino Ratio

**What it is:**
Like Sharpe, but only penalises *downside* volatility. Upside volatility (prices going up fast) is not penalised.

```
Sortino = (Annualised Return - Risk-Free Rate) / Downside Deviation
```

**Why Sortino > Sharpe for traders:**
A strategy that gains 3% per day for a week then gives back 1% will look *worse* on Sharpe (high volatility) but *better* on Sortino (the volatility was on the upside). For most traders, only losing days feel like risk.

**Backward use case:** A strategy with Sharpe 1.2 but Sortino 2.1 tells you the volatility is mostly on winning days. That is a strategy worth running. A strategy with Sharpe 1.2 and Sortino 0.8 is producing symmetric — or worse, downside-skewed — volatility.

---

### 3.3 Calmar Ratio

**What it is:**
Annualised return divided by maximum drawdown. Answers: "How much did you earn per unit of your worst loss?"

```
Calmar = Annualised Return / |Max Drawdown|
```

**Backward use case:** Funds are typically compared on Calmar. A fund returning 20% with a 40% max drawdown (Calmar 0.5) is objectively worse than one returning 15% with a 10% max drawdown (Calmar 1.5). Most investors cannot stomach drawdowns — Calmar captures that directly.

---

### 3.4 Value at Risk (VaR) & CVaR

**What it is:**
VaR answers: "What is the maximum I expect to lose on a single day with X% confidence?"

```
Historical VaR (95%) = 5th percentile of observed daily returns
Parametric VaR       = μ + z(0.05) × σ   [assumes Gaussian returns]
CVaR / Expected Shortfall = Average loss on the worst days beyond VaR
```

**Backward use case:**
Take any crash event (March 2020, August 2015). Run historical VaR on the 6 months *before* the crash. It will show the 95% VaR was around 2–3%. The actual crash was 10× that. This illustrates why CVaR matters — it tells you what happens in the *tail beyond the tail*, where fat-tailed distributions live.

**Platform role:** `risk/var.py` — four implementations. Use Historical VaR as the day-to-day risk budget; use CVaR for stress-test scenarios.

---

### 3.5 Kelly Criterion

**What it is:**
Kelly gives the mathematically optimal fraction of capital to risk on each trade to maximise long-run wealth growth without risk of ruin.

```
Kelly % = (Win Rate × Avg Win - Loss Rate × Avg Loss) / Avg Win
```

The platform uses **Half-Kelly** (divide by 2) and caps at 25% for practical safety.

**Why half-Kelly?**
Full Kelly maximises *expected log wealth* but produces extreme volatility in practice. Half-Kelly sacrifices roughly 25% of theoretical growth rate in exchange for half the drawdowns. That trade is almost always worth it.

**Backward use case:**
Look at a strategy's backtest results. Win rate 55%, average win 5%, average loss 3%. Full Kelly says bet 28% of capital per trade. Most traders would be wiped out by the inevitable losing streak at that size. Half-Kelly says 14% — still aggressive but survivable.

**Platform role:** `risk/kelly.py` — call `kelly_fraction()` after each backtest. Use it to size the next paper trade.

---

### 3.6 Correlation Matrix

**What it is:**
Pairwise return correlation between portfolio holdings. Values range from −1 (perfect inverse) to +1 (perfect co-movement).

**Why it matters:**
Holding 10 tech stocks looks like diversification. But if they are all 95% correlated, you effectively have 1 position. A single bad macro event takes down the whole portfolio.

**Backward use case:**
During any sector rotation crash, check the correlation matrix of "diversified" portfolios. They become *more* correlated in crises — correlations spike toward 1.0 precisely when diversification would help most. This is called "correlation breakdown." The correlation matrix helps you spot concentrated bets before the crisis.

**Platform role:** `risk/correlation.py` — run before adding any new ticker to the watchlist. If it correlates >0.8 with existing holdings, reconsider.

---

### 3.7 Markowitz Portfolio Optimization

**What it is:**
Markowitz mean-variance optimization finds portfolio weights that maximise return for a given level of risk (or minimise risk for a given return). The set of all optimal portfolios forms the *efficient frontier*.

**Two key points:**
- **Max Sharpe portfolio**: Best risk-adjusted return — the theoretical "right" allocation
- **Min Volatility portfolio**: Lowest variance — best for risk-averse or uncertain markets

**Backward use case:**
Run Markowitz on any 3-year period before a crash. The min-volatility portfolio almost always had dramatically less exposure to the sector that crashed — because lower-volatility names naturally get higher weight. Markowitz implicitly reduces concentration in the riskiest assets.

**Platform role:** `risk/markowitz.py` — use for periodic rebalancing suggestions on the paper portfolio.

---

## 4. Strategies

### 4.1 SMA/EMA Crossover

**Logic:**
- **BUY** when EMA-20 crosses above EMA-50 (short-term momentum overtakes medium-term)
- **SELL** when EMA-20 crosses below EMA-50

**When it works:** Trending markets with sustained directional moves. Works best on daily charts for weekly-to-monthly trend following.

**When it fails:** Choppy, sideways markets. The strategy will whipsaw — buy, then immediately sell, buy again — generating commissions with no net return.

**Backward use case:** Overlay the crossover signals on any historic chart. In 2017 crypto bull run — perfect. In 2015–2016 sideways SPY — terrible. The market regime matters more than the indicator.

---

### 4.2 RSI Mean Reversion

**Logic:**
- **BUY** when RSI drops below 30 (oversold)
- **SELL** when RSI rises above 70 (overbought)

**When it works:** Range-bound, mean-reverting assets. Works well on sector ETFs, pairs, and indices during low-trend environments.

**When it fails:** Strong trending markets. A stock in free-fall can have RSI at 25 for weeks — "oversold" keeps getting more oversold.

**Backward use case:** Study any meme stock crash (e.g., GameStop post-squeeze). RSI was below 30 for 15 consecutive days while the price halved. Mean reversion *failed* because the underlying wasn't mean-reverting — it was in structural decline.

---

### 4.3 Walk-Forward Validation

**Logic:** Slice the full price history into rolling windows. Train a strategy on one window, test it on the next (out-of-sample), step forward, repeat.

**Why it beats simple backtesting:**
A strategy that looks great on a single backtest might be *overfit* to that specific period. Walk-forward shows whether the edge persists *out-of-sample* — which is the only sample that matters.

**Consistency score:** The fraction of test windows where the strategy was profitable. A strategy with 60% Sharpe but only 30% consistency (profitable in only 3 of 10 windows) is much riskier than one with 40% Sharpe but 80% consistency.

---

### 4.4 Monte Carlo Simulation

**Logic:** Bootstrap historical returns thousands of times to generate a distribution of possible future equity curves.

**Key output:** Probability of profit, and the 5th percentile path — the bad scenario you need to be able to survive emotionally and financially.

**Backward use case:**
Before deploying any strategy live, run Monte Carlo. If the 5th percentile path loses 40% before recovering, ask yourself: will you actually hold through a 40% drawdown? If not, the strategy is not suitable for you regardless of how good the median looks.

---

## 5. Backtesting Suite

### The Full Workflow

```
Raw Data  →  Backtest  →  Walk-Forward  →  Monte Carlo  →  Paper Trade
```

| Stage | What It Catches |
|-------|----------------|
| Backtest | Basic strategy viability on historical data |
| Walk-Forward | Overfitting — does the edge hold out-of-sample? |
| Monte Carlo | Tail risk — what's the realistic worst case? |
| Paper Trade | Execution reality — slippage, timing, psychology |

**Never skip stages.** A strategy that passes Backtest but fails Walk-Forward is overfit. One that passes Walk-Forward but fails Monte Carlo has unacceptable tail risk. One that passes all three but fails Paper Trading has an execution problem.

---

## 6. Portfolio & Position Sizing

### The Sizing Hierarchy

```
1. Kelly Criterion    →  Max position size (mathematical ceiling)
2. VaR Budget         →  Daily loss limit constraint
3. Correlation Check  →  Concentration limit
4. Markowitz Weight   →  Optimal allocation within constraints
5. ATR Stop           →  Where you're wrong (exit point)
```

### The Position Sizing Formula in Practice

```
Position Size = min(Kelly%, VaR-constrained %) × Capital
Stop Price    = Entry - 2 × ATR
Risk Per Trade = Position Size × (Entry - Stop) / Entry
```

Risk per trade should never exceed 1–2% of total capital. If Kelly says 15% but the ATR stop implies a 10% loss at that size, scale down until risk per trade is within limits.

---

## 7. Using Everything Together — The Decision Stack

Before entering any trade, work through this stack top-to-bottom:

### Step 1 — Market Regime Check
- Is EMA-20 above EMA-50? (uptrend)
- Is the broader market (SPY) in an uptrend or downtrend?
- What does the VaR say about current market volatility?

→ *Only trade in your strategy's preferred regime.*

### Step 2 — Signal Confirmation
- Does the screener flag the ticker? (RSI, volume, trend signals)
- Do at least 2 indicators agree? (RSI oversold + MACD turning up + price at Bollinger lower band)
- Is there a catalyst risk? (check earnings calendar — avoid entering 1 week before earnings unless it's the strategy)

→ *Require confluence. One indicator is a hint. Three is a signal.*

### Step 3 — Risk Sizing
- Compute ATR → set stop price
- Compute Kelly fraction from backtest stats
- Compute VaR for the position size
- Check portfolio correlation — is this adding risk or diversifying?

→ *Size so that if the stop hits, you lose < 1.5% of total capital.*

### Step 4 — Backtest Validation
- Has this setup been walk-forward tested?
- What is the consistency score? (target > 60%)
- What does Monte Carlo show for the 5th percentile?

→ *If you haven't tested it, paper trade it first.*

### Step 5 — Execute & Monitor
- Enter position, log the trade
- Set the alert (price below stop → notification)
- Review portfolio correlation matrix after entry
- Monitor Sortino and Calmar on the running paper portfolio weekly

→ *A trade without a plan for exit is a guess, not a trade.*

---

## 8. Mind Flow Chart — The Trading Philosophy

```
┌─────────────────────────────────────────────────────────────────┐
│                     MARKET OBSERVATION                          │
│              (Price Action + Volume + News)                     │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   REGIME IDENTIFICATION                         │
│                                                                 │
│   EMA-20 vs EMA-50         Bollinger Band Width                 │
│   ┌──────────────┐         ┌──────────────────┐                 │
│   │ 20 > 50      │         │ Squeeze → Wait   │                 │
│   │ UPTREND ✓    │         │ Expand → Trade   │                 │
│   │ 20 < 50      │         └──────────────────┘                 │
│   │ DOWNTREND ✗  │                                              │
│   └──────────────┘                                              │
└─────────────────────────┬───────────────────────────────────────┘
                          │ Regime confirmed
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SIGNAL GENERATION                            │
│                                                                 │
│   RSI < 30         MACD Crossover        Price at BB Lower      │
│   (Oversold)       (Momentum Turn)       (Mean Reversion)       │
│       │                 │                      │                │
│       └────────────┬────┘──────────────────────┘                │
│                    │                                            │
│             CONFLUENCE CHECK                                    │
│          2+ signals required                                    │
│          ┌──────────────────────┐                               │
│          │ YES → Proceed        │                               │
│          │ NO  → Wait / Pass    │                               │
│          └──────────────────────┘                               │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                 EARNINGS CALENDAR CHECK                         │
│                                                                 │
│   Earnings within 5 days?                                       │
│   ┌──────────────────────────────────────┐                      │
│   │ YES → Avoid or size down 50%         │                      │
│   │ NO  → Proceed to risk sizing         │                      │
│   └──────────────────────────────────────┘                      │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RISK SIZING                                │
│                                                                 │
│  ATR Stop         Kelly Fraction       VaR Budget              │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │ Stop =   │    │ Half-Kelly   │    │ Daily VaR ≤  │          │
│  │ Entry −  │    │ Cap @ 25%    │    │ 2% capital   │          │
│  │ 2 × ATR  │    └──────────────┘    └──────────────┘          │
│  └──────────┘                                                   │
│                                                                 │
│  Final Size = min(Kelly%, VaR-limit%) × Capital                 │
│  Risk Check = Size × (Entry−Stop)/Entry < 1.5% capital          │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│               PORTFOLIO CONTEXT CHECK                           │
│                                                                 │
│  Correlation Matrix      Markowitz Weights                      │
│  ┌─────────────────┐    ┌─────────────────────────┐            │
│  │ Corr > 0.8 with │    │ Does this fit optimal   │            │
│  │ existing?       │    │ portfolio allocation?   │            │
│  │ → Reduce size   │    └─────────────────────────┘            │
│  └─────────────────┘                                           │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                  BACKTEST VALIDATION                            │
│                                                                 │
│  Standard Backtest    Walk-Forward       Monte Carlo            │
│  ┌─────────────┐     ┌──────────────┐  ┌────────────────┐      │
│  │ Sharpe > 1  │     │ Consistency  │  │ P(profit) > 55%│      │
│  │ Calmar > 1  │     │ Score > 60%  │  │ P5 drawdown    │      │
│  │ MaxDD < 20% │     │              │  │ survivable?    │      │
│  └─────────────┘     └──────────────┘  └────────────────┘      │
│                                                                 │
│              ALL THREE MUST PASS                                │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PAPER TRADE FIRST                            │
│                                                                 │
│   Execute in paper account  →  Monitor 20+ trades               │
│   Track Sortino, Calmar on live paper equity curve              │
│   Circuit Breaker: drawdown > 20% → halt new trades            │
└─────────────────────────┬───────────────────────────────────────┘
                          │ After 20+ trades, Sortino > 1
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   LIVE TRADING (Phase 3)                        │
│                                                                 │
│   Alpaca API   →   Small real positions   →   Full scaling       │
│   Slippage modelling ON   |   Commission accounting ON          │
│   Weekly Markowitz rebalance review                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. Backward Use Cases — Reading Signals in Reverse

One of the most powerful learning exercises is to take any major market event and run the full indicator stack *backward* — not to predict, but to understand what the signals were saying before it happened.

### 9.1 Flash Crash Diagnosis
**Event:** Sudden -5% intraday crash, quick recovery.

Work backward:
- **ATR**: Was ATR expanding in days prior? (Yes → volatility was rising, crash was within statistical bounds)
- **Bollinger Bands**: Were bands contracting? (Yes → squeeze released into the crash)
- **RSI**: Did RSI spike below 20 intraday? (Yes → extreme oversold, recovery likely)
- **VaR**: Was the daily VaR at the 95th percentile breached? (Yes → a genuine tail event)

**Learning:** Flash crashes are ATR expansion + Bollinger squeeze releases. If you held cash during low-ATR squeezes, you had the ability to buy the flash crash.

---

### 9.2 Trend Reversal Diagnosis
**Event:** A stock you owned went from +30% to -10% from its peak before you sold.

Work backward:
- **EMA**: When did EMA-20 cross below EMA-50? (That was the exit signal you missed)
- **MACD**: When did the MACD histogram start compressing? (That was the early warning 2–3 weeks earlier)
- **RSI**: Did RSI form a lower high while price formed a higher high? (Divergence — the classic top signal)
- **Calmar**: Was the strategy Calmar falling below 1 in the walk-forward test? (Yes → early evidence the edge was fading)

**Learning:** Most trend reversals announce themselves 2–3 weeks in advance through indicator divergence. The EMA crossover is the *confirmation*. By then it is too late for a great exit but not too late for a good one.

---

### 9.3 False Breakout Diagnosis
**Event:** You bought a breakout above resistance. Price immediately fell back below.

Work backward:
- **Volume**: Was volume below the 20-day average on the breakout bar? (Yes → no institutional participation, fake move)
- **Bollinger Upper Band**: Did price break above the upper band without a trend? (Yes → stretched, not sustained)
- **Walk-Forward Consistency**: Was the breakout strategy consistency score below 50% in the last test? (Yes → the edge wasn't there)

**Learning:** True breakouts have above-average volume and occur in trending markets (EMA-20 > EMA-50). Without both, the probability favors a false breakout.

---

### 9.4 Drawdown Post-Mortem
**Event:** Paper portfolio hit -18% drawdown, circuit breaker almost triggered.

Work backward:
- **Correlation**: Were all positions in the same sector? (Yes → 1 macro event took everything down together)
- **Kelly**: Were position sizes above the Kelly ceiling? (Yes → oversized, amplified the loss)
- **VaR**: Was daily VaR being monitored? (No → losses accumulated without a daily budget check)
- **Sortino**: Was Sortino falling before the drawdown? (Yes → losing days were getting larger, signal it was time to reduce)

**Learning:** Drawdowns are rarely surprises in hindsight. Correlation concentration + oversizing + ignoring Sortino trend = predictable drawdown. The tools were there; they weren't being consulted.

---

## 10. Common Anti-Patterns to Avoid

| Anti-Pattern | Why It's Dangerous | The Fix |
|---|---|---|
| Using a single indicator in isolation | One signal = 33% hit rate at best | Require 2+ indicator confluence |
| Backtesting without walk-forward | Overfit to historical period | Always walk-forward validate |
| Ignoring correlation | "10 stocks" is really 1 position | Check correlation before every new entry |
| Full Kelly position sizing | Mathematically correct, psychologically fatal | Use Half-Kelly, cap at 25% |
| Entering before earnings | Binary risk that no indicator can predict | Check earnings calendar — avoid or halve size |
| Skipping paper trading | Execution surprises destroy backtested returns | Min 20 paper trades before any real capital |
| No stop loss | Unlimited downside | ATR-based stop on every trade |
| Over-optimising parameters | RSI(12) vs RSI(14) on same dataset = noise | Walk-forward all parameter choices |
| Trading against the trend | Mean reversion in a downtrend = catching knives | Confirm EMA trend direction first |
| Ignoring VaR budget | Single bad day can ruin a month | Set and enforce a daily VaR limit |

---

## Summary Table

| Tool | Primary Signal | Used For | Works Best When |
|------|---------------|----------|-----------------|
| RSI | Momentum extremes | Entry timing | Range-bound markets |
| MACD | Trend momentum | Trend confirmation | Trending markets |
| Bollinger Bands | Volatility & range | Breakout/mean reversion | All markets |
| EMA 20/50 | Trend direction | Regime filter | Trending markets |
| ATR | Volatility magnitude | Stop placement, position size | All markets |
| Sharpe Ratio | Risk-adjusted return | Strategy comparison | Post-backtest |
| Sortino Ratio | Downside risk | Live monitoring | Post-backtest |
| Calmar Ratio | Return/drawdown | Strategy comparison | Post-backtest |
| VaR / CVaR | Daily loss estimate | Risk budget | Pre-trade, daily |
| Kelly Criterion | Position size | Capital allocation | Post-backtest |
| Correlation Matrix | Portfolio concentration | Diversification check | Pre-trade |
| Markowitz | Optimal weights | Portfolio rebalancing | Weekly/monthly |
| Walk-Forward | Out-of-sample validation | Overfitting check | Pre-live |
| Monte Carlo | Tail risk | Worst-case planning | Pre-live |
| Earnings Calendar | Binary event risk | Trade timing | Pre-trade |
| Circuit Breaker | Max drawdown limit | Risk management | Live paper trading |

---

*Last updated: April 2026 | Quant Platform v1.0 | Personal Use*
