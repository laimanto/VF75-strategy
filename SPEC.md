# production_manual_v33 — System Specification

**Strategy:** VF75 Manual v33 — VIX 18-Strike Call, rule-based entry/exit  
**Last updated:** 2026-06-23

---

## 1. Overview

This system trades a single VIX call option position at a time. It is built around a set of momentum and mean-reversion rules applied to VF75, a synthetic 75-day constant-maturity VX futures price. When all entry conditions are met it buys a VIX call at the M3 expiry. It holds until one of five exit conditions fires or the 75-day hard deadline expires.

The daily pipeline runs at 4:35 pm EDT on GitHub Actions, updates all data files in the repo, and publishes a live dashboard to GitHub Pages.

---

## 2. Repository Layout

```
production_manual_v33/
├── .github/
│   └── workflows/
│       └── daily_run.yml          # GitHub Actions cron job
├── data/
│   ├── vx_history.csv             # VX60d, VX90d, VF75, VVIX, VIX — one row per trading day
│   ├── features.csv               # all derived features — one row per trading day
│   ├── option_price.csv           # daily M3 option market mid — one row per trading day
│   ├── trades.csv                 # closed trade log
│   ├── daily_log.csv              # in-position daily ROI log (reset each new trade)
│   ├── position.json              # current open position (or out-of-position state)
│   ├── signal.json                # most recent signal output with condition detail
│   └── fetched.json               # raw fetch output for the day (transient)
├── scripts/
│   ├── setup_initial.py           # one-time bootstrap (run locally before first push)
│   ├── fetch_data.py              # daily data fetch
│   ├── eval_signal.py             # rule-based signal evaluation
│   ├── run_daily.py               # master orchestrator
│   └── gen_dashboard.py           # HTML dashboard generator
├── dashboard/
│   └── index.html                 # generated daily; served via GitHub Pages
├── requirements.txt
└── .gitignore
```

**Critical data files — never clear or truncate:**  
`option_price.csv`, `trades.csv`, `daily_log.csv`  
yfinance has no historical option data API. Every row in `option_price.csv` is permanent.

---

## 3. Key Definitions

| Symbol | Definition |
|--------|-----------|
| **VF75** | Synthetic 75-day constant-maturity VX futures price = (VX60d + VX90d) / 2 |
| **VX60d** | VX price interpolated to 60-day constant maturity from surrounding contracts |
| **VX90d** | VX price interpolated to 90-day constant maturity from surrounding contracts |
| **M3 contract** | First VX futures contract with DTE ≥ 60 from today (typically ~85 days out) |
| **M3 option** | VIX call option whose expiry matches the M3 VX contract settlement date |
| **sigma_now** | VVIX / 100, floored at 0.80 |
| **ema63** | Exponential moving average of VF75, span = 63 trading days (~3 months) |
| **MACD(5,13)** | EMA(5) − EMA(13) of VF75 |
| **roll_mu / roll_sd** | Rolling 84-day mean and std dev of VF75 |
| **roll_rank** | Percentile rank of current VF75 within 84-day window |
| **spike_level** | roll_mu + 2 × roll_sd |
| **B76** | Black-76 call price: F=VF75, K=strike, T=DTE/365, σ=sigma_now, r=4.5% |
| **entry_mid** | Real market mid = (bid + ask) / 2 at time of entry |

---

## 4. Strategy Parameters

### 4.1 Option Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Underlying | VIX call option | Option chain via yfinance `^VIX` |
| Strike K | next_even(VF75 × 1.05) | Next even integer ≥ VF75 × 5% OTM |
| Expiry | M3 settlement date | First VX contract with DTE ≥ 60 |
| Entry price | market mid = (bid + ask) / 2 | Stored in position.json as `entry_mid` |
| Exit price | market mid = (bid + ask) / 2 | |
| ROI | (exit_mid − entry_mid) / entry_mid × 100 | Simple pct; total ROI = additive sum |

### 4.2 Feature Constants

| Constant | Value |
|----------|-------|
| `WM` (rolling window) | 84 trading days |
| `EMA_SP` (EMA span) | 63 trading days |
| `MACD_F / MACD_S` | 5 / 13 |
| `ATR_WIN` | 10 trading days |
| `SIGMA_DEF` (floor) | 0.80 |
| `R` (risk-free rate) | 0.045 (4.5%) |
| `TENOR` (calendar days) | 75 |

### 4.3 Entry Conditions (ALL must be met)

| # | Condition | Threshold | Description |
|---|-----------|-----------|-------------|
| C1 | `roll_rank` | ≤ 0.50 | VF75 at or below 84-day median |
| C2 | `MACD(5,13)` | > 0 | Short-term momentum positive |
| C3 | `roc_3m` | ≥ −8.0% | No sustained 3-month decline |
| C4 | `roc_1m` | ≥ −5.0% | No sharp 1-month drop |
| C5 | VF75 vs EMA63 | VF75 < EMA63 | Price dipping below 3-month trend |
| C6 | EMA gap | (EMA63 − VF75) / EMA63 ≤ 3.5% | Not in crash / freefall |
| C7 | `atr10_filter` | > −1.0 | No strong 10-day decline in ATR units |
| C8 | `vix_spread` | < −1.0 | Normal contango (VF75 > VIX) |
| C9 | SL cooldown | Not in cooldown period | Blocked 20 trading days after SL exit |

### 4.4 Exit Conditions (ANY fires a SELL)

| Priority | Condition | Min Hold | Trigger |
|----------|-----------|----------|---------|
| 1 | **Spike TP** | None | VF75 ≥ roll_mu + 2 × roll_sd |
| 2 | **Stop Loss** | None | (current_mid − entry_mid) / entry_mid ≤ SL threshold |
| 3 | **MACD Decay** | 20 trading days | MACD(5,13) < 0 AND VF75 < entry_VF75 |
| 4 | **Vol Exit** | 20 trading days | sigma_now ≥ 1.50 |
| 5 | **Hard Deadline** | — | Calendar days held ≥ 75 |

### 4.5 Adaptive Stop Loss

| Regime | Condition at Entry | SL Threshold |
|--------|-------------------|--------------|
| Calm | sd84 < 1.0 | −35% |
| Volatile | sd84 ≥ 1.0 | −38% |

`sd84` is `roll_sd` measured at entry and stored permanently in the position. After an SL exit, entry is blocked for 20 trading days (stored as `sl_cooldown_until`).

---

## 5. VX Settlement Date Formula

VX futures settle on **Wednesday 30 calendar days before the 3rd Friday of the following month.**

```
third_friday(Y, M)  → 3rd Friday of month M in year Y
vx_settle(Y, M)     → third_friday(Y, M+1) − 30 calendar days
```

The M3 contract used for option expiry is the first upcoming settlement date with DTE ≥ 60.

---

## 6. Feature Derivations

All computed from `VF75` series in `vx_history.csv`:

```
sigma_now          = max(VVIX / 100, 0.80)
ema63              = EMA(VF75, span=63)
macd               = EMA(VF75, 5) − EMA(VF75, 13)
roll_mu            = rolling mean of VF75, window=84
roll_sd            = rolling std dev of VF75, window=84, ddof=1
roll_rank          = percentile rank of VF75 in 84-day window
roc_3m             = (VF75 / VF75.shift(63) − 1) × 100
roc_1m             = (VF75 / VF75.shift(21) − 1) × 100
vix_spread         = VIX − VF75
spike_level        = roll_mu + 2 × roll_sd
atr10              = rolling mean of |VF75.diff()|, window=10
vf75_change_10d_atr = VF75.diff(10) / atr10
```

---

## 7. Data File Schemas

### `data/vx_history.csv`
```
Date, VX60d, VX90d, VF75, VVIX, VIX
```
One row per trading day. Populated by `setup_initial.py` (full history) then `fetch_data.py` (one row/day).

### `data/features.csv`
```
Date, VX60d, VX90d, VF75, VVIX, VIX,
sigma_now, ema63, macd, roll_mu, roll_sd, roll_rank,
roc_3m, roc_1m, vix_spread, spike_level, atr10, vf75_change_10d_atr
```

### `data/option_price.csv`
```
date, vf75, strike, expiry, option_mid, option_bid, option_ask,
sigma_now, b76_mid, days_held, in_position
```
`option_mid` = (bid + ask) / 2. `b76_mid` is theoretical Black-76 mid kept for reference.  
**Append-only. Never clear.**

### `data/trades.csv`
```
trade_id, entry_date, entry_vf75, strike, entry_mid, expiry,
exit_date, exit_vf75, exit_mid, days_held, roi_pct, exit_reason,
sd84_at_entry, sl_used, notes
```
`entry_mid` and `exit_mid` are real market mids, not theoretical.  
**Append-only. Never clear.**

### `data/daily_log.csv`
```
date, vf75, vix, sigma_now, option_mid, signal, in_position, days_held, roi_mid
```
Written only when a position is open. Reset (headers only) each time a new trade opens.  
**Append-only. Never clear.**

### `data/position.json` — Out of position
```json
{
  "in_position": false,
  "last_trade_id": 1,
  "last_exit_date": "2026-06-23",
  "last_exit_reason": "MACD_DECAY",
  "last_roi_pct": 12.5,
  "sl_cooldown_until": null
}
```

### `data/position.json` — In position
```json
{
  "in_position": true,
  "trade_id": 2,
  "entry_date": "2026-06-23",
  "entry_vf75": 20.178,
  "entry_sigma": 0.9619,
  "sd84_at_entry": 0.45,
  "sl_used": 35.0,
  "strike": 22,
  "entry_mid": 2.70,
  "expiry": "2026-09-16",
  "tenor": 75,
  "sl_cooldown_until": null
}
```

### `data/signal.json`
```json
{
  "signal": "HOLD",
  "exit_reason": null,
  "conditions_met": 5,
  "conditions_total": 9,
  "conditions": { ... }
}
```

---

## 8. Initial Setup (One-Time, Run Locally)

### 8.1 Prerequisites

- Python 3.11+
- CBOE per-contract VX CSV files stored locally (one file per futures expiry)
  - File naming: `VX_YYYY-MM-DD.csv` where the date is the settlement date
  - Source: CBOE website historical data section (download once)
- Internet access for yfinance (VVIX, VIX history)

### 8.2 Install Dependencies

```
pip install -r requirements.txt
```

`requirements.txt`:
```
yfinance>=0.2.40
pandas>=2.2.0
numpy>=1.26.0
scipy>=1.12.0
requests>=2.31.0
```

### 8.3 Run Initial Setup

```
python scripts/setup_initial.py --cboe-dir <path_to_folder_with_VX_csv_files>
```

This will:
1. Read all `VX_YYYY-MM-DD.csv` files from the given directory
2. For each trading date, interpolate VX60d and VX90d from surrounding contracts
3. Download full `^VVIX` and `^VIX` history from yfinance
4. Merge into `data/vx_history.csv` (one row per trading day from 2013 onward)
5. Compute all rolling features → `data/features.csv`
6. Create empty `data/trades.csv`, `data/daily_log.csv`, `data/option_price.csv` with correct headers (skips if already present)

Expected output: ~3,400 rows in `vx_history.csv` and `features.csv` (2013-01-02 to present).

### 8.4 Commit to GitHub

After setup completes:

```
git add data/vx_history.csv data/features.csv
git commit -m "Initial data bootstrap"
git push origin main
```

The remaining data files (`trades.csv`, `option_price.csv`, `daily_log.csv`, `position.json`, `signal.json`) are committed by GitHub Actions on the first daily run.

### 8.5 Enable GitHub Pages

In the repository settings:  
**Settings → Pages → Source: Deploy from branch `main`, folder `/dashboard`**

The dashboard will be live at `https://<username>.github.io/<repo-name>/`.

---

## 9. Daily Operation

### 9.1 GitHub Actions Schedule

- **Trigger:** `cron: '35 20 * * 1-5'` — 4:35 pm EDT (20:35 UTC), Monday–Friday
- **Manual trigger:** available via `workflow_dispatch` in GitHub UI
- **Timeout:** 20 minutes
- **Runs on:** `ubuntu-latest`

### 9.2 Pipeline Sequence

```
run_daily.py
  ├── 1. fetch_data.py
  │       ├── Download M1–M5 per-contract CSVs from CBOE
  │       ├── Interpolate → VX60d, VX90d, VF75
  │       ├── Identify M3 contract (first with DTE ≥ 60)
  │       ├── Fetch ^VVIX and ^VIX from yfinance
  │       ├── Append row to vx_history.csv
  │       ├── Recompute rolling features → append row to features.csv
  │       ├── Fetch VIX call option at M3 expiry from yfinance
  │       │       mid = (bid + ask) / 2
  │       ├── Append row to option_price.csv
  │       └── Write fetched.json
  ├── 2. eval_signal.py
  │       ├── Read fetched.json + position.json
  │       ├── If in position → evaluate 5 exit conditions
  │       ├── If not in position → evaluate 9 entry conditions
  │       └── Write signal.json
  ├── 3. manage_trade()
  │       ├── BUY → append row to trades.csv (OPEN), update position.json, reset daily_log.csv
  │       ├── SELL / Hard Deadline → update trades.csv row (close), update position.json
  │       │       If SL exit: set sl_cooldown_until = today + 20 trading days
  │       └── HOLD → no file changes
  ├── 4. append_daily_log()
  │       └── If in position: append one row to daily_log.csv with today's mid and ROI
  └── 5. gen_dashboard.py
          └── Generate dashboard/index.html from all data files
```

### 9.3 CBOE Data Fetch

Each day, `fetch_data.py` downloads the per-contract CSV for each of M1–M5 active contracts:

```
URL: https://cdn.cboe.com/data/us/futures/market_statistics/historical_data/VX/VX_{settle}.csv
```

Where `{settle}` is the settlement date in `YYYY-MM-DD` format. The file contains the full price history for that contract. The function extracts today's closing price from the row matching today's date.

**Fallback:** If CBOE download fails, the last row of `vx_history.csv` is used for VX values (prices are stable day-to-day). If the option fetch fails, the B76 theoretical price is used as the fallback mid.

### 9.4 VX Interpolation

From the downloaded contract prices:

```python
VX60d = interpolate(target=60, from=[contract DTEs and prices])
VX90d = interpolate(target=90, from=[contract DTEs and prices])
VF75  = (VX60d + VX90d) / 2
```

Linear interpolation between the two bracketing contracts for each target DTE.

### 9.5 Option Pricing

```python
strike = next_even(VF75 × 1.05)        # 5% OTM call, rounded to next even integer
expiry = M3 settlement date             # first VX contract with DTE >= 60

# When OUT of position: use today's M3 expiry
# When IN position: use expiry stored in position.json (the option actually bought)

mid = (bid + ask) / 2                   # from yfinance VIX option chain
```

If `bid > 0` and `ask > 0`: `mid = (bid + ask) / 2`  
If both zero: `mid = lastPrice` (market closed / no quote)  
If no price at all: fall back to B76 theoretical

### 9.6 Trade ROI

```
ROI = (exit_mid - entry_mid) / entry_mid × 100
```

**Total portfolio ROI** = simple additive sum of all `roi_pct` values in `trades.csv`. Never use multiplicative compounding.

### 9.7 Git Commit

After the pipeline completes, GitHub Actions commits and pushes:

```
data/vx_history.csv
data/features.csv
data/option_price.csv
data/daily_log.csv
data/trades.csv
data/position.json
data/signal.json
dashboard/index.html
```

If `git push` fails due to a concurrent push, it retries with `git pull --rebase -X theirs`.

---

## 10. Running Locally

The full pipeline can be run locally at any time:

```
python scripts/run_daily.py
```

Optional flags:
```
--skip-fetch    Skip fetch_data.py (use existing fetched.json)
--skip-signal   Skip eval_signal.py (use existing signal.json)
```

Example — regenerate dashboard without refetching data:
```
python scripts/run_daily.py --skip-fetch --skip-signal
```

---

## 11. Dashboard

`dashboard/index.html` is generated by `gen_dashboard.py` after every pipeline run.

### Sections

| Section | Contents |
|---------|----------|
| **Status bar** | Signal (WAIT / IN / ENTRY), VF75, VIX, sigma, EMA63 |
| **Entry conditions** | All 9 conditions with current value, threshold, pass/fail |
| **VF75 price chart** | Price history from 2025, with EMA63, spike level, EMA floor, trade entry/exit markers |
| **MACD chart** | MACD(5,13) with zero line |
| **Sigma chart** | sigma_now with 1.50 vol-exit threshold |
| **Position detail** | Entry VF75, sigma, entry mid, current mid, ROI, days held/left, SL regime, SL trigger price |
| **B76 Calculator** | Interactive sliders for VF75, sigma, days held; auto-computes entry mid (real price if in position, B76 theoretical if out); ROI charts vs VF75 and vs days |
| **Performance stats** | Total ROI, trades, win rate, average ROI, average hold, SL count |
| **Trade history** | Table of all closed trades |

### Calculator Behavior

- **Entry mid:** When in position = `entry_mid` from `position.json` (real market price paid). When out = B76 theoretical at current parameters.
- **Current mid:** B76 at slider values (theoretical, for what-if scenarios).
- **ROI:** (current B76 mid − entry mid) / entry mid × 100
- Moving sliders updates only the current price; entry mid stays fixed.

---

## 12. Error Recovery

| Situation | Behavior |
|-----------|---------|
| CBOE download fails | Fallback: last row of `vx_history.csv` |
| Option fetch fails | Fallback: B76 theoretical mid at M3 DTE |
| `git push` conflict | Retry with `git pull --rebase -X theirs` |
| `vx_history.csv` row already present | Skip append (idempotent) |
| `features.csv` row for today exists | Overwrite (idempotent) |
| `option_price.csv` | Always appends; duplicates possible if pipeline runs twice — review manually |

---

## 13. Environment Notes

- **Python version:** 3.11 (required for `dict | None` type hints)
- **Trading days:** Counted as Monday–Friday only; no holiday calendar
- **Timezone:** All dates are US Eastern. The effective date logic in `fetch_data.py` uses 4:35 pm EDT as the close cutoff
- **B76 formula:** Uses `scipy.stats.norm` for N(d1) and N(d2)
- **Strike rounding:** `next_even(x) = ceil(x/2) * 2` — smallest even integer ≥ x
