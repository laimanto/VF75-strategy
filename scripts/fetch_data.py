"""
fetch_data.py  —  Daily data fetch for production_manual_v33.
Called by run_daily.py each trading day at ~4:35pm EDT.

Steps:
  1. Download today's VX contract prices from CBOE -> compute VX60d, VX90d, VF75
  2. Fetch ^VVIX and ^VIX from yfinance
  3. Append one row to data/vx_history.csv
  4. Compute today's rolling features -> append one row to data/features.csv
  5. Fetch VIX option bid/ask from yfinance -> append to data/option_price.csv
  6. Write data/fetched.json
"""

import csv, io, json, math, sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'

R         = 0.045
TENOR     = 75        # calendar days — used for hypothetical option expiry
SIGMA_DEF = 0.80
WM        = 84
EMA_SP    = 63
MACD_F, MACD_S = 5, 13
ATR_WIN   = 10

# CBOE per-contract CSV URL — same format as the local VX_YYYY-MM-DD.csv files.
# Each file has all trading dates for one contract up to its settlement date.
CBOE_CONTRACT_URL = (
    'https://cdn.cboe.com/data/us/futures/market_statistics/'
    'historical_data/VX/VX_{settle}.csv'
)


def ema(s, span):
    return s.ewm(span=span, adjust=False).mean()


def b76(F, K, T, sig, r=R):
    if T <= 0 or sig <= 0 or F <= 0:
        return max(0.0, F - K)
    from scipy.stats import norm
    d1 = (math.log(F / K) + 0.5 * sig ** 2 * T) / (sig * math.sqrt(T))
    return math.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d1 - sig * math.sqrt(T)))


def next_even(x):
    return int(math.ceil(x / 2) * 2)


def get_effective_date() -> date:
    """Last completed 4:35pm EDT close (skip weekends)."""
    now_utc = datetime.now(timezone.utc)
    closed  = now_utc.hour > 20 or (now_utc.hour == 20 and now_utc.minute >= 35)
    d       = now_utc.date() if closed else now_utc.date() - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def vx_settle_for_month(year: int, month: int) -> date:
    """VX settles on the Wednesday that is 30 calendar days before the 3rd Friday
    of the calendar month FOLLOWING the contract month."""
    follow_m = month + 1 if month < 12 else 1
    follow_y = year if month < 12 else year + 1
    d = date(follow_y, follow_m, 1)
    fri = 0
    while fri < 3:
        if d.weekday() == 4:
            fri += 1
        if fri < 3:
            d += timedelta(days=1)
    return d - timedelta(days=30)


def upcoming_settle_dates(ref: date, n: int = 8) -> list[tuple[date, int]]:
    """Return next n VX settlement dates >= ref with their calendar DTE from ref."""
    results = []
    y, m = ref.year, ref.month - 1
    while len(results) < n:
        m += 1
        if m > 12:
            m = 1
            y += 1
        s = vx_settle_for_month(y, m)
        if s >= ref:
            results.append((s, (s - ref).days))
    return results


def _price_from_contract_csv(text: str, ref_date: date) -> float | None:
    """
    Extract today's settle/close price from a per-contract CBOE CSV.
    Columns: Trade Date, Futures, Open, High, Low, Close, Settle, ...
    """
    try:
        df = pd.read_csv(io.StringIO(text), skip_blank_lines=True)
        df.columns = [c.strip() for c in df.columns]

        date_col = next((c for c in df.columns if 'date' in c.lower()), None)
        if date_col is None:
            return None

        df['_date'] = pd.to_datetime(df[date_col], errors='coerce').dt.date

        # Try ref_date first, then previous trading day (in case CBOE lags)
        for try_date in [ref_date, ref_date - timedelta(days=1), ref_date - timedelta(days=2)]:
            if try_date.weekday() >= 5:
                continue
            row = df[df['_date'] == try_date]
            if row.empty:
                continue
            r = row.iloc[-1]
            # Prefer Settle; fall back to Close
            price = None
            if 'Settle' in df.columns:
                price = pd.to_numeric(r.get('Settle'), errors='coerce')
            if (price is None or pd.isna(price) or price == 0) and 'Close' in df.columns:
                price = pd.to_numeric(r.get('Close'), errors='coerce')
            if price and not pd.isna(price) and price > 0:
                return float(price)
        return None
    except Exception as e:
        print(f'    [parse] {e}')
        return None


def fetch_vx_from_cboe(ref_date: date) -> dict | None:
    """
    Download individual per-contract CBOE CSV files for M1–M4 active contracts,
    extract today's settle price from each, then interpolate to VX60d / VX90d.

    Each file URL: CBOE_CONTRACT_URL.format(settle='YYYY-MM-DD')
    Same format as the local VX_YYYY-MM-DD.csv files.
    """
    settles = upcoming_settle_dates(ref_date, n=5)  # next 5 active contracts
    records = []

    for settle_dt, dte in settles:
        settle_str = settle_dt.strftime('%Y-%m-%d')
        url = CBOE_CONTRACT_URL.format(settle=settle_str)
        try:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            price = _price_from_contract_csv(resp.text, ref_date)
            if price:
                records.append({'settle': settle_dt, 'dte': dte, 'price': price})
                print(f'    VX {settle_str}  DTE={dte:3d}  price={price:.3f}')
            else:
                print(f'    VX {settle_str}  DTE={dte:3d}  no price for {ref_date}')
        except Exception as e:
            print(f'    VX {settle_str}  download failed: {e}')

    if len(records) < 2:
        print(f'  [cboe] Only {len(records)} contract(s) downloaded — need ≥ 2')
        return None

    records.sort(key=lambda x: x['dte'])
    dtes   = [r['dte']   for r in records]
    prices = [r['price'] for r in records]
    n      = len(dtes)

    def interp(target):
        if n == 1:
            return prices[0]
        for i in range(n - 1):
            if dtes[i] <= target <= dtes[i + 1]:
                w = (target - dtes[i]) / (dtes[i + 1] - dtes[i])
                return prices[i] * (1 - w) + prices[i + 1] * w
        lo, hi = (0, 1) if target < dtes[0] else (n - 2, n - 1)
        w = (target - dtes[lo]) / (dtes[hi] - dtes[lo]) if dtes[hi] != dtes[lo] else 0.0
        w = max(0.0, min(1.5, w))
        return prices[lo] * (1 - w) + prices[hi] * w

    vx60d = interp(60)
    vx90d = interp(90)
    vf75  = (vx60d + vx90d) / 2

    # M3 option expiry = first contract with DTE >= 60 (what we'd actually buy)
    m3 = next((r for r in records if r['dte'] >= 60), records[-1])
    m3_expiry = m3['settle'].strftime('%Y-%m-%d')
    print(f'  [cboe] VX60d={vx60d:.3f}  VX90d={vx90d:.3f}  VF75={vf75:.3f}  '
          f'M3={m3_expiry} (DTE={m3["dte"]})')
    return {
        'vx60d':      round(vx60d, 4),
        'vx90d':      round(vx90d, 4),
        'vf75':       round(vf75, 4),
        'm3_expiry':  m3_expiry,
    }


def fetch_vf75_from_history(ref_date: date) -> dict | None:
    """Fallback: read last row of vx_history.csv and estimate M3 expiry from calendar."""
    vx_path = DATA_DIR / 'vx_history.csv'
    if not vx_path.exists():
        return None
    df = pd.read_csv(vx_path, parse_dates=['Date'])
    df = df[df['Date'].dt.date <= ref_date]
    if df.empty:
        return None
    last = df.iloc[-1]
    print(f'  [fallback] Using last vx_history row: {last["Date"].date()}')
    # Estimate M3 expiry = first upcoming settle with DTE >= 60
    m3_expiry = next(
        (s.strftime('%Y-%m-%d') for s, dte in upcoming_settle_dates(ref_date, n=6) if dte >= 60),
        None
    )
    return {
        'vx60d':     float(last['VX60d']),
        'vx90d':     float(last['VX90d']),
        'vf75':      float(last['VF75']),
        'm3_expiry': m3_expiry,
    }


def fetch_yf_last(ticker: str) -> float | None:
    try:
        h = yf.Ticker(ticker).history(period='5d')
        if h.empty:
            return None
        return float(h['Close'].iloc[-1])
    except Exception as e:
        print(f'  [yf {ticker}] {e}')
        return None


def fetch_vix_option(strike: int, target_expiry: str) -> dict | None:
    """
    Fetch VIX call option at the given strike for the given expiry date.
    Uses the expiry closest to target_expiry in the available chain.
    Price = mid = (bid + ask) / 2.  Falls back to lastPrice if bid/ask both zero.
    Returns dict with mid, bid, ask, iv, expiry.
    """
    from datetime import datetime as dt
    target = dt.strptime(target_expiry, '%Y-%m-%d').date()

    try:
        ticker   = yf.Ticker('^VIX')
        expiries = ticker.options
        if not expiries:
            return None
        expiry = min(expiries,
                     key=lambda e: abs((dt.strptime(e, '%Y-%m-%d').date() - target).days))
        chain  = ticker.option_chain(expiry).calls
        row    = chain[chain['strike'] == float(strike)]
        if row.empty:
            row = chain.iloc[(chain['strike'] - float(strike)).abs().argsort()[:1]]
        r   = row.iloc[0]
        bid = float(r['bid'])
        ask = float(r['ask'])
        iv  = float(r['impliedVolatility'])
        lp  = float(r['lastPrice'])

        if bid > 0 and ask > 0:
            mid = (bid + ask) / 2
        elif lp > 0:
            mid = lp          # market closed / no quote — use last traded price
            bid = lp
            ask = lp
        else:
            return None       # no usable price at all

        return {
            'mid':   round(mid, 4),          # (bid+ask)/2; fallback to last price
            'last':  round(lp, 4) if lp > 0 else 0.0,  # last traded price from exchange
            'bid':   round(bid, 4),
            'ask':   round(ask, 4),
            'iv':    round(iv, 4),
            'expiry': expiry,
        }
    except Exception as e:
        print(f'  [option] Fetch failed: {e}')
        return None


def append_vx_history(ref_date: date, vx: dict, vvix: float, vix: float):
    """Append one row to vx_history.csv (skip if date already present)."""
    vx_path = DATA_DIR / 'vx_history.csv'
    if vx_path.exists():
        existing = pd.read_csv(vx_path, parse_dates=['Date'])
        if not existing[existing['Date'].dt.date == ref_date].empty:
            print(f'  vx_history: {ref_date} already present, skipping append')
            return
    row = f"{ref_date},{vx['vx60d']},{vx['vx90d']},{vx['vf75']},{round(vvix,4)},{round(vix,3)}\n"
    with open(vx_path, 'a', newline='') as f:
        f.write(row)
    print(f'  vx_history: appended {ref_date}')


def compute_and_append_features(ref_date: date) -> dict:
    """
    Load vx_history.csv, compute features for the last row, append to features.csv.
    Returns the computed feature dict for the reference date.
    """
    vx_path   = DATA_DIR / 'vx_history.csv'
    feat_path = DATA_DIR / 'features.csv'

    vx = pd.read_csv(vx_path, parse_dates=['Date'])
    vx = vx.sort_values('Date').reset_index(drop=True)

    # Need at least WM + 10 rows for rolling features
    vf = vx['VF75'].copy()
    vx['sigma_now']          = (vx['VVIX'] / 100).clip(lower=SIGMA_DEF)
    vx['ema63']              = vf.ewm(span=EMA_SP, adjust=False).mean()
    vx['macd']               = (vf.ewm(span=MACD_F, adjust=False).mean()
                                 - vf.ewm(span=MACD_S, adjust=False).mean())
    vx['roll_mu']            = vf.rolling(WM).mean()
    vx['roll_sd']            = vf.rolling(WM).std(ddof=1)
    vx['roll_rank']          = vf.rolling(WM).apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True)
    vx['roc_3m']             = vf.pct_change(63) * 100
    vx['roc_1m']             = vf.pct_change(21) * 100
    vx['vix_spread']         = vx['VIX'] - vx['VF75']
    vx['spike_level']        = vx['roll_mu'] + 2 * vx['roll_sd']
    daily_chg                = vf.diff().abs()
    vx['atr10']              = daily_chg.rolling(ATR_WIN).mean()
    chg10                    = vf.diff(10)
    vx['vf75_change_10d_atr'] = np.where(vx['atr10'] > 0,
                                          chg10 / vx['atr10'], np.nan)

    feat_cols = ['Date', 'VX60d', 'VX90d', 'VF75', 'VVIX', 'VIX', 'sigma_now',
                 'ema63', 'macd', 'roll_mu', 'roll_sd', 'roll_rank',
                 'roc_3m', 'roc_1m', 'vix_spread', 'spike_level',
                 'atr10', 'vf75_change_10d_atr']

    # Find today's row
    mask = vx['Date'].dt.date == ref_date
    if not mask.any():
        print(f'  WARNING: {ref_date} not found in vx_history after appending — skipping features')
        row = vx.iloc[-1]
    else:
        row = vx[mask].iloc[-1]

    feat_dict = {c: (None if pd.isna(row[c]) else row[c]) for c in feat_cols}
    feat_dict['Date'] = str(feat_dict['Date'])[:10]

    # Append or overwrite today's row in features.csv
    if feat_path.exists():
        existing = pd.read_csv(feat_path, parse_dates=['Date'])
        existing = existing[existing['Date'].dt.date != ref_date]
        new_row  = pd.DataFrame([feat_dict])
        new_row['Date'] = pd.to_datetime(new_row['Date'])
        out = pd.concat([existing, new_row], ignore_index=True).sort_values('Date')
    else:
        out = pd.DataFrame([feat_dict])

    out.to_csv(feat_path, index=False)
    print(f'  features.csv: updated with {ref_date}')
    return feat_dict


def main():
    ref_date   = get_effective_date()
    print(f'\nFetch  [{ref_date}]')

    # ── 1. VX futures ─────────────────────────────────────────────────────────
    print('Fetching VX futures from CBOE...')
    vx = fetch_vx_from_cboe(ref_date)
    if vx is None:
        print('  CBOE download failed — using last known VX values')
        vx = fetch_vf75_from_history(ref_date)
        if vx is None:
            print('  CRITICAL: No VX data available')
            sys.exit(1)

    # ── 2. VVIX and VIX ───────────────────────────────────────────────────────
    print('Fetching ^VVIX and ^VIX from yfinance...')
    vvix = fetch_yf_last('^VVIX') or 80.0
    vix  = fetch_yf_last('^VIX')  or vx['vf75']
    print(f'  VVIX={vvix:.2f}  VIX={vix:.2f}')

    # ── 3. Append vx_history ──────────────────────────────────────────────────
    append_vx_history(ref_date, vx, vvix, vix)

    # ── 4. Compute features ───────────────────────────────────────────────────
    feat = compute_and_append_features(ref_date)

    # ── 5. VIX option (M3 contract) ───────────────────────────────────────────
    import json
    pos_path = DATA_DIR / 'position.json'
    position = json.loads(pos_path.read_text()) if pos_path.exists() else {}
    in_pos   = position.get('in_position', False)

    sigma_now = float(feat.get('sigma_now') or SIGMA_DEF)
    vf75      = float(vx['vf75'])

    if in_pos:
        # Track the option that was actually bought at entry (expiry stored in position)
        strike        = int(position['strike'])
        entry_date    = position['entry_date']
        target_expiry = position.get('expiry') or vx.get('m3_expiry') or str(ref_date)
        entry_dt      = datetime.strptime(entry_date, '%Y-%m-%d').date()
        days_held     = (ref_date - entry_dt).days
        entry_mid     = float(position.get('entry_mid', 0))
    else:
        # Hypothetical: M3 contract we would buy today
        strike        = next_even(vf75 * 1.05)
        target_expiry = vx.get('m3_expiry') or str(ref_date)
        days_held     = 0
        entry_mid     = 0.0

    print(f'Fetching VIX option  strike={strike}  expiry={target_expiry}...')
    opt = fetch_vix_option(strike, target_expiry)

    if opt:
        opt_mid    = opt['mid']
        opt_last   = opt['last']
        opt_bid    = opt['bid']
        opt_ask    = opt['ask']
        opt_iv     = opt['iv']
        expiry     = opt['expiry']
        sigma_used = max(opt_iv, SIGMA_DEF)
        print(f'  mid={opt_mid}  last={opt_last}  bid={opt_bid}  ask={opt_ask}  iv={opt_iv:.4f}  expiry={expiry}')
    else:
        # Fallback: B76 theoretical price as mid
        dte_now    = (datetime.strptime(target_expiry, '%Y-%m-%d').date() - ref_date).days \
                     if target_expiry != str(ref_date) else TENOR
        T          = max(0, dte_now / 365)
        opt_mid    = round(b76(vf75, strike, T, sigma_now), 4)
        opt_last   = 0.0
        opt_bid    = opt_mid
        opt_ask    = opt_mid
        sigma_used = sigma_now
        expiry     = target_expiry
        print(f'  [option fallback] B76 mid={opt_mid}')

    # B76 theoretical for calculator reference
    dte_rem = (datetime.strptime(expiry, '%Y-%m-%d').date() - ref_date).days \
              if expiry not in ('estimated', str(ref_date)) else max(0, TENOR - days_held)
    b76mid = round(b76(vf75, strike, max(0, dte_rem / 365), sigma_used), 4)

    # Append to option_price.csv
    op_path = DATA_DIR / 'option_price.csv'
    with open(op_path, 'a', newline='') as f:
        csv.writer(f).writerow([ref_date, round(vf75, 3), strike, expiry,
                                 opt_mid, opt_bid, opt_ask,
                                 round(sigma_used, 4), b76mid, days_held, in_pos])
    print(f'  option_price.csv: appended {ref_date}')

    # ── 6. Write fetched.json ──────────────────────────────────────────────────
    fetched = {
        'fetch_date':    str(ref_date),
        'vf75':          round(vf75, 4),
        'vx60d':         vx['vx60d'],
        'vx90d':         vx['vx90d'],
        'vix':           round(vix, 3),
        'vvix':          round(vvix, 3),
        'sigma_now':     round(sigma_used, 4),
        'strike':        strike,
        'option_mid':    opt_mid,
        'option_last':   opt_last,
        'option_bid':    opt_bid,
        'option_ask':    opt_ask,
        'option_expiry': expiry,
        'b76_mid':       b76mid,
        'days_held':     days_held,
        'in_position':   in_pos,
        **{k: (None if feat.get(k) is None else round(float(feat[k]), 6))
           for k in ['ema63', 'macd', 'roll_mu', 'roll_sd', 'roll_rank',
                     'roc_3m', 'roc_1m', 'vix_spread', 'spike_level',
                     'vf75_change_10d_atr']},
    }
    (DATA_DIR / 'fetched.json').write_text(json.dumps(fetched, indent=2))
    print(f'  fetched.json written')
    return fetched


if __name__ == '__main__':
    main()
