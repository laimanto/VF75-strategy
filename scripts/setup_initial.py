"""
setup_initial.py  —  One-time bootstrap for production_manual_v33.

Run locally once before pushing to GitHub:
  python scripts/setup_initial.py --cboe-dir <path_to_cboe_vix_futures_folder>

What it does:
  1. Processes all CBOE per-contract CSV files into vx_history.csv
     (VX60d / VX90d / VF75 via linear interpolation, same logic as build_vx_indices.py)
  2. Downloads full ^VVIX and ^VIX history from yfinance and merges
  3. Computes all rolling features -> features.csv
  4. Confirms data files exist with correct headers (does NOT overwrite)

After running, commit data/vx_history.csv and data/features.csv to the repo.
"""

import argparse, glob, math, os, sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'

# ── Strategy constants (must match eval_signal.py / gen_dashboard.py) ─────────
WM        = 84    # rolling window for rank, mu, sd
EMA_SP    = 63    # EMA span (trading days, ~3 months)
MACD_F    = 5
MACD_S    = 13
ATR_WIN   = 10
SIGMA_DEF = 0.80  # floor for sigma_now


def ema(s, span):
    return s.ewm(span=span, adjust=False).mean()


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all rolling features to a DataFrame that already has VF75, VIX, VVIX columns."""
    df = df.sort_values('Date').reset_index(drop=True)
    vf = pd.Series(df['VF75'].values, index=df.index)

    df['sigma_now']          = (df['VVIX'] / 100).clip(lower=SIGMA_DEF)
    df['ema63']              = ema(vf, EMA_SP).values
    df['macd']               = (ema(vf, MACD_F) - ema(vf, MACD_S)).values
    df['roll_mu']            = vf.rolling(WM).mean().values
    df['roll_sd']            = vf.rolling(WM).std(ddof=1).values
    df['roll_rank']          = vf.rolling(WM).apply(
                                   lambda x: (x[:-1] < x[-1]).mean(), raw=True).values
    df['roc_3m']             = vf.pct_change(63).values * 100
    df['roc_1m']             = vf.pct_change(21).values * 100
    df['vix_spread']         = df['VIX'] - df['VF75']
    df['spike_level']        = df['roll_mu'] + 2 * df['roll_sd']
    daily_chg                = vf.diff().abs()
    df['atr10']              = daily_chg.rolling(ATR_WIN).mean().values
    chg10                    = vf.diff(10).values
    with np.errstate(divide='ignore', invalid='ignore'):
        df['vf75_change_10d_atr'] = np.where(
            df['atr10'] > 0, chg10 / df['atr10'], np.nan)
    return df


def build_vx_history(cboe_dir: str) -> pd.DataFrame:
    """Load all per-contract CBOE CSV files and build VX60d/VX90d/VF75 per day."""
    print(f'Loading CBOE per-contract CSVs from {cboe_dir}...')
    all_records = []

    for fpath in sorted(glob.glob(os.path.join(cboe_dir, 'VX_*.csv'))):
        settle_str = os.path.basename(fpath).replace('VX_', '').replace('.csv', '')
        try:
            settle_dt = pd.to_datetime(settle_str)
        except Exception:
            continue
        try:
            df = pd.read_csv(fpath, skip_blank_lines=True)
            df.columns = [c.strip() for c in df.columns]
            df['trade_date'] = pd.to_datetime(df['Trade Date'], errors='coerce')
            df = df.dropna(subset=['trade_date'])

            if 'Settle' in df.columns:
                df['price'] = pd.to_numeric(df['Settle'], errors='coerce')
                zero = df['price'].isna() | (df['price'] == 0)
                if zero.any() and 'Close' in df.columns:
                    df.loc[zero, 'price'] = pd.to_numeric(df['Close'], errors='coerce')[zero]
            elif 'Close' in df.columns:
                df['price'] = pd.to_numeric(df['Close'], errors='coerce')
            else:
                continue

            df = df.dropna(subset=['price'])
            df = df[df['price'] > 0]
            for _, row in df.iterrows():
                all_records.append({
                    'trade_date':  row['trade_date'],
                    'settle_date': settle_dt,
                    'price':       float(row['price']),
                })
        except Exception as e:
            print(f'  Skip {fpath}: {e}')

    print(f'  Loaded {len(all_records):,} (date, contract) records')
    rec = pd.DataFrame(all_records).sort_values(['trade_date', 'settle_date']).reset_index(drop=True)

    rows = []
    for trade_date, grp in rec.groupby('trade_date'):
        active = grp[grp['settle_date'] >= trade_date].copy()
        active['dte'] = (active['settle_date'] - trade_date).dt.days
        active = active.sort_values('dte').reset_index(drop=True)
        if active.empty:
            continue

        dtes   = active['dte'].tolist()
        prices = active['price'].tolist()
        n      = len(dtes)

        def interp(target):
            if n == 1:
                return prices[0]
            for i in range(n - 1):
                if dtes[i] <= target <= dtes[i + 1]:
                    d1, d2 = dtes[i], dtes[i + 1]
                    p1, p2 = prices[i], prices[i + 1]
                    w = max(0.0, min(1.5, (target - d1) / (d2 - d1)))
                    return p1 * (1 - w) + p2 * w
            lo, hi = (0, 1) if target < dtes[0] else (n - 2, n - 1)
            d1, d2 = dtes[lo], dtes[hi]
            p1, p2 = prices[lo], prices[hi]
            w = max(0.0, min(1.5, (target - d1) / (d2 - d1))) if d1 != d2 else 0.0
            return p1 * (1 - w) + p2 * w

        vx60d = interp(60)
        vx90d = interp(90)
        rows.append({
            'Date':  trade_date.strftime('%Y-%m-%d'),
            'VX60d': round(vx60d, 4),
            'VX90d': round(vx90d, 4),
            'VF75':  round((vx60d + vx90d) / 2, 4),
        })

    df = pd.DataFrame(rows).sort_values('Date').reset_index(drop=True)
    print(f'  Built {len(df)} VX daily rows  ({df["Date"].iloc[0]} — {df["Date"].iloc[-1]})')
    return df


def fetch_yf_series(ticker: str, column: str = 'Close') -> pd.Series:
    print(f'  Downloading {ticker}...')
    t = yf.Ticker(ticker)
    hist = t.history(period='max')
    if hist.empty:
        raise RuntimeError(f'{ticker} returned empty history')
    s = hist[column].copy()
    s.index = pd.to_datetime(s.index.date)
    s.name = ticker
    print(f'    {len(s)} rows  ({s.index[0]} — {s.index[-1]})')
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cboe-dir', required=True,
                        help='Path to folder containing VX_YYYY-MM-DD.csv files')
    args = parser.parse_args()

    cboe_dir = args.cboe_dir
    if not os.path.isdir(cboe_dir):
        sys.exit(f'ERROR: cboe-dir not found: {cboe_dir}')

    # ── 1. Build VX history ───────────────────────────────────────────────────
    vx = build_vx_history(cboe_dir)

    # ── 2. Fetch VVIX and VIX from yfinance ──────────────────────────────────
    print('\nFetching yfinance data...')
    vvix_s = fetch_yf_series('^VVIX')
    vix_s  = fetch_yf_series('^VIX')

    # ── 3. Merge ──────────────────────────────────────────────────────────────
    vx['Date'] = pd.to_datetime(vx['Date'])
    vvix_df = vvix_s.rename('VVIX').reset_index().rename(columns={'index': 'Date'})
    vix_df  = vix_s.rename('VIX').reset_index().rename(columns={'index': 'Date'})

    df = (vx
          .merge(vvix_df, on='Date', how='left')
          .merge(vix_df,  on='Date', how='left'))

    df['VVIX'] = df['VVIX'].ffill().fillna(80.0)
    df['VIX']  = df['VIX'].ffill()
    df = df.dropna(subset=['VF75']).reset_index(drop=True)

    # ── 4. Save vx_history.csv ────────────────────────────────────────────────
    vx_path = DATA_DIR / 'vx_history.csv'
    df[['Date', 'VX60d', 'VX90d', 'VF75', 'VVIX', 'VIX']].to_csv(vx_path, index=False)
    print(f'\nSaved {vx_path}  ({len(df)} rows)')

    # ── 5. Compute features ───────────────────────────────────────────────────
    print('Computing rolling features...')
    df = compute_features(df)

    feat_cols = ['Date', 'VX60d', 'VX90d', 'VF75', 'VVIX', 'VIX', 'sigma_now',
                 'ema63', 'macd', 'roll_mu', 'roll_sd', 'roll_rank',
                 'roc_3m', 'roc_1m', 'vix_spread', 'spike_level',
                 'atr10', 'vf75_change_10d_atr']
    feat_path = DATA_DIR / 'features.csv'
    df[feat_cols].to_csv(feat_path, index=False)
    print(f'Saved {feat_path}  ({len(df)} rows)')

    # ── 6. Confirm data files (do NOT overwrite existing) ────────────────────
    headers = {
        'trades.csv':       'trade_id,entry_date,entry_vf75,strike,entry_mid,expiry,'
                            'exit_date,exit_vf75,exit_mid,days_held,roi_pct,exit_reason,'
                            'sd84_at_entry,sl_used,notes\n',
        'daily_log.csv':    'date,vf75,vix,sigma_now,option_mid,'
                            'signal,in_position,days_held,roi_mid\n',
        'option_price.csv': 'date,vf75,strike,expiry,option_mid,option_bid,option_ask,'
                            'sigma_now,b76_mid,days_held,in_position\n',
    }
    for fname, header in headers.items():
        p = DATA_DIR / fname
        if not p.exists():
            p.write_text(header)
            print(f'Created {fname}')
        else:
            print(f'Kept existing {fname}')

    import json
    for fname, content in [
        ('position.json', {"in_position": False, "sl_cooldown_until": None}),
        ('signal.json',   {"signal": "HOLD", "exit_reason": None,
                           "conditions_met": 0, "conditions_total": 8}),
    ]:
        p = DATA_DIR / fname
        if not p.exists():
            p.write_text(json.dumps(content, indent=2))
            print(f'Created {fname}')
        else:
            print(f'Kept existing {fname}')

    print('\nInitial setup complete.')
    print('Next: git add data/vx_history.csv data/features.csv data/*.csv data/*.json')
    print('      git commit -m "Initial production data"')
    print('      git push origin main')


if __name__ == '__main__':
    main()
