"""
gen_dashboard.py  —  Generate dashboard/index.html for production_manual_v33.
Called by run_daily.py after manage_trade.

Reads:
  data/features.csv   — rolling chart data (last 500+ rows)
  data/position.json  — current position state
  data/trades.csv     — closed trade history
  data/daily_log.csv  — in-position daily ROI log
  data/signal.json    — current entry conditions

Writes:
  dashboard/index.html
"""

import csv, json, math
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
DASH_DIR = BASE_DIR          # serve index.html from repo root for GitHub Pages

# Strategy constants
R         = 0.045
TENOR     = 75
SIGMA_DEF = 0.80
WM        = 84
EMA_SP    = 63
MACD_F, MACD_S = 5, 13
K_MULT    = 1.05
MAX_GAP   = 0.035
PCT_THRESH= 0.50
ROC3M_MIN = -8.0
ROC1M_MIN = -5.0
ATR10_MIN = -1.0
SPRD_MAX  = -1.0
VOL_SIG   = 1.50
SPIKE_M   = 2.0
SL_CALM   = 35.0
SL_VOL    = 38.0
SD84_T    = 1.0


def b76(F, K, T, sig, r=R):
    if T <= 0 or sig <= 0 or F <= 0:
        return max(0.0, F - K)
    d1 = (math.log(F / K) + 0.5 * sig ** 2 * T) / (sig * math.sqrt(T))
    return math.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d1 - sig * math.sqrt(T)))


def next_even(x): return int(math.ceil(x / 2) * 2)


def js(v): return json.dumps(v)


def fmt_roi(v, digits=1):
    sign = '+' if v >= 0 else ''
    return f'{sign}{v:.{digits}f}%'


def fn(v, d=3):
    """Format number with at most d decimal places, stripping trailing zeros."""
    s = f'{v:.{d}f}'
    return s.rstrip('0').rstrip('.')


def main():
    today_str = str(date.today())

    # ── Load data files ───────────────────────────────────────────────────────
    feat_path = DATA_DIR / 'features.csv'
    if not feat_path.exists():
        raise RuntimeError('features.csv not found — run setup_initial.py first')

    feat = pd.read_csv(feat_path, parse_dates=['Date'])
    feat = feat.sort_values('Date').reset_index(drop=True)

    position  = json.loads((DATA_DIR / 'position.json').read_text()) \
                if (DATA_DIR / 'position.json').exists() else {}
    signal_j  = json.loads((DATA_DIR / 'signal.json').read_text()) \
                if (DATA_DIR / 'signal.json').exists() else {}
    fetched   = json.loads((DATA_DIR / 'fetched.json').read_text()) \
                if (DATA_DIR / 'fetched.json').exists() else {}

    trades_rows = list(csv.DictReader(open(DATA_DIR / 'trades.csv', encoding='utf-8')))
    closed      = [r for r in trades_rows if r.get('exit_date', '')]
    daily_log   = list(csv.DictReader(open(DATA_DIR / 'daily_log.csv', encoding='utf-8')))

    # ── Current snapshot (last row) ───────────────────────────────────────────
    cur          = feat.iloc[-1]
    cur_date     = str(cur['Date'])[:10]
    cur_vf75     = float(cur['VF75'])
    cur_vix      = float(cur['VIX'])
    cur_sigma    = float(cur.get('sigma_now', SIGMA_DEF))
    cur_ema63    = float(cur.get('ema63', cur_vf75))
    cur_macd     = float(cur.get('macd', 0)) if not pd.isna(cur.get('macd')) else 0.0
    cur_rr       = float(cur.get('roll_rank', 0.5)) if not pd.isna(cur.get('roll_rank')) else 0.5
    cur_roc3m    = float(cur.get('roc_3m', 0)) if not pd.isna(cur.get('roc_3m')) else 0.0
    cur_roc1m    = float(cur.get('roc_1m', 0)) if not pd.isna(cur.get('roc_1m')) else 0.0
    cur_sd84     = float(cur.get('roll_sd', 0)) if not pd.isna(cur.get('roll_sd')) else 0.0
    cur_mu84     = float(cur.get('roll_mu', cur_vf75)) if not pd.isna(cur.get('roll_mu')) else cur_vf75
    cur_atr10    = float(cur.get('vf75_change_10d_atr', 0)) if not pd.isna(cur.get('vf75_change_10d_atr')) else 0.0
    cur_spread   = float(cur.get('vix_spread', 0)) if not pd.isna(cur.get('vix_spread')) else 0.0

    cur_strike   = next_even(cur_vf75 * K_MULT)
    cur_b76mid   = b76(cur_vf75, cur_strike, TENOR / 365, cur_sigma)
    spike_level  = cur_mu84 + SPIKE_M * cur_sd84 if cur_sd84 > 0 else None
    spike_str    = fn(spike_level) if spike_level else '—'
    gap          = (cur_ema63 - cur_vf75) / cur_ema63 if cur_ema63 > 0 else 0

    sl_regime    = 'calm'    if cur_sd84 < SD84_T else 'volatile'
    sl_level     = SL_CALM  if cur_sd84 < SD84_T else SL_VOL

    # ── Position state ────────────────────────────────────────────────────────
    in_pos       = position.get('in_position', False)
    entry_vf75   = float(position.get('entry_vf75', cur_vf75)) if in_pos else cur_vf75
    entry_sigma  = float(position.get('entry_sigma', cur_sigma))
    entry_mid    = float(position.get('entry_mid', cur_b76mid)) if in_pos else cur_b76mid
    entry_strike = int(position.get('strike', cur_strike))      if in_pos else cur_strike
    entry_date   = position.get('entry_date', '—')
    entry_expiry = position.get('expiry', '—')
    sd84_entry   = float(position.get('sd84_at_entry', cur_sd84))
    sl_entry     = float(position.get('sl_used', sl_level))
    entry_sigma  = entry_sigma if in_pos else cur_sigma  # for ENTRY_SIGMA JS constant
    entry_mid    = float(position.get('entry_mid', cur_b76mid)) if in_pos else cur_b76mid

    days_held    = 0
    cal_days_left= TENOR
    hard_deadline= '—'
    cur_roi      = 0.0
    cur_mid_live = 0.0

    if in_pos and entry_date != '—':
        entry_dt    = datetime.strptime(entry_date, '%Y-%m-%d').date()
        today_dt    = datetime.strptime(cur_date, '%Y-%m-%d').date()
        days_held   = (today_dt - entry_dt).days
        cal_days_left = max(0, TENOR - days_held)
        hd_dt       = entry_dt + __import__('datetime').timedelta(days=TENOR)
        hard_deadline = hd_dt.strftime('%Y-%m-%d')

    # Current option market data — prefer fetched.json (today's real data), fall back to option_price.csv
    cur_bid          = float(fetched.get('option_bid', 0))
    cur_ask          = float(fetched.get('option_ask', 0))
    cur_option_expiry = fetched.get('option_expiry', '—')
    op_rows = list(csv.DictReader(open(DATA_DIR / 'option_price.csv', encoding='utf-8')))
    if op_rows:
        last_op      = op_rows[-1]
        cur_mid_live = float(last_op.get('option_mid', cur_b76mid))
        if cur_bid == 0:
            cur_bid = float(last_op.get('option_bid', 0))
        if cur_ask == 0:
            cur_ask = float(last_op.get('option_ask', 0))
        if cur_option_expiry == '—':
            cur_option_expiry = last_op.get('expiry', '—')
        if in_pos and entry_mid > 0:
            cur_roi = (cur_mid_live - entry_mid) / entry_mid * 100

    # Market-implied vol: back-solve from real market mid; fall back to VVIX/100
    cur_iv_mid = cur_sigma
    ref_price  = cur_mid_live if cur_mid_live > 0 else cur_b76mid
    if ref_price > 0 and cur_vf75 > 0:
        try:
            cur_iv_mid = brentq(
                lambda s: b76(cur_vf75, cur_strike, TENOR / 365, s) - ref_price,
                0.01, 5.0
            )
        except Exception:
            pass

    # Theta: daily decay as % of market price, using market-consistent IV
    cur_theta_pct = ((b76(cur_vf75, cur_strike, max(0, (TENOR - 1) / 365), cur_iv_mid)
                      - ref_price) / ref_price * 100) if ref_price > 0 else 0.0

    # ── Performance stats ─────────────────────────────────────────────────────
    total_roi = sum(float(r['roi_pct']) for r in closed if r.get('roi_pct'))
    n_trades  = len(closed)
    n_wins    = sum(1 for r in closed if float(r.get('roi_pct', 0)) > 0)
    n_sl      = sum(1 for r in closed if r.get('exit_reason') == 'SL')
    win_rate  = n_wins / n_trades * 100 if n_trades > 0 else 0
    avg_roi   = total_roi / n_trades if n_trades > 0 else 0
    avg_hold  = (sum(int(r['days_held']) for r in closed if r.get('days_held'))
                 / n_trades) if n_trades > 0 else 0

    # ── Entry condition evaluation ────────────────────────────────────────────
    cond_info = signal_j.get('conditions', {})

    def cval(key, fallback):
        c = cond_info.get(key, {})
        return c.get('value', fallback)

    def cmet(key, fallback):
        return cond_info.get(key, {}).get('met', fallback)

    c1 = cmet('pct_rank84',       cur_rr <= PCT_THRESH)
    c2 = cmet('macd_5_13',        cur_macd > 0)
    c3 = cmet('roc_3m',           cur_roc3m >= ROC3M_MIN)
    c4 = cmet('roc_1m',           cur_roc1m >= ROC1M_MIN)
    c5 = cmet('vf75_below_ema63', cur_vf75 < cur_ema63)
    c6 = cmet('ema63_gap',        gap <= MAX_GAP)
    c7 = cmet('atr10_filter',     cur_atr10 > ATR10_MIN)
    c8 = cmet('vix_spread',       cur_spread < SPRD_MAX)

    conditions = [
        (c1, 'pct_rank(84d)',     f'{cur_rr:.2f}',         f'≤ {PCT_THRESH}',      'VF75 at/below 4-month median'),
        (c2, 'MACD(5,13)',        fn(cur_macd),            '> 0',                  'Short-term momentum positive'),
        (c3, 'roc_3m',            f'{cur_roc3m:.1f}%',     f'≥ {ROC3M_MIN}%',      'No sustained 3-month decline'),
        (c4, 'roc_1m',            f'{cur_roc1m:.1f}%',     f'≥ {ROC1M_MIN}%',      'No sharp 1-month drop'),
        (c5, 'VF75 vs EMA(3m)',   fn(cur_vf75),            f'< {fn(cur_ema63)}',   'Dip below 3-month trend'),
        (c6, 'EMA gap',           f'{gap*100:.2f}%',       f'≤ {fn(MAX_GAP*100)}%', 'Not in crash / freefall'),
        (c7, 'ATR10',             fn(cur_atr10),           f'> {ATR10_MIN}',       'No strong 10-day decline'),
        (c8, 'VIX spread',        f'{cur_spread:.2f}',     f'< {SPRD_MAX}',        'Normal contango (VF75>VIX)'),
    ]
    all_entry = all([c1, c2, c3, c4, c5, c6, c7, c8]) and not in_pos

    cond_rows_html = ''
    for ok, name, val, thresh, desc in conditions:
        icon = '✓' if ok else '✗'
        cls  = 'cond-met' if ok else 'cond-not'
        cond_rows_html += f'''
    <div class="cond-row">
      <span class="cond-icon {cls}">{icon}</span>
      <span class="cond-name">{name}</span>
      <span class="cond-val">{val}</span>
      <span class="cond-thresh">{thresh}</span>
      <span class="cond-desc">{desc}</span>
    </div>'''

    # ── Chart data (2025+ or last 500 rows) ──────────────────────────────────
    h25 = feat[feat['Date'] >= '2025-01-01'].reset_index(drop=True)
    if len(h25) == 0:
        h25 = feat.tail(500).reset_index(drop=True)

    def safe_list(col, digits=3):
        return [round(float(v), digits) if not pd.isna(v) else None for v in h25[col]]

    dates_js  = js([str(d)[:10] for d in h25['Date']])
    vf75_js   = js(safe_list('VF75'))
    vix_js    = js(safe_list('VIX'))
    ema63_js  = js(safe_list('ema63'))
    floor_js  = js([round(float(v) * (1 - MAX_GAP), 3) if not pd.isna(v) else None
                    for v in h25['ema63']])
    spike_js  = js([round(float(m) + SPIKE_M * float(s), 3)
                    if not pd.isna(m) and not pd.isna(s) else None
                    for m, s in zip(h25['roll_mu'], h25['roll_sd'])])
    sigma_js  = js(safe_list('sigma_now', 4))

    # Trade markers
    entry_w_x, entry_w_y = [], []
    entry_l_x, entry_l_y = [], []
    exit_w_x,  exit_w_y  = [], []
    exit_l_x,  exit_l_y  = [], []

    for r in closed:
        roi = float(r.get('roi_pct', 0))
        ed  = r.get('entry_date', '')
        exd = r.get('exit_date', '')
        evf = float(r.get('entry_vf75', 0))
        xvf = float(r.get('exit_vf75', 0))
        if roi >= 0:
            entry_w_x.append(ed);  entry_w_y.append(evf)
            exit_w_x.append(exd);  exit_w_y.append(xvf)
        else:
            entry_l_x.append(ed);  entry_l_y.append(evf)
            exit_l_x.append(exd);  exit_l_y.append(xvf)

    if in_pos:
        entry_w_x.append(entry_date)
        entry_w_y.append(entry_vf75)

    # ── Position detail strings ───────────────────────────────────────────────
    pos_roi_str   = fmt_roi(cur_roi)   if in_pos else '—'
    pos_roi_cls   = ('green' if cur_roi >= 0 else 'red') if in_pos else 'gray'
    total_roi_str = fmt_roi(total_roi) if n_trades > 0 else '—'
    total_roi_cls = 'green' if total_roi >= 0 else 'red'

    signal_text = ('IN' if in_pos else ('ENTRY' if all_entry else 'WAIT'))
    signal_cls  = ('blue' if in_pos else ('green' if all_entry else 'orange'))
    pos_card_cls= ('c-in' if in_pos else ('c-entry' if all_entry else 'c-out'))

    try:
        from zoneinfo import ZoneInfo
        gen_ts = datetime.now(ZoneInfo('America/Toronto')).strftime('%Y-%m-%d %H:%M ET')
    except Exception:
        gen_ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')

    # ── Pre-compute display strings (avoid conditional exprs inside format specs) ─
    s_entry_vf75   = fn(entry_vf75)          if in_pos else '—'
    s_entry_sigma  = fn(entry_sigma)         if in_pos else '—'
    s_entry_mid    = f'${fn(entry_mid)}'     if in_pos else '—'
    s_cur_mid      = f'${fn(cur_mid_live)}'  if cur_mid_live > 0 else '—'
    s_bid          = f'${fn(cur_bid)}'       if cur_bid > 0 else '—'
    s_ask          = f'${fn(cur_ask)}'       if cur_ask > 0 else '—'
    show_expiry    = entry_expiry if (in_pos and entry_expiry != '—') else cur_option_expiry
    s_days_held    = str(days_held)          if in_pos else '—'
    s_days_left    = str(cal_days_left)      if in_pos else '—'
    sl_now         = sl_entry if in_pos else sl_level
    s_sl_regime    = f'−{sl_now:.0f}% ({sl_regime} regime, sd84={"entry "+fn(sd84_entry) if in_pos else fn(cur_sd84)})'
    sl_trigger_px  = max(0.0, entry_mid * (1 + (-sl_now) / 100))
    s_sl_trigger   = f'${fn(sl_trigger_px)}' if in_pos else '—'
    s_calc_entry_vf = fn(entry_vf75)        if in_pos else fn(cur_vf75)

    # ── Trade history table rows ──────────────────────────────────────────────
    trade_rows_html = ''
    if closed:
        for r in reversed(closed):
            roi   = float(r.get('roi_pct', 0))
            rcls  = 'green' if roi >= 0 else 'red'
            trade_rows_html += f'''
  <tr>
    <td>{r["trade_id"]}</td>
    <td>{r["entry_date"]}</td>
    <td>{r["entry_vf75"]}</td>
    <td>{r["strike"]}</td>
    <td>{r["entry_mid"]}</td>
    <td>{r["exit_date"]}</td>
    <td>{r["exit_vf75"]}</td>
    <td>{r["days_held"]}</td>
    <td class="{rcls}">{fmt_roi(roi)}</td>
    <td>{r["exit_reason"]}</td>
    <td>{r.get("notes","")}</td>
  </tr>'''
    else:
        trade_rows_html = '''
  <tr><td colspan="11" style="text-align:center;color:#8b949e;padding:20px;font-style:italic">
    No closed trades yet.
  </td></tr>'''

    # ── Cumulative and bar ROI data for charts ────────────────────────────────
    if closed:
        trade_ids  = [int(r['trade_id'])  for r in closed]
        trade_rois = [float(r['roi_pct']) for r in closed]
        cum_rois   = list(np.cumsum(trade_rois))
        roi_colors = ['#3fb950' if x >= 0 else '#f85149' for x in trade_rois]
    else:
        trade_ids = trade_rois = cum_rois = roi_colors = []

    # ── HTML ──────────────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VF75 Strategy | Live Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d1117;color:#c9d1d9;font-family:'Segoe UI',Arial,sans-serif;font-size:14px}}
header{{background:#161b22;border-bottom:1px solid #30363d;padding:14px 24px;display:flex;align-items:center;justify-content:space-between}}
.header-left h1{{font-size:20px;font-weight:700;color:#58a6ff}}
.header-left .sub{{font-size:12px;color:#8b949e;margin-top:3px}}
.header-right{{text-align:right;font-size:12px;color:#8b949e;line-height:1.8}}
.header-right strong{{color:#c9d1d9}}
main{{padding:18px 24px;max-width:1440px;margin:0 auto}}
.sec{{font-size:11px;font-weight:700;color:#8b949e;text-transform:uppercase;letter-spacing:.9px;margin:18px 0 10px;padding-bottom:5px;border-bottom:1px solid #21262d}}
.status-row{{display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:10px}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:13px 15px}}
.card .lbl{{font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:.7px;margin-bottom:5px}}
.card .val{{font-size:22px;font-weight:700}}
.card .hint{{font-size:11px;color:#8b949e;margin-top:4px}}
.c-out{{border-color:#30363d}}.c-wait{{border-color:#9e6a03}}
.c-entry{{border-color:#238636}}.c-in{{border-color:#1f6feb}}
.green{{color:#3fb950}}.red{{color:#f85149}}.blue{{color:#58a6ff}}
.orange{{color:#e3b341}}.white{{color:#e6edf3}}.gray{{color:#8b949e}}
.alert{{border-radius:6px;padding:9px 14px;margin-top:12px;font-size:13px;border-left:4px solid}}
.alert-wait{{background:#161200;border-color:#e3b341;color:#e3b341}}
.alert-entry{{background:#0a1f0e;border-color:#3fb950;color:#3fb950}}
.alert-in{{background:#0a1628;border-color:#1f6feb;color:#58a6ff}}
.chart-box{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px}}
.chart-row{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
.cond-panel{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px 16px}}
.cond-header{{display:grid;grid-template-columns:28px 160px 120px 90px 1fr;gap:8px;padding:0 4px 8px;border-bottom:1px solid #21262d;margin-bottom:4px}}
.cond-header span{{font-size:10px;color:#58a6ff;text-transform:uppercase;letter-spacing:.7px;font-weight:700}}
.cond-row{{display:grid;grid-template-columns:28px 160px 120px 90px 1fr;gap:8px;padding:5px 4px;border-bottom:1px solid #0d1117;align-items:center}}
.cond-row:last-child{{border-bottom:none}}
.cond-icon{{font-size:14px;font-weight:700;text-align:center}}
.cond-met{{color:#3fb950}}.cond-not{{color:#f85149}}
.cond-name{{font-size:12px;font-weight:600;color:#e6edf3}}
.cond-val{{font-size:12px;color:#58a6ff;font-family:monospace}}
.cond-thresh{{font-size:11px;color:#8b949e}}
.cond-desc{{font-size:11px;color:#8b949e;font-style:italic}}
.pos-groups{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}}
.pos-group{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px 16px}}
.pos-group h3{{font-size:11px;color:#58a6ff;text-transform:uppercase;letter-spacing:.8px;margin-bottom:10px;border-bottom:1px solid #21262d;padding-bottom:6px}}
.pos-row{{display:flex;justify-content:space-between;align-items:baseline;padding:4px 0;border-bottom:1px solid #0d1117}}
.pos-row:last-child{{border-bottom:none}}
.pos-row .k{{font-size:12px;color:#8b949e}}
.pos-row .v{{font-size:13px;font-weight:600;color:#e6edf3}}
.pos-row .v.green{{color:#3fb950}}.pos-row .v.red{{color:#f85149}}
.pos-row .v.orange{{color:#e3b341}}.pos-row .v.gray{{color:#8b949e}}
.calc-box{{background:#161b22;border:1px solid #58a6ff;border-radius:8px;padding:16px}}
.calc-grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
.calc-inputs{{display:flex;flex-direction:column;gap:16px}}
.slider-group label{{font-size:12px;color:#8b949e;display:flex;justify-content:space-between;margin-bottom:5px}}
.slider-group label span{{color:#58a6ff;font-weight:700;font-size:14px}}
input[type=range]{{width:100%;accent-color:#58a6ff;cursor:pointer}}
.calc-output{{background:#0d1117;border:1px solid #30363d;border-radius:6px;padding:14px}}
.calc-output h4{{font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:.7px;margin-bottom:10px}}
.calc-row{{display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #21262d;font-size:13px}}
.calc-row:last-child{{border-bottom:none;margin-top:6px;padding-top:8px;border-top:1px solid #30363d}}
.calc-row .ck{{color:#8b949e}}.calc-row .cv{{font-weight:700}}
.perf-row{{display:grid;grid-template-columns:repeat(6,1fr);gap:10px}}
.perf-cell{{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:10px 14px;text-align:center}}
.perf-cell .lbl{{font-size:11px;color:#8b949e;margin-bottom:4px}}
.perf-cell .val{{font-size:18px;font-weight:700}}
.tbl-wrap{{background:#161b22;border:1px solid #30363d;border-radius:8px;overflow:hidden}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{background:#161b22;color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:.6px;padding:8px 12px;border-bottom:1px solid #30363d;text-align:left}}
td{{padding:7px 12px;border-bottom:1px solid #21262d;color:#c9d1d9}}
tr:last-child td{{border-bottom:none}}
tr:hover td{{background:#161b22}}
td.green{{color:#3fb950}}td.red{{color:#f85149}}
</style>
</head>
<body>

<header>
  <div class="header-left">
    <h1>VF75 Call Strategy — Live Dashboard</h1>
    <div class="sub">manual_v33 &nbsp;|&nbsp; B76 pricing &nbsp;|&nbsp; Adaptive SL &nbsp;|&nbsp; CBOE VX futures</div>
  </div>
  <div class="header-right">
    <div>Updated: <strong>{gen_ts}</strong></div>
    <div>Data date: <strong>{cur_date}</strong></div>
  </div>
</header>
<main>

<!-- ═══ STATUS CARDS ════════════════════════════════════════════════════════ -->
<div class="sec">Market Snapshot &amp; Position Status</div>
<div class="status-row">
  <div class="card {pos_card_cls}">
    <div class="lbl">Position</div>
    <div class="val {signal_cls}">{signal_text}</div>
    <div class="hint">{'Entry #'+str(position.get('trade_id','')) if in_pos else signal_j.get('note','')}</div>
  </div>
  <div class="card c-out">
    <div class="lbl">Entry Signal</div>
    <div class="val {'green' if all_entry else 'orange'}">{sum([c1,c2,c3,c4,c5,c6,c7,c8])}/8</div>
    <div class="hint">conditions met</div>
  </div>
  <div class="card c-out">
    <div class="lbl">VF75</div>
    <div class="val white">{fn(cur_vf75)}</div>
    <div class="hint">(VX60d+VX90d)/2</div>
  </div>
  <div class="card c-out">
    <div class="lbl">VIX Spot</div>
    <div class="val white">{cur_vix:.2f}</div>
    <div class="hint">CBOE VIX index</div>
  </div>
  <div class="card c-out">
    <div class="lbl">sigma_now</div>
    <div class="val {'red' if cur_sigma>=1.5 else 'white'}">{fn(cur_sigma)}</div>
    <div class="hint">VVIX/100, floor 0.80</div>
  </div>
  <div class="card c-out">
    <div class="lbl">Strike K</div>
    <div class="val white">{cur_strike}</div>
    <div class="hint">next even ≥ VF75×1.05</div>
  </div>
  <div class="card c-out">
    <div class="lbl">Market Mid</div>
    <div class="val white">${fn(cur_mid_live)}</div>
    <div class="hint">B76 theo: ${fn(cur_b76mid)}</div>
  </div>
  <div class="card c-out">
    <div class="lbl">SL Regime</div>
    <div class="val {'orange' if sl_regime=='volatile' else 'white'}">{sl_regime}</div>
    <div class="hint">−{sl_level:.0f}%  sd84={fn(cur_sd84)}</div>
  </div>
  <div class="card c-out">
    <div class="lbl">Current ROI</div>
    <div class="val {pos_roi_cls}">{pos_roi_str}</div>
    <div class="hint">{'vs entry B76' if in_pos else 'no position'}</div>
  </div>
  <div class="card c-out">
    <div class="lbl">Total ROI</div>
    <div class="val {total_roi_cls}">{total_roi_str}</div>
    <div class="hint">additive, {n_trades} closed trade{'s' if n_trades!=1 else ''}</div>
  </div>
</div>

{'<div class="alert alert-in">IN POSITION — Trade #'+str(position.get("trade_id",""))+' &nbsp;|&nbsp; Entry '+entry_date+' &nbsp;|&nbsp; Days held: '+str(days_held)+' &nbsp;|&nbsp; SL: −'+str(sl_entry)+'%</div>' if in_pos else
 '<div class="alert alert-entry">ALL CONDITIONS MET — Entry signal active today</div>' if all_entry else
 '<div class="alert alert-wait">WAITING — '+str(sum([c1,c2,c3,c4,c5,c6,c7,c8]))+'/8 entry conditions met</div>'}

<!-- ═══ TRADE CHART ══════════════════════════════════════════════════════════ -->
<div class="sec">VF75 Price Chart (2025+)</div>
<div class="chart-box">
  <div id="priceChart" style="height:440px"></div>
</div>

<!-- ═══ ENTRY CONDITIONS ════════════════════════════════════════════════════ -->
<div class="sec">Entry Conditions (today)</div>
<div class="cond-panel">
  <div class="cond-header">
    <span></span><span>Condition</span><span>Today's Value</span>
    <span>Threshold</span><span>Purpose</span>
  </div>
  {cond_rows_html}
</div>

<!-- ═══ POSITION DETAIL ══════════════════════════════════════════════════════ -->
<div class="sec">Open Position Detail</div>
<div class="pos-groups">
  <div class="pos-group">
    <h3>VF75 &amp; Volatility</h3>
    <div class="pos-row"><span class="k">Entry VF75</span>
      <span class="v">{s_entry_vf75}</span></div>
    <div class="pos-row"><span class="k">Current VF75</span>
      <span class="v">{fn(cur_vf75)}</span></div>
    <div class="pos-row"><span class="k">Entry sigma</span>
      <span class="v">{s_entry_sigma}</span></div>
    <div class="pos-row"><span class="k">VVIX/100 (proxy σ)</span>
      <span class="v gray">{fn(cur_sigma)}</span></div>
    <div class="pos-row"><span class="k">Market IV (from mid)</span>
      <span class="v">{fn(cur_iv_mid, 4)} ({fn(cur_iv_mid * 100, 2)}%)</span></div>
    <div class="pos-row"><span class="k">Theta (decay/day)</span>
      <span class="v red">{cur_theta_pct:.2f}%/day</span></div>
    <div class="pos-row"><span class="k">Adaptive SL</span>
      <span class="v">{s_sl_regime}</span></div>
    <div class="pos-row"><span class="k">Current ROI</span>
      <span class="v {pos_roi_cls}">{pos_roi_str}</span></div>
  </div>
  <div class="pos-group">
    <h3>Option{'  (in position)' if in_pos else '  (target)'}</h3>
    <div class="pos-row"><span class="k">Strike K</span>
      <span class="v">{entry_strike if in_pos else cur_strike}</span></div>
    <div class="pos-row"><span class="k">Expiry</span>
      <span class="v">{show_expiry}</span></div>
    <div class="pos-row"><span class="k">Bid</span>
      <span class="v">{s_bid}</span></div>
    <div class="pos-row"><span class="k">Ask</span>
      <span class="v">{s_ask}</span></div>
    <div class="pos-row"><span class="k">Mid (market)</span>
      <span class="v green">{s_cur_mid}</span></div>
    <div class="pos-row"><span class="k">B76 theo</span>
      <span class="v gray">${fn(cur_b76mid)}</span></div>
    <div class="pos-row"><span class="k">Tenor</span>
      <span class="v">75 calendar days</span></div>
    <div class="pos-row"><span class="k">Entry mid (paid)</span>
      <span class="v">{s_entry_mid}</span></div>
    <div class="pos-row"><span class="k">Spike TP level</span>
      <span class="v orange">{spike_str}</span></div>
    <div class="pos-row"><span class="k">SL trigger price</span>
      <span class="v red">{s_sl_trigger}</span></div>
  </div>
  <div class="pos-group">
    <h3>Time &amp; Dates</h3>
    <div class="pos-row"><span class="k">Entry Date</span>
      <span class="v">{entry_date}</span></div>
    <div class="pos-row"><span class="k">Today</span>
      <span class="v">{cur_date}</span></div>
    <div class="pos-row"><span class="k">Days Held</span>
      <span class="v">{s_days_held}</span></div>
    <div class="pos-row"><span class="k">Days Remaining</span>
      <span class="v">{s_days_left}</span></div>
    <div class="pos-row"><span class="k">Hard Deadline</span>
      <span class="v red">{hard_deadline}</span></div>
    <div class="pos-row"><span class="k">SL Cooldown</span>
      <span class="v gray">{position.get('sl_cooldown_until') or 'none'}</span></div>
    <div class="pos-row"><span class="k">Total ROI (closed)</span>
      <span class="v {total_roi_cls}">{total_roi_str}</span></div>
  </div>
</div>

<!-- ═══ B76 CALCULATOR ═══════════════════════════════════════════════════════ -->
<div class="sec">B76 Option Calculator — Trade ROI</div>
<div class="calc-box">
  <div class="calc-grid">
    <div class="calc-inputs">
      <div style="background:#0d1117;border:1px solid #30363d;border-radius:6px;padding:10px 12px;margin-bottom:4px">
        <div style="font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:.6px;margin-bottom:8px">Entry (auto-priced)</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;align-items:end">
          <div>
            <label style="font-size:11px;color:#8b949e;display:block;margin-bottom:3px">Entry VF75</label>
            <input type="number" id="calcEntryVF" step="0.01" min="10" max="60"
              style="width:100%;box-sizing:border-box;background:#161b22;border:1px solid #58a6ff;border-radius:4px;color:#e6edf3;padding:5px 8px;font-size:13px" oninput="updateCalc()">
          </div>
          <div>
            <div style="font-size:11px;color:#8b949e;margin-bottom:3px">Entry mid price</div>
            <div id="cEntryPrice" style="background:#0d1117;border:1px solid #30363d;border-radius:4px;padding:5px 8px;font-size:13px;font-weight:700;color:#58a6ff">—</div>
          </div>
        </div>
        <div style="font-size:10px;color:#8b949e;margin-top:6px">K = next even ≥ Entry VF75×1.05 &nbsp;|&nbsp; T=75/365 &nbsp;|&nbsp; σ = sigma at entry</div>
      </div>
      <div class="slider-group">
        <label>Current VF75 <span id="calcVFVal">{cur_vf75:.3f}</span></label>
        <input type="range" id="calcVF" min="10" max="60" step="0.01" value="{cur_vf75:.3f}" oninput="updateCalc()">
      </div>
      <div class="slider-group">
        <label>Implied Vol σ (market IV) <span id="calcSigVal">{cur_iv_mid:.3f}</span></label>
        <input type="range" id="calcSig" min="0.50" max="3.00" step="0.001" value="{cur_iv_mid:.3f}" oninput="updateCalc()">
      </div>
      <div class="slider-group">
        <label>Days Held <span id="calcDaysVal">{days_held}</span></label>
        <input type="range" id="calcDays" min="0" max="75" step="1" value="{days_held}" oninput="updateCalc()">
      </div>
      <div style="font-size:12px;color:#8b949e;line-height:1.7;margin-top:4px">
        <b style="color:#c9d1d9">Pricing:</b> Black-76 &nbsp;|&nbsp; r=4.5% &nbsp;|&nbsp; Spread ±$0.01
      </div>
    </div>
    <div class="calc-output">
      <h4>Current Position Value</h4>
      <div class="calc-row"><span class="ck">Strike K</span><span class="cv white" id="cStrike">—</span></div>
      <div class="calc-row"><span class="ck">Entry mid (paid)</span><span class="cv blue" id="cEntryOut">—</span></div>
      <div class="calc-row"><span class="ck">Current B76 Mid</span><span class="cv white" id="cMid">—</span></div>
      <div class="calc-row"><span class="ck">Current Bid (sell at)</span><span class="cv green" id="cBid">—</span></div>
      <div class="calc-row"><span class="ck">Current Ask (buy at)</span><span class="cv white" id="cAsk">—</span></div>
      <div class="calc-row"><span class="ck">Days Remaining</span><span class="cv white" id="cDaysLeft">—</span></div>
      <div class="calc-row"><span class="ck">Theta (decay/day)</span><span class="cv red" id="cTheta">—</span></div>
      <div class="calc-row"><span class="ck">Delta</span><span class="cv white" id="cDelta">—</span></div>
      <div class="calc-row"><span class="ck">Gamma</span><span class="cv white" id="cGamma">—</span></div>
      <div class="calc-row"><span class="ck">Vega (per 1% IV)</span><span class="cv white" id="cVega">—</span></div>
      <div style="margin-top:12px;padding:12px 14px;background:#0a1628;border:1px solid #1f6feb;border-radius:6px;text-align:center">
        <div style="font-size:11px;color:#8b949e;letter-spacing:.6px;text-transform:uppercase;margin-bottom:4px">Current Trade ROI <span style="font-size:10px">(mid vs entry mid)</span></div>
        <div id="cROI" style="font-size:36px;font-weight:800;line-height:1.1">—</div>
      </div>
    </div>
  </div>
  <div style="margin-top:16px;display:grid;grid-template-columns:1fr 1fr;gap:14px">
    <div>
      <div style="font-size:11px;color:#8b949e;margin-bottom:6px">VF75 Sensitivity</div>
      <div id="calcVFChart" style="height:220px"></div>
    </div>
    <div>
      <div style="font-size:11px;color:#8b949e;margin-bottom:6px">Time Decay</div>
      <div id="calcDecayChart" style="height:220px"></div>
    </div>
  </div>
</div>

<!-- ═══ PERFORMANCE ══════════════════════════════════════════════════════════ -->
<div class="sec">Performance — Closed Trades</div>
<div class="perf-row">
  <div class="perf-cell">
    <div class="lbl">Closed Trades</div>
    <div class="val {'white' if n_trades>0 else 'gray'}">{n_trades}</div>
  </div>
  <div class="perf-cell">
    <div class="lbl">Win Rate</div>
    <div class="val {'green' if win_rate>=50 else 'red'}">{f'{win_rate:.0f}%' if n_trades>0 else '—'}</div>
  </div>
  <div class="perf-cell">
    <div class="lbl">Avg ROI / Trade</div>
    <div class="val {'green' if avg_roi>=0 else 'red'}">{fmt_roi(avg_roi) if n_trades>0 else '—'}</div>
  </div>
  <div class="perf-cell">
    <div class="lbl">Total ROI</div>
    <div class="val {total_roi_cls}">{total_roi_str}</div>
  </div>
  <div class="perf-cell">
    <div class="lbl">Stop Losses</div>
    <div class="val {'red' if n_sl>0 else 'gray'}">{n_sl}</div>
  </div>
  <div class="perf-cell">
    <div class="lbl">Avg Hold</div>
    <div class="val white">{f'{avg_hold:.0f}d' if n_trades>0 else '—'}</div>
  </div>
</div>

<!-- ═══ ROI CHARTS ═══════════════════════════════════════════════════════════ -->
<div class="sec">ROI Charts</div>
<div class="chart-row">
  <div class="chart-box">
    <div style="font-size:11px;color:#8b949e;margin-bottom:8px">CUMULATIVE ROI — closed trades</div>
    <div id="cumulChart" style="height:260px"></div>
  </div>
  <div class="chart-box">
    <div style="font-size:11px;color:#8b949e;margin-bottom:8px">PER-TRADE ROI — closed trades</div>
    <div id="barChart" style="height:260px"></div>
  </div>
</div>

<!-- ═══ TRADE HISTORY ════════════════════════════════════════════════════════ -->
<div class="sec">Trade History</div>
<div class="tbl-wrap">
<table>
  <thead><tr>
    <th>#</th><th>Entry</th><th>VF75</th><th>Strike</th>
    <th>B76 Paid</th><th>Exit</th><th>Exit VF75</th>
    <th>Days</th><th>ROI%</th><th>Reason</th><th>Notes</th>
  </tr></thead>
  <tbody>{trade_rows_html}</tbody>
</table>
</div>

</main>

<script>
// ── B76 ────────────────────────────────────────────────────────────────────────
function normCDF(x){{
  const t=1/(1+0.2316419*Math.abs(x)),d=0.3989423*Math.exp(-x*x/2);
  let p=d*t*(0.3193815+t*(-0.3565638+t*(1.7814779+t*(-1.8212560+t*1.3302744))));
  return x>0?1-p:p;
}}
function b76(F,K,T,sig,r){{
  if(T<=0) return Math.max(0,F-K);
  if(sig<=0||F<=0) return 0;
  const d1=(Math.log(F/K)+0.5*sig*sig*T)/(sig*Math.sqrt(T));
  return Math.exp(-r*T)*(F*normCDF(d1)-K*normCDF(d1-sig*Math.sqrt(T)));
}}
function nextEven(x){{return Math.ceil(x/2)*2;}}
function bid(m){{return Math.max(0,m-0.01);}}
function ask_(m){{return m+0.01;}}
function normPDF(x){{return Math.exp(-0.5*x*x)/Math.sqrt(2*Math.PI);}}
function fmtROI(v){{const s=v>=0?'+':'';return `<span style="color:${{v>=0?'#3fb950':'#f85149'}}">${{s}}${{v.toFixed(1)}}%</span>`;}}
function fmtN(v,d=3){{return parseFloat(v.toFixed(d)).toString();}}

const R=0.045, TENOR=75;
const CUR_VF={cur_vf75:.3f};
const VVIX_SIG={cur_sigma:.3f};      // VVIX/100 — used for strategy signal conditions
const CUR_IV_MID={cur_iv_mid:.4f};  // market-implied vol from real mid price
// ENTRY_SIGMA: actual sigma recorded at entry; defaults to VVIX/100 when OUT
const ENTRY_SIGMA={entry_sigma:.3f};
// ENTRY_MID_FIXED: real market mid paid at entry (0 = not in position)
const ENTRY_MID_FIXED={entry_mid:.4f};
// CUR_MID_MARKET: today's real market mid from option_price.csv
const CUR_MID_MARKET={cur_mid_live:.4f};

// ── Trade chart data ───────────────────────────────────────────────────────────
const dates     = {dates_js};
const vf75arr   = {vf75_js};
const vixArr    = {vix_js};
const ema63arr  = {ema63_js};
const ema63floor= {floor_js};
const spikeLvl  = {spike_js};

// ── Entry/exit markers ─────────────────────────────────────────────────────────
const entryWx={js(entry_w_x)}, entryWy={js(entry_w_y)};
const entryLx={js(entry_l_x)}, entryLy={js(entry_l_y)};
const exitWx ={js(exit_w_x)},  exitWy ={js(exit_w_y)};
const exitLx ={js(exit_l_x)},  exitLy ={js(exit_l_y)};

// ── Price chart ────────────────────────────────────────────────────────────────
(function(){{
  const hover=dates.map((d,i)=>
    `${{d}}<br>VF75: ${{vf75arr[i]!=null?fmtN(vf75arr[i]):'—'}}<br>VIX: ${{vixArr[i]?.toFixed(2)}}<br>`+
    `EMA(3m): ${{ema63arr[i]!=null?fmtN(ema63arr[i]):'—'}}<br>Spike: ${{spikeLvl[i]!=null?fmtN(spikeLvl[i]):'—'}}`);
  const traces=[
    {{x:dates,y:vixArr,mode:'lines',name:'VIX',
      line:{{color:'rgba(200,200,200,0.28)',width:1.2}},hoverinfo:'skip',showlegend:true}},
    {{x:dates,y:ema63floor,mode:'lines',name:'EMA floor (−3.5%)',
      line:{{color:'rgba(227,179,65,0.40)',width:1.2,dash:'dash'}},hoverinfo:'skip'}},
    {{x:dates,y:spikeLvl,mode:'lines',name:'Spike TP',
      line:{{color:'rgba(63,185,80,0.35)',width:1.2,dash:'dot'}},hoverinfo:'skip'}},
    {{x:dates,y:ema63arr,mode:'lines',name:'EMA(3m)',
      line:{{color:'#e3b341',width:1.4,dash:'dot'}},hoverinfo:'skip'}},
    {{x:dates,y:vf75arr,mode:'lines',name:'VF75',
      line:{{color:'#58a6ff',width:2}},
      hovertemplate:'%{{customdata}}<extra></extra>',customdata:hover}},
    {{x:entryWx,y:entryWy,mode:'markers',name:'Entry (win)',
      marker:{{symbol:'triangle-up',size:10,color:'#3fb950'}}}},
    {{x:entryLx,y:entryLy,mode:'markers',name:'Entry (loss)',
      marker:{{symbol:'triangle-up',size:10,color:'#f85149'}}}},
    {{x:exitWx, y:exitWy, mode:'markers',name:'Exit (win)',
      marker:{{symbol:'triangle-down',size:10,color:'#3fb950'}}}},
    {{x:exitLx, y:exitLy, mode:'markers',name:'Exit (loss)',
      marker:{{symbol:'triangle-down',size:10,color:'#f85149'}}}},
  ];
  const layout={{
    paper_bgcolor:'#161b22',plot_bgcolor:'#0d1117',
    font:{{color:'#c9d1d9',size:11}},
    xaxis:{{gridcolor:'#21262d',rangeslider:{{visible:true,thickness:0.04}},type:'date'}},
    yaxis:{{title:'Price',gridcolor:'#21262d'}},
    legend:{{orientation:'h',y:-0.18,font:{{size:10}}}},
    margin:{{t:30,r:20,b:60,l:55}},hovermode:'closest',
    annotations:[{{
      xref:'paper',yref:'paper',x:0.01,y:0.97,xanchor:'left',yanchor:'top',
      text:`VF75 {fn(cur_vf75)} | VIX {cur_vix:.2f} | EMA {fn(cur_ema63)} | σ {fn(cur_sigma)} | Spike {spike_str}`,
      showarrow:false,font:{{size:10,color:'#8b949e'}},
      bgcolor:'rgba(13,17,23,0.7)',bordercolor:'#30363d',borderwidth:1
    }}]
  }};
  Plotly.newPlot('priceChart',traces,layout,{{responsive:true}});
}})();

// ── ROI charts ──────────────────────────────────────────────────────────────────
(function(){{
  const tradeIds  = {js(trade_ids)};
  const tradeRois = {js(trade_rois)};
  const cumRois   = {js(cum_rois)};
  const roiColors = {js(roi_colors)};
  const emptyLayout={{
    paper_bgcolor:'#161b22',plot_bgcolor:'#0d1117',font:{{color:'#c9d1d9',size:11}},
    xaxis:{{gridcolor:'#21262d'}},yaxis:{{gridcolor:'#21262d',ticksuffix:'%'}},
    margin:{{t:10,r:15,b:45,l:65}},
    annotations:[{{xref:'paper',yref:'paper',x:0.5,y:0.5,
      text:'No closed trades yet',showarrow:false,font:{{size:14,color:'#8b949e'}}}}]
  }};
  if(tradeIds.length>0){{
    Plotly.newPlot('cumulChart',[{{x:tradeIds,y:cumRois,mode:'lines+markers',
      line:{{color:'#58a6ff',width:2}},marker:{{color:roiColors,size:7}},
      hovertemplate:'Trade %{{x}}<br>Cumulative: %{{y:.1f}}%<extra></extra>'}}],
      {{...emptyLayout,yaxis:{{...emptyLayout.yaxis,title:'Cumulative ROI %'}}}},{{responsive:true}});
    Plotly.newPlot('barChart',[{{x:tradeIds,y:tradeRois,type:'bar',
      marker:{{color:roiColors}},
      hovertemplate:'Trade %{{x}}<br>ROI: %{{y:.1f}}%<extra></extra>'}}],
      {{...emptyLayout,yaxis:{{...emptyLayout.yaxis,title:'ROI %'}}}},{{responsive:true}});
  }} else {{
    Plotly.newPlot('cumulChart',[],emptyLayout,{{responsive:true}});
    Plotly.newPlot('barChart',[],emptyLayout,{{responsive:true}});
  }}
}})();

// ── B76 Calculator ─────────────────────────────────────────────────────────────
function updateCalc(){{
  const F  =+document.getElementById('calcVF').value;
  const sig=+document.getElementById('calcSig').value;
  const d  =+document.getElementById('calcDays').value;
  document.getElementById('calcVFVal').textContent  =fmtN(F);
  document.getElementById('calcSigVal').textContent =fmtN(sig);
  document.getElementById('calcDaysVal').textContent=d;

  const eVF=parseFloat(document.getElementById('calcEntryVF').value)||CUR_VF;
  const K  =nextEven(eVF*1.05);
  // Entry mid: real price paid (in position), today's market mid (out), or B76 fallback
  const entryMid = ENTRY_MID_FIXED > 0 ? ENTRY_MID_FIXED
                 : CUR_MID_MARKET   > 0 ? CUR_MID_MARKET
                 : b76(eVF,K,TENOR/365,ENTRY_SIGMA,R);
  document.getElementById('cEntryPrice').textContent='$'+fmtN(entryMid);
  document.getElementById('cStrike').textContent=K;
  const entryLabel = ENTRY_MID_FIXED > 0
    ? '$'+fmtN(entryMid)+' (market mid paid at entry)'
    : CUR_MID_MARKET > 0
    ? '$'+fmtN(entryMid)+' (current market mid, K='+K+')'
    : '$'+fmtN(entryMid)+' (B76 theoretical, K='+K+')';
  document.getElementById('cEntryOut').textContent=entryLabel;

  const T  =Math.max(0,(TENOR-d)/365);
  const mid=b76(F,K,T,sig,R);
  const b  =bid(mid),a=ask_(mid);
  const th =T>1/365?((b76(F,K,T-1/365,sig,R)-mid)/mid*100):0;
  const sqrtT=T>0?Math.sqrt(T):0;
  const d1  =T>0?(Math.log(F/K)+0.5*sig*sig*T)/(sig*sqrtT):Infinity;
  const delta=T>0?Math.exp(-R*T)*normCDF(d1):(F>K?1:0);
  const gamma=T>0?Math.exp(-R*T)*normPDF(d1)/(F*sig*sqrtT):0;
  const vega =T>0?F*Math.exp(-R*T)*normPDF(d1)*sqrtT*0.01:0; // per 1% IV move
  document.getElementById('cMid').textContent    ='$'+fmtN(mid);
  document.getElementById('cBid').textContent    ='$'+fmtN(b);
  document.getElementById('cAsk').textContent    ='$'+fmtN(a);
  document.getElementById('cDaysLeft').textContent=Math.max(0,TENOR-d)+' days';
  document.getElementById('cTheta').textContent  =th.toFixed(2)+'%/day';
  document.getElementById('cDelta').textContent  =fmtN(delta,4);
  document.getElementById('cGamma').textContent  =fmtN(gamma,5);
  document.getElementById('cVega').textContent   ='$'+fmtN(vega,4);
  document.getElementById('cROI').innerHTML=fmtROI((mid-entryMid)/entryMid*100);

  // VF75 sensitivity chart (mid prices; ROI vs entry mid)
  const vr=[],vb=[],vROI=[];
  for(let v=10;v<=60;v+=0.25){{
    vr.push(+v.toFixed(2));
    const vm=b76(v,K,T,sig,R);
    vb.push(+vm.toFixed(4));
    vROI.push(+((vm-entryMid)/entryMid*100).toFixed(1));
  }}
  const shapes1=[
    {{type:'line',xref:'x',yref:'paper',x0:F,x1:F,y0:0,y1:1,line:{{color:'#e3b341',width:1.5,dash:'dot'}}}},
    {{type:'line',xref:'x',yref:'paper',x0:eVF,x1:eVF,y0:0,y1:1,line:{{color:'#58a6ff',width:1,dash:'dash'}}}},
  ];
  Plotly.newPlot('calcVFChart',[
    {{x:vr,y:vb,mode:'lines',name:'Mid',line:{{color:'#3fb950',width:2}},
      hovertemplate:'VF75 %{{x}}<br>Mid $%{{y:.3g}}<br>ROI %{{customdata}}%<extra></extra>',customdata:vROI}},
    {{x:[10,60],y:[entryMid,entryMid],mode:'lines',name:'Entry mid',
      line:{{color:'#58a6ff',width:1.2,dash:'dot'}},hoverinfo:'skip'}},
  ],{{paper_bgcolor:'#161b22',plot_bgcolor:'#0d1117',font:{{color:'#c9d1d9',size:11}},
    xaxis:{{title:'VF75',gridcolor:'#21262d'}},yaxis:{{title:'Price ($)',gridcolor:'#21262d',tickprefix:'$'}},
    legend:{{orientation:'h',y:-0.35,font:{{size:10}}}},
    margin:{{t:10,r:15,b:70,l:60}},hovermode:'x unified',shapes:shapes1,
    annotations:[
      {{xref:'x',yref:'paper',x:F,y:1.05,text:'Now',showarrow:false,font:{{color:'#e3b341',size:9}},xanchor:'center'}},
      {{xref:'x',yref:'paper',x:eVF,y:1.05,text:'Entry',showarrow:false,font:{{color:'#58a6ff',size:9}},xanchor:'center'}},
    ]}},{{responsive:true}});

  // Decay chart (mid prices; ROI vs entry mid)
  const dx=[],db=[],dROI=[];
  for(let dd=0;dd<=TENOR;dd++){{
    dx.push(dd);
    const dm=b76(F,K,Math.max(0,(TENOR-dd)/365),sig,R);
    db.push(+dm.toFixed(4));
    dROI.push(+((dm-entryMid)/entryMid*100).toFixed(1));
  }}
  Plotly.newPlot('calcDecayChart',[
    {{x:dx,y:db,mode:'lines',name:'Mid',line:{{color:'#3fb950',width:2}},
      hovertemplate:'Day %{{x}}<br>Mid $%{{y:.3g}}<br>ROI %{{customdata}}%<extra></extra>',customdata:dROI}},
    {{x:[0,TENOR],y:[entryMid,entryMid],mode:'lines',name:'Entry mid',
      line:{{color:'#58a6ff',width:1.2,dash:'dot'}},hoverinfo:'skip'}},
  ],{{paper_bgcolor:'#161b22',plot_bgcolor:'#0d1117',font:{{color:'#c9d1d9',size:11}},
    xaxis:{{title:'Days Held',gridcolor:'#21262d'}},yaxis:{{title:'Price ($)',gridcolor:'#21262d',tickprefix:'$'}},
    legend:{{orientation:'h',y:-0.35,font:{{size:10}}}},margin:{{t:10,r:15,b:70,l:60}},
    hovermode:'x unified',
    shapes:[{{type:'line',x0:d,x1:d,y0:0,y1:1,xref:'x',yref:'paper',
      line:{{color:'#e3b341',width:1.5,dash:'dot'}}}}],
    annotations:[{{xref:'x',yref:'paper',x:d,y:1.05,text:'Today',
      showarrow:false,font:{{color:'#e3b341',size:9}},xanchor:'center'}}]
  }},{{responsive:true}});
}}

// Init calculator
document.getElementById('calcEntryVF').value={s_calc_entry_vf};
document.getElementById('calcVF').value      ={cur_vf75:.3f};
document.getElementById('calcSig').value     ={cur_iv_mid:.4f};
updateCalc();
</script>
</body>
</html>"""

    out_path = DASH_DIR / 'index.html'
    out_path.write_text(html, encoding='utf-8')
    print(f'  Dashboard written: {out_path}')
    return str(out_path)


if __name__ == '__main__':
    main()
