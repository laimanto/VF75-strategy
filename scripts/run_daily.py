"""
run_daily.py  —  Master daily runner for production_manual_v33.
Called by GitHub Actions at 20:35 UTC (4:35pm EDT) Mon-Fri.

Sequence:
  1. fetch_data    — VX futures + VVIX + VIX + VIX option bid/ask
  2. eval_signal   — rule-based BUY / SELL / HOLD
  3. manage_trade  — open on BUY, close on SELL/deadline, track SL cooldown
  4. append_daily_log
  5. gen_dashboard — write dashboard/index.html
"""

import argparse, csv, json, math, sys
from datetime import date, datetime, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
DASH_DIR = BASE_DIR / 'dashboard'

TENOR     = 75
R         = 0.045
SIGMA_DEF = 0.80
SL_CALM   = -35.0
SL_VOL    = -38.0
SD84_THRESH = 1.0


def b76(F, K, T, sig, r=R):
    if T <= 0 or sig <= 0 or F <= 0:
        return max(0.0, F - K)
    from scipy.stats import norm
    d1 = (math.log(F / K) + 0.5 * sig ** 2 * T) / (sig * math.sqrt(T))
    return math.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d1 - sig * math.sqrt(T)))


def next_even(x): return int(math.ceil(x / 2) * 2)


def sl_pct(sd84): return SL_CALM if sd84 < SD84_THRESH else SL_VOL


def trading_days_after(d_str: str, n: int) -> str:
    """Return date string n trading days (weekdays) after d_str."""
    d = datetime.strptime(d_str, '%Y-%m-%d').date()
    count = 0
    while count < n:
        d += timedelta(days=1)
        if d.weekday() < 5:
            count += 1
    return d.strftime('%Y-%m-%d')


def run_step(name: str, func):
    print(f'\n{"="*55}\n  {name}\n{"="*55}')
    try:
        result = func()
        print(f'  OK  {name}')
        return result
    except Exception as e:
        import traceback
        print(f'  FAIL  {name}: {e}')
        traceback.print_exc()
        return None


def manage_trade(fetched: dict, signal_info: dict, position: dict) -> dict:
    """Open on BUY, close on SELL, handle hard deadline.  Updates files in place."""
    sig        = signal_info.get('signal', 'HOLD')
    in_pos     = position.get('in_position', False)
    today      = fetched.get('fetch_date', str(date.today()))
    vf75       = float(fetched.get('vf75', 0))
    opt_mid    = float(fetched.get('option_mid', 0))
    sigma_now  = float(fetched.get('sigma_now', SIGMA_DEF))
    roll_sd    = float(fetched.get('roll_sd') or 0)
    trades_path = DATA_DIR / 'trades.csv'

    if not in_pos and sig == 'BUY':
        # ── Open new trade ────────────────────────────────────────────────────
        strike    = next_even(vf75 * 1.05)
        sd84_now  = roll_sd
        sl_used   = sl_pct(sd84_now)
        entry_mid = float(fetched.get('option_mid', 0))   # real market mid at entry
        expiry    = fetched.get('option_expiry', '')

        rows = list(csv.DictReader(open(trades_path, encoding='utf-8')))
        next_id = max(int(r['trade_id']) for r in rows) + 1 if rows else 1

        with open(trades_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([next_id, today, round(vf75, 3), strike,
                             round(entry_mid, 4), expiry,
                             '', '', '', '', '', 'OPEN',
                             round(sd84_now, 4), round(abs(sl_used), 1), ''])

        new_pos = {
            'in_position':    True,
            'trade_id':       next_id,
            'entry_date':     today,
            'entry_vf75':     round(vf75, 3),
            'entry_sigma':    round(sigma_now, 4),
            'sd84_at_entry':  round(sd84_now, 4),
            'sl_used':        round(abs(sl_used), 1),
            'strike':         strike,
            'entry_mid':      round(entry_mid, 4),   # real market mid paid
            'expiry':         expiry,
            'tenor':          TENOR,
            'sl_cooldown_until': position.get('sl_cooldown_until'),
        }
        (DATA_DIR / 'position.json').write_text(json.dumps(new_pos, indent=2))
        (DATA_DIR / 'daily_log.csv').write_text(
            'date,vf75,vix,sigma_now,option_mid,signal,in_position,days_held,roi_mid\n')

        print(f'  AUTO-BUY: trade #{next_id}  VF75={vf75:.3f}  K={strike}'
              f'  entry_mid={entry_mid:.4f}  SL={sl_used:.0f}%  sd84={sd84_now:.3f}')
        return new_pos

    elif in_pos and (sig == 'SELL' or _hard_deadline_hit(position, today)):
        # ── Close trade ───────────────────────────────────────────────────────
        reason     = signal_info.get('exit_reason', 'HARD_DEADLINE') if sig == 'SELL' else 'HARD_DEADLINE'
        entry_mid  = float(position.get('entry_mid', 0.001))
        entry_date = position.get('entry_date', today)
        trade_id   = position.get('trade_id')
        entry_dt   = datetime.strptime(entry_date, '%Y-%m-%d').date()
        today_dt   = datetime.strptime(today, '%Y-%m-%d').date()
        days_held  = (today_dt - entry_dt).days
        opt_mid    = float(fetched.get('option_mid', opt_bid))
        roi_pct    = round((opt_mid - entry_mid) / entry_mid * 100, 2) if entry_mid > 0 else 0.0
        exit_vf75  = round(vf75, 3)

        rows      = list(csv.DictReader(open(trades_path, encoding='utf-8')))
        fieldnames = list(rows[0].keys()) if rows else []
        with open(trades_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                if int(row['trade_id']) == trade_id:
                    row['exit_date']    = today
                    row['exit_vf75']    = exit_vf75
                    row['exit_mid']     = opt_mid
                    row['days_held']    = days_held
                    row['roi_pct']      = roi_pct
                    row['exit_reason']  = reason
                writer.writerow(row)

        # SL cooldown: block entry for 20 trading days after an SL exit
        cooldown = (trading_days_after(today, 20) if reason == 'SL'
                    else position.get('sl_cooldown_until'))

        new_pos = {
            'in_position':      False,
            'last_trade_id':    trade_id,
            'last_exit_date':   today,
            'last_exit_reason': reason,
            'last_roi_pct':     roi_pct,
            'sl_cooldown_until': cooldown,
        }
        (DATA_DIR / 'position.json').write_text(json.dumps(new_pos, indent=2))
        print(f'  AUTO-{reason}: trade #{trade_id}  mid={opt_mid:.4f}'
              f'  ROI={roi_pct:+.2f}%  days={days_held}')
        if reason == 'SL':
            print(f'  SL cooldown: blocked until {cooldown}')
        return new_pos

    else:
        print(f'  [manage_trade] signal={sig}  in_pos={in_pos}  no action')
        return position


def _hard_deadline_hit(position: dict, today: str) -> bool:
    if not position.get('in_position'):
        return False
    entry_date = position.get('entry_date', today)
    entry_dt   = datetime.strptime(entry_date, '%Y-%m-%d').date()
    today_dt   = datetime.strptime(today, '%Y-%m-%d').date()
    return (today_dt - entry_dt).days >= TENOR


def append_daily_log(fetched: dict, signal_info: dict, position: dict):
    if not position.get('in_position'):
        return
    today      = fetched.get('fetch_date', str(date.today()))
    vf75       = fetched.get('vf75', 0)
    vix        = fetched.get('vix', 0)
    sigma      = fetched.get('sigma_now', SIGMA_DEF)
    opt_mid    = fetched.get('option_mid', 0)
    sig        = signal_info.get('signal', 'HOLD')
    entry_date = position.get('entry_date', today)
    entry_mid  = float(position.get('entry_mid', 0.001))
    entry_dt   = datetime.strptime(entry_date, '%Y-%m-%d').date()
    today_dt   = datetime.strptime(today, '%Y-%m-%d').date()
    days_held  = (today_dt - entry_dt).days
    roi_mid    = round((float(opt_mid) - entry_mid) / entry_mid * 100, 2) if entry_mid > 0 else 0.0

    with open(DATA_DIR / 'daily_log.csv', 'a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow([today, round(float(vf75), 3), round(float(vix), 3),
                                 round(float(sigma), 4), opt_mid,
                                 sig, True, days_held, roi_mid])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-fetch',  action='store_true')
    parser.add_argument('--skip-signal', action='store_true')
    args = parser.parse_args()

    print(f'\nVF75 Strategy — Daily Run  [{date.today()}]')
    print(f'BASE_DIR: {BASE_DIR}')

    # ── Step 1: Fetch ──────────────────────────────────────────────────────────
    if not args.skip_fetch:
        sys.path.insert(0, str(Path(__file__).parent))
        import fetch_data
        run_step('fetch_data', fetch_data.main)
    else:
        print('\n[skip] fetch_data')

    # ── Step 2: Eval signal ────────────────────────────────────────────────────
    if not args.skip_signal:
        import eval_signal
        run_step('eval_signal', eval_signal.main)
    else:
        print('\n[skip] eval_signal')

    # ── Load state ─────────────────────────────────────────────────────────────
    fetched_path  = DATA_DIR / 'fetched.json'
    signal_path   = DATA_DIR / 'signal.json'
    position_path = DATA_DIR / 'position.json'

    fetched     = json.loads(fetched_path.read_text())  if fetched_path.exists()  else {}
    signal_info = json.loads(signal_path.read_text())   if signal_path.exists()   else {'signal': 'HOLD'}
    position    = json.loads(position_path.read_text()) if position_path.exists() else {}

    # ── Step 3: Manage trade ───────────────────────────────────────────────────
    position = run_step('manage_trade',
                        lambda: manage_trade(fetched, signal_info, position))
    if position is None:
        position = json.loads(position_path.read_text())

    # Reload position (may have changed)
    position = json.loads(position_path.read_text())

    # ── Step 4: Daily log ──────────────────────────────────────────────────────
    run_step('append_daily_log', lambda: append_daily_log(fetched, signal_info, position))

    # ── Step 5: Dashboard ──────────────────────────────────────────────────────
    import gen_dashboard
    run_step('gen_dashboard', gen_dashboard.main)

    print('\nDaily run complete.')


if __name__ == '__main__':
    main()
