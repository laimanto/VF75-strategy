"""
eval_signal.py  —  Rule-based signal evaluation for manual_v33 strategy.
Called by run_daily.py after fetch_data.py.

Entry: ALL 8 conditions must be met (and not in SL cooldown).
Exit: ANY of 5 conditions; SL and Spike TP have no min-hold requirement.
"""

import json, math
from datetime import date, datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'

# ── Strategy constants ─────────────────────────────────────────────────────────
PCT_THRESH   = 0.50
ROC3M_MIN    = -8.0
ROC1M_MIN    = -5.0
MAX_GAP_EMA  = 0.035   # EMA floor: VF75 must be within 3.5% below EMA63
ATR10_MIN    = -1.0
VIX_SPRD_MAX = -1.0
VOL_EXIT_SIG = 1.50
SPIKE_MULT   = 2.0
MIN_HOLD_DAYS = 20     # trading days for MACD decay and vol exit
SL_CALM      = -35.0   # % SL when sd84 < 1.0
SL_VOL       = -38.0   # % SL when sd84 >= 1.0
SD84_THRESH  = 1.0


def sl_pct(sd84: float) -> float:
    return SL_CALM if sd84 < SD84_THRESH else SL_VOL


def trading_days_between(d1_str: str, d2_str: str) -> int:
    """Approximate trading days (excludes weekends only — no holiday calendar)."""
    d1 = datetime.strptime(d1_str, '%Y-%m-%d').date()
    d2 = datetime.strptime(d2_str, '%Y-%m-%d').date()
    days = 0
    cur = min(d1, d2)
    end = max(d1, d2)
    while cur < end:
        if cur.weekday() < 5:
            days += 1
        cur = cur.fromordinal(cur.toordinal() + 1)
    return days


def evaluate_entry(f: dict, position: dict) -> tuple[bool, dict]:
    """
    Evaluate 8 entry conditions.
    Returns (all_met: bool, conditions_detail: dict).
    """
    vf75     = f.get('vf75', 0) or 0
    ema63    = f.get('ema63')
    macd     = f.get('macd')
    roll_mu  = f.get('roll_mu')
    roll_sd  = f.get('roll_sd')
    roll_rank= f.get('roll_rank')
    roc_3m   = f.get('roc_3m')
    roc_1m   = f.get('roc_1m')
    vix_sprd = f.get('vix_spread')
    atr10    = f.get('vf75_change_10d_atr')
    sigma    = f.get('sigma_now', 1.5) or 1.5

    def safe(v): return v is not None and not math.isnan(float(v))

    gap = (float(ema63) - vf75) / float(ema63) if safe(ema63) and float(ema63) > 0 else 999

    c1 = safe(roll_rank) and float(roll_rank) <= PCT_THRESH
    c2 = safe(macd)      and float(macd) > 0
    c3 = safe(roc_3m)    and float(roc_3m) >= ROC3M_MIN
    c4 = safe(roc_1m)    and float(roc_1m) >= ROC1M_MIN
    c5 = safe(ema63)     and vf75 < float(ema63)
    c6 = gap <= MAX_GAP_EMA
    c7 = safe(atr10)     and float(atr10) > ATR10_MIN
    c8 = safe(vix_sprd)  and float(vix_sprd) < VIX_SPRD_MAX

    # SL cooldown check
    cooldown_until = position.get('sl_cooldown_until')
    in_cooldown    = False
    if cooldown_until:
        try:
            cd = datetime.strptime(cooldown_until, '%Y-%m-%d').date()
            if date.today() < cd:
                in_cooldown = True
        except Exception:
            pass

    all_met = c1 and c2 and c3 and c4 and c5 and c6 and c7 and c8 and not in_cooldown

    detail = {
        'pct_rank84':          {'met': c1, 'value': round(float(roll_rank), 3) if safe(roll_rank) else None,
                                 'threshold': f'<= {PCT_THRESH}'},
        'macd_5_13':           {'met': c2, 'value': round(float(macd), 4)      if safe(macd)      else None,
                                 'threshold': '> 0'},
        'roc_3m':              {'met': c3, 'value': round(float(roc_3m), 2)    if safe(roc_3m)    else None,
                                 'threshold': f'>= {ROC3M_MIN}%'},
        'roc_1m':              {'met': c4, 'value': round(float(roc_1m), 2)    if safe(roc_1m)    else None,
                                 'threshold': f'>= {ROC1M_MIN}%'},
        'vf75_below_ema63':    {'met': c5, 'value': round(vf75, 3),
                                 'threshold': f'< EMA63 ({round(float(ema63), 3) if safe(ema63) else "?"})'},
        'ema63_gap':           {'met': c6, 'value': round(gap * 100, 2),
                                 'threshold': f'<= {MAX_GAP_EMA*100}%'},
        'atr10_filter':        {'met': c7, 'value': round(float(atr10), 3)     if safe(atr10)     else None,
                                 'threshold': f'> {ATR10_MIN}'},
        'vix_spread':          {'met': c8, 'value': round(float(vix_sprd), 2)  if safe(vix_sprd)  else None,
                                 'threshold': f'< {VIX_SPRD_MAX}'},
        'sl_cooldown':         {'met': not in_cooldown,
                                 'value': cooldown_until or 'none',
                                 'threshold': 'not in cooldown'},
    }
    return all_met, detail


def evaluate_exit(f: dict, position: dict) -> tuple[str | None, str]:
    """
    Evaluate 5 exit conditions for an open position.
    Returns (exit_signal: 'SELL' or None, reason_code: str).

    Reason codes: SPIKE_TP, MACD_DECAY, VOL_EXIT, SL, HARD_DEADLINE
    """
    vf75        = f.get('vf75', 0) or 0
    macd        = f.get('macd')
    roll_mu     = f.get('roll_mu')
    roll_sd     = f.get('roll_sd')
    sigma_now   = f.get('sigma_now', 1.5) or 1.5
    opt_mid     = f.get('option_mid', 0) or 0
    fetch_date  = f.get('fetch_date', str(date.today()))

    entry_vf75  = float(position.get('entry_vf75', 0))
    entry_date  = position.get('entry_date', fetch_date)
    entry_mid   = float(position.get('entry_mid', 0.001))
    sd84_entry  = float(position.get('sd84_at_entry', 1.0))
    sl_threshold= sl_pct(sd84_entry)  # negative, e.g. -35.0

    # Days held (calendar) for hard deadline; trading days for min-hold conditions
    entry_dt    = datetime.strptime(entry_date, '%Y-%m-%d').date()
    fetch_dt    = datetime.strptime(fetch_date, '%Y-%m-%d').date()
    cal_days    = (fetch_dt - entry_dt).days
    td_days     = trading_days_between(entry_date, fetch_date)

    def safe(v): return v is not None and not math.isnan(float(v))

    spike_level = (float(roll_mu) + SPIKE_MULT * float(roll_sd)
                   if safe(roll_mu) and safe(roll_sd) else None)

    # 1. Spike TP — no min hold
    if spike_level is not None and vf75 >= spike_level:
        return 'SELL', 'SPIKE_TP'

    # 2. SL — no min hold; uses real market mid at entry
    if entry_mid > 0:
        roi = (opt_mid - entry_mid) / entry_mid * 100
        if roi <= sl_threshold:
            return 'SELL', 'SL'

    # 3. MACD decay — min hold 20 trading days
    if (td_days >= MIN_HOLD_DAYS
            and safe(macd) and float(macd) < 0
            and vf75 < entry_vf75):
        return 'SELL', 'MACD_DECAY'

    # 4. Vol exit — min hold 20 trading days
    if td_days >= MIN_HOLD_DAYS and float(sigma_now) >= VOL_EXIT_SIG:
        return 'SELL', 'VOL_EXIT'

    # 5. Hard deadline — 75 calendar days
    if cal_days >= 75:
        return 'SELL', 'HARD_DEADLINE'

    return None, ''


def main():
    fetched_path  = DATA_DIR / 'fetched.json'
    position_path = DATA_DIR / 'position.json'
    signal_path   = DATA_DIR / 'signal.json'

    fetched  = json.loads(fetched_path.read_text())
    position = json.loads(position_path.read_text()) if position_path.exists() else {}

    in_pos = position.get('in_position', False)
    result = {}

    if in_pos:
        sell, reason = evaluate_exit(fetched, position)
        if sell:
            result = {'signal': 'SELL', 'exit_reason': reason}
            print(f'  Signal: SELL  reason={reason}')
        else:
            entry_date  = position.get('entry_date', fetched['fetch_date'])
            td_days     = trading_days_between(entry_date, fetched['fetch_date'])
            result = {'signal': 'HOLD', 'exit_reason': None,
                      'days_held_trading': td_days}
            print(f'  Signal: HOLD  (in position, {td_days} trading days)')
    else:
        all_met, detail = evaluate_entry(fetched, position)
        n_met = sum(1 for v in detail.values() if v['met'])
        if all_met:
            result = {'signal': 'BUY', 'exit_reason': None,
                      'conditions_met': n_met, 'conditions_total': len(detail),
                      'conditions': detail}
            print(f'  Signal: BUY  ({n_met}/{len(detail)} conditions met)')
        else:
            result = {'signal': 'HOLD', 'exit_reason': None,
                      'conditions_met': n_met, 'conditions_total': len(detail),
                      'conditions': detail}
            print(f'  Signal: HOLD  ({n_met}/{len(detail)} conditions met)')

    signal_path.write_text(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    main()
