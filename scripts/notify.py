"""
notify.py  —  Email alert when the dashboard position flips IN or OUT.

Called by run_daily.py only when position.json's `in_position` actually changed
during the run (out -> in on a BUY, in -> out on a SELL or hard deadline).
HOLD days send nothing.

Credentials come from the environment (GitHub Actions secrets):
  GMAIL_USER          Gmail address the alert is sent from
  GMAIL_APP_PASSWORD  16-char Google App Password (NOT the account password)
  NOTIFY_TO           recipient (default: laimanto@gmail.com)
  DASHBOARD_URL       link included in the body

A missing secret must never fail the daily pipeline, so send_position_change
logs what it would have sent and returns False instead of raising.
"""

import csv
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'

DEFAULT_TO   = 'laimanto@gmail.com'
DEFAULT_DASH = 'https://laimanto.github.io/VF75-strategy/'
SMTP_HOST    = os.environ.get('SMTP_HOST', 'smtp.gmail.com')
SMTP_PORT    = int(os.environ.get('SMTP_PORT', '465'))

# Reason codes produced by eval_signal.py
EXIT_REASONS = {
    'SPIKE_TP':      'spike take-profit',
    'MACD_DECAY':    'MACD momentum decay',
    'VOL_EXIT':      'volatility exit',
    'SL':            'stop loss',
    'HARD_DEADLINE': 'hard deadline reached',
}


def _trade_row(trade_id):
    """Return the trades.csv row for trade_id, or {} if not found."""
    path = DATA_DIR / 'trades.csv'
    if not path.exists() or trade_id is None:
        return {}
    with open(path, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            if str(row.get('trade_id')) == str(trade_id):
                return row
    return {}


def _build_entered(position, fetched, dash_url):
    trade_id = position.get('trade_id')
    subject  = f'[VF75] ENTERED position - trade #{trade_id}'
    body = f"""The strategy has OPENED a position.

Signal:       BUY
Date:         {fetched.get('fetch_date', '')}
Trade ID:     {trade_id}

Entry VF75:   {position.get('entry_vf75', '')}
Strike:       {position.get('strike', '')}
Entry mid:    {position.get('entry_mid', '')}
Expiry:       {position.get('expiry', '')}
Tenor:        {position.get('tenor', '')} calendar days

Sigma:        {position.get('entry_sigma', '')}
sd84 @ entry: {position.get('sd84_at_entry', '')}
Stop loss:    -{position.get('sl_used', '')}%

VIX:          {fetched.get('vix', '')}
VVIX:         {fetched.get('vvix', '')}

Dashboard:    {dash_url}

Reminder: the model produces the signal — you still place the trade with your
broker. Confirm the strike and expiry are quoted before entering.
"""
    return subject, body


def _build_exited(position, fetched, dash_url):
    trade_id = position.get('last_trade_id', position.get('trade_id'))
    row      = _trade_row(trade_id)
    reason   = position.get('last_exit_reason') or row.get('exit_reason', '')
    roi      = position.get('last_roi_pct', row.get('roi_pct', ''))
    subject  = f'[VF75] EXITED position - trade #{trade_id} ({reason} {roi:+}%)' \
               if isinstance(roi, (int, float)) else \
               f'[VF75] EXITED position - trade #{trade_id} ({reason})'
    body = f"""The strategy has CLOSED its position.

Date:         {position.get('last_exit_date') or fetched.get('fetch_date', '')}
Trade ID:     {trade_id}
Exit reason:  {reason} - {EXIT_REASONS.get(reason, 'see dashboard')}
Days held:    {row.get('days_held', '')} calendar days
ROI (mid):    {roi}%

Entry date:   {row.get('entry_date', '')}
Entry mid:    {row.get('entry_mid', '')}
Exit mid:     {row.get('exit_mid', '')}
Entry VF75:   {row.get('entry_vf75', '')}
Exit VF75:    {row.get('exit_vf75', '')}
Strike:       {row.get('strike', '')}
Expiry:       {row.get('expiry', '')}

Dashboard:    {dash_url}
"""
    cooldown = position.get('sl_cooldown_until')
    if cooldown:
        body += (f'\nStop-loss cooldown active — no new entry until {cooldown}.\n')
    body += ('\nReminder: the model produces the signal — you still close the trade\n'
             'with your broker.\n')
    return subject, body


def send_position_change(new_state, position, fetched=None, signal_info=None):
    """
    Email the position flip.  `new_state` is 'IN' or 'OUT'.
    Returns True if the mail was accepted by the SMTP server, else False.
    """
    fetched = fetched or {}

    user = os.environ.get('GMAIL_USER', '').strip()
    pw   = os.environ.get('GMAIL_APP_PASSWORD', '').strip()
    to   = os.environ.get('NOTIFY_TO', DEFAULT_TO).strip()
    dash = os.environ.get('DASHBOARD_URL', DEFAULT_DASH).strip()

    if new_state == 'IN':
        subject, body = _build_entered(position, fetched, dash)
    else:
        subject, body = _build_exited(position, fetched, dash)

    if not user or not pw:
        print('  [notify] GMAIL_USER / GMAIL_APP_PASSWORD not set — email skipped')
        print(f'  [notify] would have sent to {to}: {subject}')
        return False

    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From']    = user
    msg['To']      = to
    msg.set_content(body)

    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=30) as smtp:
        smtp.login(user, pw)
        smtp.send_message(msg)

    print(f'  [notify] sent "{subject}" to {to}')
    return True
