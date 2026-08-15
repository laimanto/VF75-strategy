"""
notify.py  —  Alert when the dashboard position flips IN or OUT.

Called by run_daily.py only when position.json's `in_position` actually changed
during the run (out -> in on a BUY, in -> out on a SELL or hard deadline).
HOLD days send nothing.

Delivery is GitHub's own notification system: the run opens an issue on this
repo and assigns it to NOTIFY_ASSIGNEE. GitHub then emails that account.
Assignment is what guarantees delivery — it notifies regardless of whether the
account is "watching" the repo. The body also @-mentions the assignee as a
second, independent trigger.

No SMTP credentials and no repo secrets: Actions injects GITHUB_TOKEN
automatically. The workflow only has to grant `issues: write`.

Environment:
  GITHUB_TOKEN        provided automatically by Actions
  GITHUB_REPOSITORY   "owner/repo", set automatically by Actions
  NOTIFY_ASSIGNEE     GitHub login to assign + mention (default: laimanto)
  DASHBOARD_URL       link included in the body

Outside Actions (local runs) GITHUB_TOKEN is absent, so the functions log what
they would have posted and return False rather than raising.
"""

import csv
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'

DEFAULT_ASSIGNEE = 'laimanto'
DEFAULT_REPO     = 'laimanto/VF75-strategy'
DEFAULT_DASH     = 'https://laimanto.github.io/VF75-strategy/'
API_ROOT         = 'https://api.github.com'

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


def _table(rows):
    """Render (label, value) pairs as a GitHub markdown table."""
    out = ['| | |', '|---|---|']
    out += [f'| {label} | {"" if value is None else value} |' for label, value in rows]
    return '\n'.join(out)


def _build_entered(position, fetched, dash_url, mention):
    trade_id = position.get('trade_id')
    title = f'ENTERED position - trade #{trade_id}'
    body = f"""{mention} — the strategy has **OPENED** a position.

{_table([
    ('Signal',       '**BUY**'),
    ('Date',         fetched.get('fetch_date', '')),
    ('Trade ID',     trade_id),
    ('Entry VF75',   position.get('entry_vf75')),
    ('Strike',       position.get('strike')),
    ('Entry mid',    position.get('entry_mid')),
    ('Expiry',       position.get('expiry')),
    ('Tenor',        f"{position.get('tenor')} calendar days"),
    ('Sigma',        position.get('entry_sigma')),
    ('sd84 @ entry', position.get('sd84_at_entry')),
    ('Stop loss',    f"-{position.get('sl_used')}%"),
    ('VIX',          fetched.get('vix')),
    ('VVIX',         fetched.get('vvix')),
])}

[Open the dashboard]({dash_url})

> The model produces the signal — you still place the trade with your broker.
> Confirm the strike and expiry are quoted before entering.

Close this issue once you have placed the trade.
"""
    return title, body


def _build_exited(position, fetched, dash_url, mention, row=None):
    trade_id = position.get('last_trade_id', position.get('trade_id'))
    row      = _trade_row(trade_id) if row is None else row
    reason   = position.get('last_exit_reason') or row.get('exit_reason', '')
    roi      = position.get('last_roi_pct', row.get('roi_pct', ''))
    # Always show the sign, so a gain reads +116.79% and not a bare 116.79%.
    roi_str  = f'{roi:+}' if isinstance(roi, (int, float)) else str(roi)

    title = (f'EXITED position - trade #{trade_id} ({reason} {roi_str}%)'
             if isinstance(roi, (int, float)) else
             f'EXITED position - trade #{trade_id} ({reason})')

    body = f"""{mention} — the strategy has **CLOSED** its position.

{_table([
    ('Date',        position.get('last_exit_date') or fetched.get('fetch_date', '')),
    ('Trade ID',    trade_id),
    ('Exit reason', f"**{reason}** — {EXIT_REASONS.get(reason, 'see dashboard')}"),
    ('Days held',   f"{row.get('days_held', '')} calendar days"),
    ('ROI (mid)',   f"**{roi_str}%**"),
    ('Entry date',  row.get('entry_date', '')),
    ('Entry mid',   row.get('entry_mid', '')),
    ('Exit mid',    row.get('exit_mid', '')),
    ('Entry VF75',  row.get('entry_vf75', '')),
    ('Exit VF75',   row.get('exit_vf75', '')),
    ('Strike',      row.get('strike', '')),
    ('Expiry',      row.get('expiry', '')),
])}

[Open the dashboard]({dash_url})
"""
    cooldown = position.get('sl_cooldown_until')
    if cooldown:
        body += f'\n> **Stop-loss cooldown active** — no new entry until {cooldown}.\n'
    body += ('\n> The model produces the signal — you still close the trade with your broker.\n'
             '\nClose this issue once you have closed the trade.\n')
    return title, body


def _create_issue(repo, token, title, body, assignee):
    payload = json.dumps({
        'title':     title,
        'body':      body,
        'assignees': [assignee] if assignee else [],
    }).encode('utf-8')

    req = urllib.request.Request(
        f'{API_ROOT}/repos/{repo}/issues',
        data=payload,
        method='POST',
        headers={
            'Authorization':        f'Bearer {token}',
            'Accept':               'application/vnd.github+json',
            'X-GitHub-Api-Version': '2022-11-28',
            'Content-Type':         'application/json',
            'User-Agent':           'vf75-notify',
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode('utf-8'))


def send_position_change(new_state, position, fetched=None, signal_info=None,
                         sample=False, row=None):
    """
    Open a GitHub issue for the position flip.  `new_state` is 'IN' or 'OUT'.
    GitHub emails the assignee. Returns True if the issue was created.

    sample=True marks the issue as a drill: the title is prefixed [TEST] and a
    banner is prepended, so it can never be mistaken for a live signal.
    """
    fetched  = fetched or {}
    token    = os.environ.get('GITHUB_TOKEN', '').strip()
    repo     = os.environ.get('GITHUB_REPOSITORY', DEFAULT_REPO).strip()
    assignee = os.environ.get('NOTIFY_ASSIGNEE', DEFAULT_ASSIGNEE).strip()
    dash     = os.environ.get('DASHBOARD_URL', DEFAULT_DASH).strip()
    mention  = f'@{assignee}' if assignee else ''

    if new_state == 'IN':
        title, body = _build_entered(position, fetched, dash, mention)
    else:
        title, body = _build_exited(position, fetched, dash, mention, row=row)

    if sample:
        title = f'[TEST] {title}'
        body  = ('> [!WARNING]\n'
                 '> **Test alert — not a real signal.** No trade was opened or '
                 'closed, and no data file was changed. Triggered by hand from '
                 'the *Send test alert* workflow to check that GitHub '
                 'notifications reach your inbox. Safe to close.\n\n'
                 '---\n\n') + body

    if not token:
        print('  [notify] GITHUB_TOKEN not set (running outside Actions?) — issue skipped')
        print(f'  [notify] would have opened on {repo}: {title}')
        return False

    try:
        issue = _create_issue(repo, token, title, body, assignee)
    except urllib.error.HTTPError as e:
        detail = e.read().decode('utf-8', 'replace')[:300]
        # 403/404 here almost always means the workflow is missing
        # `permissions: issues: write`.
        print(f'  [notify] GitHub API {e.code} creating issue on {repo}: {detail}')
        raise

    print(f'  [notify] opened issue #{issue["number"]} "{title}" '
          f'-> assigned to {assignee}')
    print(f'  [notify] {issue["html_url"]}')
    return True


# ── Sample alerts ─────────────────────────────────────────────────────────────
# `python scripts/notify.py --sample entered|exited` posts a clearly-marked test
# alert. Run from Actions so the author is github-actions[bot]; GitHub suppresses
# notifications for your own activity, so a locally-posted test will not email.

SAMPLE_FETCHED = {
    'fetch_date': '2026-08-17', 'vf75': 20.178, 'vix': 19.49, 'vvix': 99.5,
}

SAMPLE_ENTERED = {
    'in_position': True, 'trade_id': 2, 'entry_date': '2026-08-17',
    'entry_vf75': 20.178, 'entry_sigma': 0.9619, 'sd84_at_entry': 1.2999,
    'sl_used': 38.0, 'strike': 22, 'entry_mid': 2.70,
    'expiry': '2026-11-18', 'tenor': 75, 'sl_cooldown_until': None,
}

SAMPLE_EXITED = {
    'in_position': False, 'last_trade_id': 1, 'last_exit_date': '2026-08-17',
    'last_exit_reason': 'SPIKE_TP', 'last_roi_pct': 116.79,
    'sl_cooldown_until': None,
}

SAMPLE_EXIT_ROW = {
    'trade_id': '1', 'entry_date': '2026-07-21', 'entry_vf75': '20.113',
    'strike': '22', 'entry_mid': '2.74', 'expiry': '2026-10-21',
    'exit_date': '2026-08-17', 'exit_vf75': '26.884', 'exit_mid': '5.94',
    'days_held': '27', 'roi_pct': '116.79', 'exit_reason': 'SPIKE_TP',
}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', choices=['entered', 'exited'], required=True,
                        help='which alert to post as a marked test')
    args = parser.parse_args()

    if args.sample == 'entered':
        ok = send_position_change('IN', SAMPLE_ENTERED, SAMPLE_FETCHED, {},
                                  sample=True)
    else:
        ok = send_position_change('OUT', SAMPLE_EXITED, SAMPLE_FETCHED, {},
                                  sample=True, row=SAMPLE_EXIT_ROW)
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
