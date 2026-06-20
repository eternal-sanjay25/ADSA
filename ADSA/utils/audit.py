import datetime


def log_action(state: dict, agent: str, action: str, detail: str):
    """Append an audit entry to the state's audit trail."""
    entry = {
        'agent':     agent,
        'action':    action,
        'detail':    detail,
        'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    state.setdefault('audit_trail', []).append(entry)
    print(f'[{agent}] {action}: {detail}')
