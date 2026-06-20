import pandas as pd
from core.state import AgentState
from utils.llm_client import get_llm
from utils.audit import log_action

CLASSIFICATION_KEYWORDS = [
    'classify', 'churn', 'detect fraud', 'identify spam',
    'predict if', 'predict whether',
]
REGRESSION_KEYWORDS = [
    'predict amount', 'forecast value', 'estimate price',
    'predict sales', 'how much',
]
TIMESERIES_KEYWORDS = [
    'next month', 'future sales', 'forecast trend',
    'predict next', 'time series', 'over time',
]
CLUSTERING_KEYWORDS = [
    'segment', 'group customers', 'cluster',
    'find patterns', 'categorize',
]


def planner_agent(state: AgentState) -> AgentState:
    """Detect task type and target column from user goal and data shape."""
    goal = state['user_goal'].lower()
    df   = state['raw_dataset']

    scores = {
        'classification': 0,
        'regression': 0,
        'timeseries': 0,
        'clustering': 0,
    }

    for kw in CLASSIFICATION_KEYWORDS:
        if kw in goal:
            scores['classification'] += 2
    for kw in REGRESSION_KEYWORDS:
        if kw in goal:
            scores['regression'] += 2
    for kw in TIMESERIES_KEYWORDS:
        if kw in goal:
            scores['timeseries'] += 3
    for kw in CLUSTERING_KEYWORDS:
        if kw in goal:
            scores['clustering'] += 2

    # Date column detection for timeseries boost
    date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
    if date_cols:
        try:
            pd.to_datetime(df[date_cols[0]])
            scores['timeseries'] += 2
        except Exception:
            pass

    # Guess target column
    target_hint = _guess_target(df, goal)
    if target_hint:
        n_unique = df[target_hint].nunique()
        if 2 <= n_unique <= 20:
            scores['classification'] += 3
        elif n_unique > 20:
            scores['regression'] += 3

    best       = max(scores, key=scores.get)
    total      = sum(scores.values()) or 1
    confidence = round(scores[best] / total, 2)

    # LLM fallback for low confidence
    if confidence < 0.5:
        best, confidence = _llm_classify(goal, df.columns.tolist())

    log_action(state, 'PlannerAgent', 'task_detection',
               f'{best} confidence={confidence}')

    return {
        **state,
        'task_type':       best,
        'task_confidence': confidence,
        'target_column':   target_hint or '',
        'workflow_steps':  _get_workflow(best),
    }


def _guess_target(df, goal):
    """Heuristic target column detection from common names."""
    candidates = [
        'churn', 'target', 'label', 'class', 'y',
        'output', 'result', 'price', 'sales',
    ]
    for c in df.columns:
        if c.lower() in candidates or c.lower() in goal:
            return c
    return df.columns[-1]


def _get_workflow(task_type):
    """Return the ordered list of pipeline steps for the task type."""
    base = [
        'profiling', 'cleaning', 'eda', 'feature_eng',
        'modeling', 'hpo', 'evaluation', 'explainability',
        'insights', 'report',
    ]
    if task_type == 'clustering':
        return [s for s in base if s != 'explainability']
    return base


def _llm_classify(goal, columns):
    """Use LLM to classify the task when heuristics are unsure."""
    try:
        llm    = get_llm()
        prompt = (
            f'Given goal: "{goal}" and columns: {columns}, '
            'reply with one word only: classification, regression, '
            'clustering, or timeseries.'
        )
        result = llm.invoke(prompt).content.strip().lower()
        valid  = {'classification', 'regression', 'clustering', 'timeseries'}
        if result in valid:
            return result, 0.6
    except Exception:
        pass
    return 'classification', 0.5
