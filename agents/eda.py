import pandas as pd
import numpy as np
from core.state import AgentState
from utils.audit import log_action


def eda_agent(state: AgentState) -> AgentState:
    """Exploratory Data Analysis: numeric stats, categorical distributions,
    correlation analysis, and target distribution."""
    df     = state['cleaned_dataset']
    target = state.get('target_column', '')
    summary = {}

    # Numeric column stats
    num_df = df.select_dtypes(include=[np.number])
    summary['numeric'] = {}
    for col in num_df.columns:
        summary['numeric'][col] = {
            'mean':   round(float(num_df[col].mean()), 4),
            'median': round(float(num_df[col].median()), 4),
            'std':    round(float(num_df[col].std()), 4),
            'skew':   round(float(num_df[col].skew()), 4),
            'min':    round(float(num_df[col].min()), 4),
            'max':    round(float(num_df[col].max()), 4),
        }

    # Categorical distributions
    cat_df = df.select_dtypes(include='object')
    summary['categorical'] = {}
    for col in cat_df.columns:
        summary['categorical'][col] = (
            df[col].value_counts(normalize=True)
            .head(10).round(4).to_dict()
        )

    # Top correlation pairs
    corr = num_df.corr(method='pearson').round(4)
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append({
                'col1': cols[i],
                'col2': cols[j],
                'corr': float(corr.iloc[i, j]),
            })
    summary['top_correlations'] = sorted(
        pairs, key=lambda x: abs(x['corr']), reverse=True,
    )[:10]

    # Target distribution
    if target and target in df.columns:
        summary['target'] = {
            'column':       target,
            'unique':        int(df[target].nunique()),
            'distribution': (
                df[target].value_counts(normalize=True)
                .round(4).to_dict()
            ),
        }

    log_action(
        state, 'EDAAgent', 'eda_complete',
        f'{len(num_df.columns)} numeric, {len(cat_df.columns)} categorical',
    )
    return {**state, 'eda_summary': summary, 'correlation_matrix': corr}
