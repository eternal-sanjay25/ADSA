import pandas as pd
import numpy as np
from core.state import AgentState
from utils.audit import log_action


def profiling_agent(state: AgentState) -> AgentState:
    """Generate a comprehensive data profile report."""
    df = state['raw_dataset'].copy()

    if df.empty:
        raise ValueError('EMPTY_DATASET: Uploaded file contains no data.')

    profile = {
        'rows':              len(df),
        'columns':           len(df.columns),
        'column_names':      df.columns.tolist(),
        'dtypes':            df.dtypes.astype(str).to_dict(),
        'missing_count':     df.isnull().sum().to_dict(),
        'missing_pct':       (df.isnull().mean() * 100).round(2).to_dict(),
        'unique_counts':     df.nunique().to_dict(),
        'duplicate_rows':    int(df.duplicated().sum()),
        'memory_mb':         round(df.memory_usage(deep=True).sum() / 1e6, 2),
        'id_columns':        [
            c for c in df.columns
            if df[c].nunique() == len(df) and df[c].dtype == object
        ],
        'high_missing_cols': [
            c for c in df.columns
            if (df[c].isnull().mean() * 100) > 60
        ],
        'numeric_summary':   df.describe().to_dict(),
        'date_columns':      [],
    }

    # Detect date columns among object types
    for col in df.select_dtypes(include='object').columns:
        try:
            pd.to_datetime(df[col].dropna().head(100))
            profile['date_columns'].append(col)
        except Exception:
            pass

    log_action(
        state, 'ProfilingAgent', 'profiled',
        f"{profile['rows']} rows x {profile['columns']} cols",
    )
    return {**state, 'profile_report': profile}
