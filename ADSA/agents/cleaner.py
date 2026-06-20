import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from core.state import AgentState
from utils.audit import log_action


def cleaning_agent(state: AgentState) -> AgentState:
    """Clean the dataset: drop IDs, convert dates, remove duplicates,
    impute missing values (mean/median/KNN), and cap outliers."""
    df      = state['raw_dataset'].copy()
    profile = state['profile_report']
    log     = []
    errors  = list(state.get('errors', []))

    # 1. Drop ID columns
    for col in profile.get('id_columns', []):
        df = df.drop(columns=[col], errors='ignore')
        log.append({'action': 'drop_id_column', 'column': col})

    # 2. Convert date strings
    for col in profile.get('date_columns', []):
        try:
            df[col] = pd.to_datetime(df[col], infer_datetime_format=True)
            log.append({'action': 'convert_datetime', 'column': col})
        except Exception as e:
            errors.append({
                'agent': 'CleaningAgent', 'column': col, 'error': str(e),
            })

    # 3. Remove duplicates
    n_before = len(df)
    df = df.drop_duplicates()
    if len(df) < n_before:
        log.append({
            'action': 'drop_duplicates',
            'rows_removed': n_before - len(df),
        })

    # 4. Handle missing values
    for col in df.columns:
        miss_pct = df[col].isnull().mean() * 100
        if miss_pct == 0:
            continue
        if miss_pct > 60:
            log.append({
                'action': 'flagged_high_missing',
                'column': col,
                'missing_pct': round(miss_pct, 2),
            })
            continue

        dtype = str(df[col].dtype)
        if 'float' in dtype or 'int' in dtype:
            skew = abs(df[col].skew()) if df[col].std() > 0 else 0
            if miss_pct <= 5 and skew < 1:
                df[col] = df[col].fillna(df[col].mean())
                method = 'mean'
            elif miss_pct <= 20:
                df[col] = df[col].fillna(df[col].median())
                method = 'median'
            else:
                result = _knn_impute(df, col)
                if result is None:
                    df[col] = df[col].fillna(df[col].median())
                    method = 'median_fallback'
                else:
                    df[col] = result
                    method = 'knn'
            log.append({
                'action': 'impute', 'column': col,
                'method': method, 'missing_pct': round(miss_pct, 2),
            })
        else:
            mode_val = (
                df[col].mode()[0]
                if not df[col].mode().empty
                else 'Unknown'
            )
            df[col] = df[col].fillna(mode_val)
            log.append({
                'action': 'impute', 'column': col,
                'method': 'mode', 'missing_pct': round(miss_pct, 2),
            })

    # 5. Outlier capping (IQR 3x)
    target = state.get('target_column', '')
    for col in df.select_dtypes(include=[np.number]).columns:
        if col == target:
            continue
        try:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            if IQR == 0:
                continue
            lo, hi = Q1 - 3 * IQR, Q3 + 3 * IQR
            n_out = int(((df[col] < lo) | (df[col] > hi)).sum())
            if n_out > 0:
                df[col] = df[col].clip(lo, hi)
                log.append({
                    'action': 'clip_outliers', 'column': col, 'count': n_out,
                })
        except Exception as e:
            errors.append({
                'agent': 'CleaningAgent', 'action': 'outlier',
                'column': col, 'error': str(e),
            })

    # 6. Final null safety
    for col in df.columns:
        if df[col].isnull().any():
            fill = 0 if df[col].dtype != object else 'Unknown'
            df[col] = df[col].fillna(fill)

    log_action(state, 'CleaningAgent', 'cleaning_complete',
               f'{len(log)} actions')

    return {
        **state,
        'cleaned_dataset':   df,
        'original_snapshot':  state['raw_dataset'].copy(),
        'cleaning_log':      log,
        'errors':            errors,
    }


def _knn_impute(df, col):
    """KNN imputation for numeric columns with moderate missingness."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2 or col not in num_cols:
        return None

    features = [
        c for c in num_cols
        if c != col and df[c].isnull().sum() == 0
    ]
    if not features:
        return None

    train_mask = df[col].notna()
    if train_mask.sum() < 10:
        return None

    try:
        model = KNeighborsRegressor(
            n_neighbors=min(5, train_mask.sum()),
        )
        model.fit(
            df.loc[train_mask, features],
            df.loc[train_mask, col],
        )
        pred_mask = df[col].isnull()
        df.loc[pred_mask, col] = model.predict(
            df.loc[pred_mask, features],
        )
        return df[col]
    except Exception:
        return None
