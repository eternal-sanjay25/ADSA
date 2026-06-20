import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from core.state import AgentState
from utils.audit import log_action


def feature_agent(state: AgentState) -> AgentState:
    """Feature engineering: date decomposition, encoding, scaling, selection."""
    df     = state['cleaned_dataset'].copy()
    target = state.get('target_column', '')
    task   = state.get('task_type', 'classification')
    log    = []

    y = df[target].copy() if target in df.columns else None
    X = df.drop(columns=[target]) if target in df.columns else df.copy()

    # Date decomposition
    for col in X.select_dtypes(include='datetime64').columns:
        X[f'{col}_year']      = X[col].dt.year
        X[f'{col}_month']     = X[col].dt.month
        X[f'{col}_dayofweek'] = X[col].dt.dayofweek
        X[f'{col}_isweekend'] = (X[col].dt.dayofweek >= 5).astype(int)
        X = X.drop(columns=[col])
        log.append({'action': 'date_decompose', 'column': col})

    # Encoding
    for col in X.select_dtypes(include='object').columns:
        n = X[col].nunique()
        if n == 2:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            log.append({'action': 'label_encode', 'column': col})
        elif n <= 10:
            dummies = pd.get_dummies(
                X[col], prefix=col, drop_first=True,
            ).astype(int)
            X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
            log.append({'action': 'one_hot_encode', 'column': col})
        else:
            freq = X[col].value_counts(normalize=True).to_dict()
            X[col] = X[col].map(freq)
            log.append({'action': 'frequency_encode', 'column': col})

    # Scaling
    for col in X.select_dtypes(include=[np.number]).columns:
        if X[col].std() == 0:
            continue
        scaler = (
            RobustScaler()
            if abs(X[col].skew()) > 1
            else StandardScaler()
        )
        X[col] = scaler.fit_transform(X[[col]]).ravel()
        log.append({'action': 'scale', 'column': col})

    # Feature selection via mutual information
    selected = X.columns.tolist()
    if y is not None and len(X.columns) > 15:
        try:
            fn = (
                mutual_info_classif
                if task == 'classification'
                else mutual_info_regression
            )
            scores  = fn(X.fillna(0), y)
            ranking = sorted(
                zip(X.columns, scores),
                key=lambda x: x[1], reverse=True,
            )
            selected = [c for c, s in ranking[:20]]
            X = X[selected]
            log.append({
                'action': 'feature_selection',
                'method': 'mutual_info',
                'kept': len(selected),
            })
        except Exception:
            pass

    # Reassemble with target
    if y is not None:
        X[target] = y.values

    log_action(state, 'FeatureAgent', 'features_complete',
               f'{len(X.columns)} features, {len(log)} transforms')

    return {
        **state,
        'feature_dataset':  X,
        'selected_features': selected,
        'feature_log':       log,
    }
