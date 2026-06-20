import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from core.state import AgentState
from utils.audit import log_action


def modeling_agent(state: AgentState) -> AgentState:
    """Train multiple models with cross-validation and select the best."""
    df     = state['feature_dataset']
    target = state.get('target_column', '')
    task   = state.get('task_type', 'classification')

    if target not in df.columns:
        state.setdefault('errors', []).append({
            'agent': 'ModelingAgent',
            'error': f'Target column "{target}" not found in feature dataset',
        })
        return state

    X = df.drop(columns=[target])
    y = df[target]

    # Define models
    if task == 'classification':
        models = {
            'LogisticRegression': LogisticRegression(
                max_iter=1000, random_state=42,
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=100, random_state=42,
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100, eval_metric='logloss',
                random_state=42, verbosity=0,
            ),
        }
        n_splits = min(5, int(y.value_counts().min()))
        if n_splits < 2:
            cv = KFold(n_splits=2, shuffle=True, random_state=42)
        else:
            cv = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=42,
            )
        scoring = 'f1_macro'
    else:
        models = {
            'LinearRegression': LinearRegression(),
            'RandomForest': RandomForestRegressor(
                n_estimators=100, random_state=42,
            ),
            'XGBoost': XGBRegressor(
                n_estimators=100, random_state=42, verbosity=0,
            ),
        }
        cv = KFold(
            n_splits=min(5, len(df)), shuffle=True, random_state=42,
        )
        scoring = 'neg_root_mean_squared_error'

    best_score      = float('-inf')
    best_model_name = None
    cv_results      = {}
    trained_models  = {}

    for name, model in models.items():
        try:
            scores     = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            mean_score = float(scores.mean())
            cv_results[name] = mean_score

            # Train on full data
            model.fit(X, y)
            trained_models[name] = model

            if mean_score > best_score:
                best_score      = mean_score
                best_model_name = name

            log_action(state, 'ModelingAgent', 'train_model',
                       f'{name}: {scoring}={mean_score:.4f}')
        except Exception as e:
            state.setdefault('errors', []).append({
                'agent': 'ModelingAgent', 'model': name, 'error': str(e),
            })

    if not trained_models:
        state.setdefault('errors', []).append({
            'agent': 'ModelingAgent',
            'error': 'All models failed to train',
        })
        return state

    log_action(state, 'ModelingAgent', 'best_model',
               f'{best_model_name} score={best_score:.4f}')

    return {
        **state,
        'trained_models':  trained_models,
        'best_model':      trained_models[best_model_name],
        'best_model_name': best_model_name,
        'cv_results':      cv_results,
    }
