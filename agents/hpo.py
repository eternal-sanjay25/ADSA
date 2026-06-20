import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from core.state import AgentState
from utils.audit import log_action


def hpo_agent(state: AgentState) -> AgentState:
    """Hyperparameter optimization with Optuna Bayesian search."""
    df              = state.get('feature_dataset')
    best_model_name = state.get('best_model_name', '')
    target          = state.get('target_column', '')
    task            = state.get('task_type', 'classification')

    if df is None or not best_model_name:
        return state

    # Only tune RF and XGBoost
    if best_model_name not in ('RandomForest', 'XGBoost'):
        log_action(state, 'HPOAgent', 'skip_tuning',
                   f'{best_model_name} not tunable')
        return state

    X = df.drop(columns=[target])
    y = df[target]

    # CV setup
    if task == 'classification':
        n_splits = min(3, int(y.value_counts().min()))
        if n_splits < 2:
            cv = KFold(n_splits=2, shuffle=True, random_state=42)
        else:
            cv = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=42,
            )
        scoring = 'f1_macro'
    else:
        cv = KFold(
            n_splits=min(3, len(df)), shuffle=True, random_state=42,
        )
        scoring = 'neg_root_mean_squared_error'

    def objective(trial):
        if best_model_name == 'RandomForest':
            from sklearn.ensemble import (
                RandomForestClassifier, RandomForestRegressor,
            )
            params = {
                'n_estimators':     trial.suggest_int('n_estimators', 50, 300),
                'max_depth':        trial.suggest_int('max_depth', 3, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'random_state':     42,
            }
            model = (
                RandomForestClassifier(**params)
                if task == 'classification'
                else RandomForestRegressor(**params)
            )
        else:  # XGBoost
            from xgboost import XGBClassifier, XGBRegressor
            params = {
                'n_estimators':  trial.suggest_int('n_estimators', 50, 300),
                'max_depth':     trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample':     trial.suggest_float('subsample', 0.6, 1.0),
                'random_state':  42,
                'verbosity':     0,
            }
            if task == 'classification':
                params['eval_metric'] = 'logloss'
                model = XGBClassifier(**params)
            else:
                model = XGBRegressor(**params)

        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        return scores.mean()

    # Run optimization
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')

    try:
        study.optimize(objective, n_trials=20, show_progress_bar=False)

        current_best = state.get('cv_results', {}).get(best_model_name, float('-inf'))

        if study.best_value > current_best:
            # Retrain with best params
            if best_model_name == 'RandomForest':
                from sklearn.ensemble import (
                    RandomForestClassifier, RandomForestRegressor,
                )
                model = (
                    RandomForestClassifier(**study.best_params, random_state=42)
                    if task == 'classification'
                    else RandomForestRegressor(**study.best_params, random_state=42)
                )
            else:
                from xgboost import XGBClassifier, XGBRegressor
                extra = {'random_state': 42, 'verbosity': 0}
                if task == 'classification':
                    extra['eval_metric'] = 'logloss'
                    model = XGBClassifier(**study.best_params, **extra)
                else:
                    model = XGBRegressor(**study.best_params, **extra)

            model.fit(X, y)
            state_updates = {
                **state,
                'best_model': model,
            }
            # Update cv_results
            cv_results = dict(state.get('cv_results', {}))
            cv_results[best_model_name] = study.best_value
            state_updates['cv_results'] = cv_results

            log_action(state, 'HPOAgent', 'tuned',
                       f'score {current_best:.4f} → {study.best_value:.4f}')
            return state_updates
        else:
            log_action(state, 'HPOAgent', 'no_improvement',
                       f'best trial {study.best_value:.4f} <= current {current_best:.4f}')
    except Exception as e:
        state.setdefault('errors', []).append({
            'agent': 'HPOAgent', 'error': str(e),
        })

    return state
