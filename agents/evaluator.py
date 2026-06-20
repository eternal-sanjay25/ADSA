import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score,
)
from core.state import AgentState
from utils.audit import log_action

CHART_DIR = 'outputs/charts'
os.makedirs(CHART_DIR, exist_ok=True)


def evaluation_agent(state: AgentState) -> AgentState:
    """Evaluate the best model on a holdout set and generate evaluation charts."""
    df         = state.get('feature_dataset')
    best_model = state.get('best_model')
    target     = state.get('target_column', '')
    task       = state.get('task_type', 'classification')

    if df is None or best_model is None:
        return state

    X = df.drop(columns=[target])
    y = df[target]
    saved = list(state.get('charts_saved', []))

    # 80/20 holdout split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )

    try:
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        eval_metrics = {}
        cm = None

        if task == 'classification':
            eval_metrics['accuracy']  = round(float(accuracy_score(y_test, y_pred)), 4)
            eval_metrics['precision'] = round(float(precision_score(
                y_test, y_pred, average='weighted', zero_division=0,
            )), 4)
            eval_metrics['recall']    = round(float(recall_score(
                y_test, y_pred, average='weighted', zero_division=0,
            )), 4)
            eval_metrics['f1_score']  = round(float(f1_score(
                y_test, y_pred, average='weighted', zero_division=0,
            )), 4)

            # Confusion matrix chart
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(6, 5))
            fig.patch.set_facecolor('#0A0F1E')
            ax.set_facecolor('#0D1526')
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                annot_kws={'color': 'white'},
            )
            ax.set_title('Confusion Matrix', color='white', fontsize=14)
            ax.set_xlabel('Predicted', color='#94a3b8')
            ax.set_ylabel('Actual', color='#94a3b8')
            ax.tick_params(colors='#94a3b8')
            path = f'{CHART_DIR}/confusion_matrix.png'
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            saved.append(path)
        else:
            eval_metrics['mae']  = round(float(mean_absolute_error(y_test, y_pred)), 4)
            eval_metrics['rmse'] = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)
            eval_metrics['r2']   = round(float(r2_score(y_test, y_pred)), 4)

            # Residual plot
            residuals = y_test - y_pred
            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor('#0A0F1E')
            ax.set_facecolor('#0D1526')
            ax.scatter(y_pred, residuals, alpha=0.5, color='#00D4FF', s=10)
            ax.axhline(y=0, color='#FF6B6B', linestyle='--', linewidth=1)
            ax.set_title('Residual Plot', color='white', fontsize=14)
            ax.set_xlabel('Predicted', color='#94a3b8')
            ax.set_ylabel('Residuals', color='#94a3b8')
            ax.tick_params(colors='#94a3b8')
            path = f'{CHART_DIR}/residual_plot.png'
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            saved.append(path)

        # Retrain on full data
        best_model.fit(X, y)

        log_action(state, 'EvaluationAgent', 'evaluated',
                   ', '.join(f'{k}={v}' for k, v in eval_metrics.items()))

        return {
            **state,
            'eval_metrics':     eval_metrics,
            'confusion_matrix': cm,
            'best_model':       best_model,
            'charts_saved':     saved,
        }
    except Exception as e:
        state.setdefault('errors', []).append({
            'agent': 'EvaluationAgent', 'error': str(e),
        })
        return state
