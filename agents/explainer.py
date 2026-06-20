import os
import shap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core.state import AgentState
from utils.audit import log_action

CHART_DIR = 'outputs/charts'
os.makedirs(CHART_DIR, exist_ok=True)


def explainability_agent(state: AgentState) -> AgentState:
    """Generate SHAP explanations: summary plot, bar plot, feature importance."""
    df              = state.get('feature_dataset')
    best_model      = state.get('best_model')
    best_model_name = state.get('best_model_name', '')
    target          = state.get('target_column', '')

    if df is None or best_model is None:
        return state

    X = df.drop(columns=[target])
    saved = list(state.get('charts_saved', []))

    # Sample for SHAP (slow on large datasets)
    X_sample = X.sample(n=min(500, len(X)), random_state=42)

    try:
        # Choose explainer
        if best_model_name in ('RandomForest', 'XGBoost'):
            explainer = shap.TreeExplainer(best_model)
        else:
            masker    = shap.maskers.Independent(data=X_sample)
            explainer = shap.LinearExplainer(best_model, masker)

        shap_values = explainer.shap_values(X_sample)

        # Handle multi-class / binary format differences
        if isinstance(shap_values, list):
            # Binary classification RF returns list of [class_0, class_1]
            shap_vals_for_plot = shap_values[1]
            shap_mean = np.abs(shap_values[1]).mean(axis=0)
        elif len(shap_values.shape) == 3:
            # XGBoost multi-output
            shap_vals_for_plot = shap_values[:, :, 1]
            shap_mean = np.abs(shap_values[:, :, 1]).mean(axis=0)
        else:
            shap_vals_for_plot = shap_values
            shap_mean = np.abs(shap_values).mean(axis=0)

        # Feature importance dict
        importance_dict = dict(zip(X.columns, shap_mean))
        importance_dict = {
            k: round(float(v), 6)
            for k, v in sorted(
                importance_dict.items(),
                key=lambda x: x[1], reverse=True,
            )
        }

        # SHAP Summary Plot
        plt.figure(figsize=(10, 8))
        plt.gcf().patch.set_facecolor('#0A0F1E')
        shap.summary_plot(shap_vals_for_plot, X_sample, show=False)
        plt.title('SHAP Feature Importance', color='white', fontsize=14)
        plt.tight_layout()
        path_summary = f'{CHART_DIR}/shap_summary.png'
        plt.savefig(path_summary, dpi=150, bbox_inches='tight',
                    facecolor='#0A0F1E')
        plt.close()
        saved.append(path_summary)

        # SHAP Bar Plot
        plt.figure(figsize=(10, 6))
        plt.gcf().patch.set_facecolor('#0A0F1E')
        top_n = min(15, len(importance_dict))
        top_feats = list(importance_dict.keys())[:top_n]
        top_vals  = [importance_dict[f] for f in top_feats]
        y_pos = np.arange(top_n)
        plt.barh(y_pos, top_vals[::-1], color='#00D4FF')
        plt.yticks(y_pos, top_feats[::-1], color='#94a3b8')
        plt.xlabel('Mean |SHAP value|', color='#94a3b8')
        plt.title('Top Feature Importance (SHAP)', color='white', fontsize=14)
        plt.gca().set_facecolor('#0D1526')
        plt.gca().tick_params(colors='#94a3b8')
        plt.tight_layout()
        path_bar = f'{CHART_DIR}/shap_bar.png'
        plt.savefig(path_bar, dpi=150, bbox_inches='tight',
                    facecolor='#0A0F1E')
        plt.close()
        saved.append(path_bar)

        log_action(state, 'ExplainAgent', 'shap_complete',
                   f'top features: {", ".join(top_feats[:5])}')

        return {
            **state,
            'shap_values':        shap_values,
            'feature_importance': importance_dict,
            'charts_saved':       saved,
        }

    except Exception as e:
        state.setdefault('errors', []).append({
            'agent': 'ExplainAgent', 'error': str(e),
        })
        return state
