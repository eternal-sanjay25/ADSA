from core.state import AgentState
from utils.audit import log_action


def insight_agent(state: AgentState) -> AgentState:
    """Generate business insights from analysis results using LLM or heuristics."""
    feature_importance = state.get('feature_importance', {})
    eval_metrics       = state.get('eval_metrics', {})
    eda_summary        = state.get('eda_summary', {})
    task_type          = state.get('task_type', 'classification')
    target             = state.get('target_column', 'target')
    best_model_name    = state.get('best_model_name', 'Unknown')

    # Try LLM-powered insights first
    insights = _llm_insights(
        task_type, target, best_model_name,
        eval_metrics, feature_importance, eda_summary,
    )

    # Fallback to heuristic insights
    if not insights:
        insights = _heuristic_insights(
            task_type, target, best_model_name,
            eval_metrics, feature_importance, eda_summary,
        )

    log_action(state, 'InsightAgent', 'insights_generated',
               f'{len(insights)} insights')

    return {**state, 'insights': insights}


def _llm_insights(task_type, target, model_name, metrics, importance, eda):
    """Generate insights using LLM."""
    try:
        from utils.llm_client import get_llm
        llm = get_llm(temperature=0.4)

        top_features = list(importance.keys())[:5]
        top_corr = eda.get('top_correlations', [])[:3]
        corr_text = ', '.join(
            f'{p["col1"]}-{p["col2"]}({p["corr"]:.2f})'
            for p in top_corr
        )

        prompt = f"""You are a senior data scientist. Based on this analysis, provide 5-7 
actionable business insights in bullet point format. Be specific and quantitative.

Task: {task_type} predicting '{target}'
Best Model: {model_name}
Metrics: {metrics}
Top Features by SHAP: {top_features}
Top Correlations: {corr_text}
Target Distribution: {eda.get('target', {})}

Provide insights as a numbered list. Each insight should be one clear sentence.
Focus on business implications, not technical details."""

        result = llm.invoke(prompt).content.strip()

        # Parse numbered list
        insights = []
        for line in result.split('\n'):
            line = line.strip()
            if not line:
                continue
            # Remove numbering
            for prefix in ['1.', '2.', '3.', '4.', '5.', '6.', '7.',
                           '- ', '• ', '* ']:
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
                    break
            if len(line) > 10:
                insights.append(line)

        return insights[:7] if insights else None

    except Exception:
        return None


def _heuristic_insights(task_type, target, model_name, metrics, importance, eda):
    """Generate insights using heuristics when LLM is unavailable."""
    insights = []

    insights.append(
        f'Task identified as {task_type} for predicting "{target}".'
    )

    if metrics:
        if task_type == 'classification':
            acc = metrics.get('accuracy', 0)
            f1  = metrics.get('f1_score', 0)
            insights.append(
                f'{model_name} achieved {acc:.1%} accuracy and '
                f'{f1:.1%} F1-score on the holdout set.'
            )
            if acc > 0.9:
                insights.append(
                    'Model performance is excellent (>90% accuracy). '
                    'Consider deploying to production.'
                )
            elif acc < 0.7:
                insights.append(
                    'Model accuracy is below 70%. Consider gathering '
                    'more data or engineering additional features.'
                )
        else:
            r2 = metrics.get('r2', 0)
            insights.append(
                f'{model_name} explains {r2:.1%} of the variance '
                f'in {target} (R²).'
            )

    if importance:
        top_3 = list(importance.keys())[:3]
        insights.append(
            f'The most influential factors are: {", ".join(top_3)}.'
        )

    # Correlation insights
    top_corr = eda.get('top_correlations', [])
    if top_corr and abs(top_corr[0].get('corr', 0)) > 0.7:
        p = top_corr[0]
        insights.append(
            f'Strong correlation ({p["corr"]:.2f}) found between '
            f'{p["col1"]} and {p["col2"]}.'
        )

    # Target distribution insight
    target_info = eda.get('target', {})
    dist = target_info.get('distribution', {})
    if len(dist) == 2 and task_type == 'classification':
        values = list(dist.values())
        if min(values) < 0.2:
            insights.append(
                'The target variable is imbalanced. Consider using '
                'SMOTE or class weights for better minority class detection.'
            )

    return insights
