from typing import TypedDict, List, Optional, Any
import pandas as pd


class AgentState(TypedDict, total=False):
    # Input
    raw_dataset:        pd.DataFrame
    user_goal:          str
    file_path:          str

    # Planner
    task_type:          str       # classification | regression | clustering | timeseries
    task_confidence:    float
    target_column:      str
    workflow_steps:     List[str]

    # Profiling
    profile_report:     dict

    # Cleaning
    cleaned_dataset:    pd.DataFrame
    original_snapshot:  pd.DataFrame
    cleaning_log:       List[dict]

    # EDA
    eda_summary:        dict
    correlation_matrix: Optional[pd.DataFrame]

    # Feature Engineering
    feature_dataset:    pd.DataFrame
    selected_features:  List[str]
    feature_log:        List[dict]

    # Modeling
    trained_models:     dict
    best_model:         Any
    best_model_name:    str
    cv_results:         dict

    # Evaluation
    eval_metrics:       dict
    confusion_matrix:   Optional[Any]

    # Explainability
    shap_values:        Optional[Any]
    feature_importance: dict

    # Insights
    insights:           List[str]

    # System
    audit_trail:        List[dict]
    errors:             List[dict]
    charts_saved:       List[str]
