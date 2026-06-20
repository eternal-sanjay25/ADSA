"""Unit tests for ADSA v2 core agents."""

import sys
import os
import pytest
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_sample_df(n=200):
    """Create a small sample DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'CustomerID': [f'C{i}' for i in range(n)],
        'Age': np.random.normal(40, 10, n).clip(18, 80),
        'Monthly': np.random.normal(60, 20, n).clip(10, 150),
        'Tenure': np.random.randint(1, 60, n),
        'Category': np.random.choice(['A', 'B', 'C'], n),
        'Churn': np.random.choice([0, 1], n, p=[0.7, 0.3]),
    })


def _make_state(df=None):
    """Create a minimal pipeline state dict."""
    if df is None:
        df = _make_sample_df()
    return {
        'raw_dataset': df,
        'user_goal': 'predict churn',
        'file_path': 'test.csv',
        'audit_trail': [],
        'errors': [],
        'charts_saved': [],
    }


# ---- Planner Tests ----

class TestPlanner:
    def test_classification_detection(self):
        from agents.planner import planner_agent
        state = _make_state()
        result = planner_agent(state)
        assert result['task_type'] == 'classification'
        assert result['task_confidence'] > 0
        assert 'Churn' in result['target_column']

    def test_regression_detection(self):
        from agents.planner import planner_agent
        state = _make_state()
        state['user_goal'] = 'predict sales amount'
        result = planner_agent(state)
        assert result['task_type'] in ('regression', 'classification')

    def test_workflow_steps(self):
        from agents.planner import planner_agent
        state = _make_state()
        result = planner_agent(state)
        assert 'profiling' in result['workflow_steps']
        assert 'report' in result['workflow_steps']


# ---- Profiler Tests ----

class TestProfiler:
    def test_profiling(self):
        from agents.profiler import profiling_agent
        state = _make_state()
        result = profiling_agent(state)
        profile = result['profile_report']
        assert profile['rows'] == 200
        assert profile['columns'] == 6
        assert 'CustomerID' in profile['id_columns']

    def test_empty_dataset_raises(self):
        from agents.profiler import profiling_agent
        state = _make_state(pd.DataFrame())
        with pytest.raises(ValueError, match='EMPTY_DATASET'):
            profiling_agent(state)


# ---- Cleaner Tests ----

class TestCleaner:
    def test_cleaning(self):
        from agents.profiler import profiling_agent
        from agents.cleaner import cleaning_agent

        df = _make_sample_df()
        # Add some missing values
        df.loc[0:5, 'Age'] = np.nan
        state = _make_state(df)

        state = profiling_agent(state)
        result = cleaning_agent(state)

        assert 'cleaned_dataset' in result
        assert result['cleaned_dataset']['Age'].isnull().sum() == 0
        assert len(result['cleaning_log']) > 0

    def test_duplicate_removal(self):
        from agents.profiler import profiling_agent
        from agents.cleaner import cleaning_agent

        df = _make_sample_df()
        # Add duplicates
        df = pd.concat([df, df.iloc[:5]], ignore_index=True)
        state = _make_state(df)

        state = profiling_agent(state)
        result = cleaning_agent(state)

        assert len(result['cleaned_dataset']) < len(df)


# ---- EDA Tests ----

class TestEDA:
    def test_eda_summary(self):
        from agents.profiler import profiling_agent
        from agents.cleaner import cleaning_agent
        from agents.eda import eda_agent

        state = _make_state()
        state = profiling_agent(state)
        state = cleaning_agent(state)
        result = eda_agent(state)

        assert 'numeric' in result['eda_summary']
        assert 'categorical' in result['eda_summary']
        assert 'top_correlations' in result['eda_summary']
        assert result['correlation_matrix'] is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
