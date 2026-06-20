from langgraph.graph import StateGraph, END
from core.state import AgentState
from agents.planner     import planner_agent
from agents.profiler    import profiling_agent
from agents.cleaner     import cleaning_agent
from agents.eda         import eda_agent
from agents.visualizer  import visualizer_agent
from agents.feature_eng import feature_agent
from agents.modeler     import modeling_agent
from agents.hpo         import hpo_agent
from agents.evaluator   import evaluation_agent
from agents.explainer   import explainability_agent
from agents.insight     import insight_agent
from agents.reporter    import report_agent


def build_graph():
    """Build and compile the ADSA pipeline graph."""
    graph = StateGraph(AgentState)

    graph.add_node('planner',        planner_agent)
    graph.add_node('profiler',       profiling_agent)
    graph.add_node('cleaner',        cleaning_agent)
    graph.add_node('eda',            eda_agent)
    graph.add_node('visualizer',     visualizer_agent)
    graph.add_node('feature_eng',    feature_agent)
    graph.add_node('modeling',       modeling_agent)
    graph.add_node('hpo',            hpo_agent)
    graph.add_node('evaluation',     evaluation_agent)
    graph.add_node('explainability', explainability_agent)
    graph.add_node('insights',       insight_agent)
    graph.add_node('report',         report_agent)

    graph.set_entry_point('planner')
    graph.add_edge('planner',        'profiler')
    graph.add_edge('profiler',       'cleaner')
    graph.add_edge('cleaner',        'eda')
    graph.add_edge('eda',            'visualizer')
    graph.add_edge('visualizer',     'feature_eng')
    graph.add_edge('feature_eng',    'modeling')
    graph.add_edge('modeling',       'hpo')
    graph.add_edge('hpo',            'evaluation')
    graph.add_edge('evaluation',     'explainability')
    graph.add_edge('explainability', 'insights')
    graph.add_edge('insights',       'report')
    graph.add_edge('report',         END)

    return graph.compile()


def run_pipeline(file_path: str, user_goal: str):
    """Convenience function: load data and run the full pipeline."""
    import pandas as pd

    app = build_graph()
    ext = file_path.split('.')[-1].lower()
    df  = pd.read_csv(file_path) if ext == 'csv' else pd.read_excel(file_path)

    initial_state = {
        'raw_dataset':  df,
        'user_goal':    user_goal,
        'file_path':    file_path,
        'audit_trail':  [],
        'errors':       [],
        'charts_saved': [],
    }
    return app.invoke(initial_state)
