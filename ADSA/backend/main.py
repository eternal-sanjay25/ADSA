import os
import sys
import json
import uuid
import asyncio
import traceback
from typing import Dict, Any

# Fix Windows console encoding for arrows (e.g., in Optuna output)
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel

# Add project root to path so agents/core/utils are importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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
from agents.chat        import chat_agent

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(title='ADSA Backend API', version='2.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# In-memory session storage
sessions: Dict[str, Dict[str, Any]] = {}

UPLOAD_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'uploads')
CHART_DIR  = os.path.join(PROJECT_ROOT, 'outputs', 'charts')
REPORT_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'reports')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Pipeline steps in order
PIPELINE_STEPS = [
    ('planner',        planner_agent),
    ('profiler',       profiling_agent),
    ('cleaner',        cleaning_agent),
    ('eda',            eda_agent),
    ('visualizer',     visualizer_agent),
    ('feature_eng',    feature_agent),
    ('modeling',       modeling_agent),
    ('hpo',            hpo_agent),
    ('evaluation',     evaluation_agent),
    ('explainability', explainability_agent),
    ('insights',       insight_agent),
    ('report',         report_agent),
]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    question: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post('/api/upload')
async def upload_dataset(
    file: UploadFile = File(...),
    user_goal: str = Form(...),
):
    """Upload a CSV/Excel file and create a session."""
    session_id = str(uuid.uuid4())

    # Save file
    ext = file.filename.split('.')[-1].lower() if file.filename else 'csv'
    file_path = os.path.join(UPLOAD_DIR, f'{session_id}.{ext}')
    content = await file.read()
    with open(file_path, 'wb') as f:
        f.write(content)

    # Read into DataFrame
    try:
        df = pd.read_csv(file_path) if ext == 'csv' else pd.read_excel(file_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Failed to read file: {e}')

    # Store session
    sessions[session_id] = {
        'state': {
            'raw_dataset':  df,
            'user_goal':    user_goal,
            'file_path':    file_path,
            'audit_trail':  [],
            'errors':       [],
            'charts_saved': [],
        },
        'status':   'ready',
        'columns':  df.columns.tolist(),
        'rows':     len(df),
    }

    return {
        'session_id': session_id,
        'columns':    df.columns.tolist(),
        'rows':       len(df),
        'dtypes':     df.dtypes.astype(str).to_dict(),
    }


@app.get('/api/run/{session_id}')
async def run_pipeline_sse(session_id: str):
    """Run the full pipeline with Server-Sent Events streaming."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail='Session not found')

    async def event_stream():
        state = sessions[session_id]['state']
        sessions[session_id]['status'] = 'running'

        for step_name, step_fn in PIPELINE_STEPS:
            # Send progress event
            yield f'data: {json.dumps({"step": step_name, "status": "running"})}\n\n'
            await asyncio.sleep(0.1)  # Let the event flush

            try:
                # Run the CPU-bound agent function in a separate thread so we don't block the async event loop
                state = await asyncio.to_thread(step_fn, state)
                sessions[session_id]['state'] = state

                # Build step result
                result = {'step': step_name, 'status': 'done'}

                # Add relevant data per step
                if step_name == 'planner':
                    result['task_type']       = state.get('task_type')
                    result['task_confidence'] = state.get('task_confidence')
                    result['target_column']   = state.get('target_column')
                elif step_name == 'profiler':
                    profile = state.get('profile_report', {})
                    result['rows']    = profile.get('rows')
                    result['columns'] = profile.get('columns')
                    result['duplicates'] = profile.get('duplicate_rows')
                elif step_name == 'cleaner':
                    result['actions'] = len(state.get('cleaning_log', []))
                elif step_name == 'eda':
                    result['num_numeric']     = len(state.get('eda_summary', {}).get('numeric', {}))
                    result['num_categorical'] = len(state.get('eda_summary', {}).get('categorical', {}))
                elif step_name == 'visualizer':
                    result['charts'] = len(state.get('charts_saved', []))
                elif step_name == 'feature_eng':
                    result['features'] = len(state.get('selected_features', []))
                elif step_name == 'modeling':
                    result['best_model'] = state.get('best_model_name')
                    result['cv_results'] = {
                        k: round(v, 4)
                        for k, v in state.get('cv_results', {}).items()
                    }
                elif step_name == 'hpo':
                    result['best_model'] = state.get('best_model_name')
                elif step_name == 'evaluation':
                    result['metrics'] = state.get('eval_metrics', {})
                elif step_name == 'explainability':
                    imp = state.get('feature_importance', {})
                    result['top_features'] = list(imp.keys())[:5]
                elif step_name == 'insights':
                    result['insights'] = state.get('insights', [])
                elif step_name == 'report':
                    result['report_ready'] = True

                yield f'data: {json.dumps(result)}\n\n'

            except Exception as e:
                error_detail = {
                    'step':   step_name,
                    'status': 'error',
                    'error':  str(e),
                    'trace':  traceback.format_exc(),
                }
                yield f'data: {json.dumps(error_detail)}\n\n'
                # Continue to next step despite errors
                state.setdefault('errors', []).append({
                    'agent': step_name, 'error': str(e),
                })
                sessions[session_id]['state'] = state

        # Final completion event
        sessions[session_id]['status'] = 'completed'
        final = {
            'step':        'complete',
            'status':      'completed',
            'metrics':     state.get('eval_metrics', {}),
            'insights':    state.get('insights', []),
            'best_model':  state.get('best_model_name', ''),
            'charts':      _get_chart_list(),
            'errors':      state.get('errors', []),
        }
        yield f'data: {json.dumps(final)}\n\n'

    return StreamingResponse(
        event_stream(),
        media_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',
        },
    )


@app.post('/api/chat/{session_id}')
async def chat_endpoint(session_id: str, req: ChatRequest):
    """RAG chat about analysis results."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail='Session not found')

    state = sessions[session_id]['state']
    answer = chat_agent(state, req.question)
    return {'answer': answer}


@app.get('/api/charts')
async def list_charts():
    """List all generated chart files."""
    return {'charts': _get_chart_list()}


@app.get('/api/charts/{filename}')
async def get_chart(filename: str):
    """Serve a chart image."""
    path = os.path.join(CHART_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail='Chart not found')
    return FileResponse(path, media_type='image/png')


@app.get('/api/report/{session_id}')
async def get_report(session_id: str):
    """Download the PDF report."""
    pdf_path = os.path.join(REPORT_DIR, 'ADSA_Report.pdf')
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail='Report not generated yet')
    return FileResponse(
        pdf_path,
        media_type='application/pdf',
        filename='ADSA_Report.pdf',
    )


@app.get('/api/status/{session_id}')
async def get_status(session_id: str):
    """Get current session status."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail='Session not found')
    session = sessions[session_id]
    state   = session['state']
    return {
        'status':     session['status'],
        'metrics':    state.get('eval_metrics', {}),
        'insights':   state.get('insights', []),
        'best_model': state.get('best_model_name', ''),
        'errors':     state.get('errors', []),
    }


def _get_chart_list():
    """Return list of chart filenames in the chart directory."""
    if not os.path.exists(CHART_DIR):
        return []
    return [
        f for f in os.listdir(CHART_DIR)
        if f.endswith('.png')
    ]


# ---------------------------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------------------------
frontend_dir = os.path.join(PROJECT_ROOT, 'frontend')
if os.path.isdir(frontend_dir):
    app.mount('/', StaticFiles(directory=frontend_dir, html=True), name='frontend')
