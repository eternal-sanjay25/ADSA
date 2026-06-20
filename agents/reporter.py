import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch, cm
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from core.state import AgentState
from utils.audit import log_action

REPORT_DIR = 'outputs/reports'
CHART_DIR  = 'outputs/charts'
os.makedirs(REPORT_DIR, exist_ok=True)


def report_agent(state: AgentState) -> AgentState:
    """Generate a PDF report with all analysis results."""
    pdf_path = os.path.join(REPORT_DIR, 'ADSA_Report.pdf')

    try:
        doc = SimpleDocTemplate(
            pdf_path, pagesize=A4,
            topMargin=1 * cm, bottomMargin=1 * cm,
            leftMargin=1.5 * cm, rightMargin=1.5 * cm,
        )
        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle', parent=styles['Title'],
            fontSize=24, spaceAfter=20,
            textColor=HexColor('#1a1a2e'),
        )
        heading_style = ParagraphStyle(
            'CustomHeading', parent=styles['Heading2'],
            fontSize=16, spaceBefore=20, spaceAfter=10,
            textColor=HexColor('#16213e'),
        )
        body_style = ParagraphStyle(
            'CustomBody', parent=styles['Normal'],
            fontSize=11, spaceAfter=8, leading=16,
        )
        bullet_style = ParagraphStyle(
            'CustomBullet', parent=styles['Normal'],
            fontSize=11, spaceAfter=6, leftIndent=20, leading=16,
        )

        elements = []

        # Title page
        elements.append(Spacer(1, 2 * inch))
        elements.append(Paragraph('ADSA Analysis Report', title_style))
        elements.append(Spacer(1, 0.5 * inch))
        elements.append(Paragraph(
            f'Task: {state.get("task_type", "N/A").title()} — '
            f'Target: {state.get("target_column", "N/A")}',
            body_style,
        ))
        elements.append(Paragraph(
            f'Best Model: {state.get("best_model_name", "N/A")}',
            body_style,
        ))
        elements.append(PageBreak())

        # Executive Summary
        elements.append(Paragraph('Executive Summary', heading_style))
        for insight in state.get('insights', []):
            elements.append(Paragraph(f'• {insight}', bullet_style))
        elements.append(Spacer(1, 0.3 * inch))

        # Data Profile
        profile = state.get('profile_report', {})
        if profile:
            elements.append(Paragraph('Data Profile', heading_style))
            profile_data = [
                ['Metric', 'Value'],
                ['Rows', str(profile.get('rows', ''))],
                ['Columns', str(profile.get('columns', ''))],
                ['Duplicate Rows', str(profile.get('duplicate_rows', ''))],
                ['Memory (MB)', str(profile.get('memory_mb', ''))],
            ]
            table = Table(profile_data, colWidths=[3 * inch, 3 * inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#16213e')),
                ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ffffff')),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1),
                 [HexColor('#f0f0f0'), HexColor('#ffffff')]),
            ]))
            elements.append(table)
            elements.append(Spacer(1, 0.3 * inch))

        # Cleaning Log
        cleaning_log = state.get('cleaning_log', [])
        if cleaning_log:
            elements.append(Paragraph('Data Cleaning Summary', heading_style))
            for entry in cleaning_log[:15]:
                action = entry.get('action', '')
                col = entry.get('column', '')
                detail = entry.get('method', entry.get('count', ''))
                elements.append(Paragraph(
                    f'• {action}: {col} ({detail})', bullet_style,
                ))
            elements.append(Spacer(1, 0.3 * inch))

        # Model Results
        eval_metrics = state.get('eval_metrics', {})
        if eval_metrics:
            elements.append(Paragraph('Model Evaluation', heading_style))
            metrics_data = [['Metric', 'Value']]
            for k, v in eval_metrics.items():
                metrics_data.append([k.title(), f'{v:.4f}'])
            table = Table(metrics_data, colWidths=[3 * inch, 3 * inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#16213e')),
                ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ffffff')),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1),
                 [HexColor('#f0f0f0'), HexColor('#ffffff')]),
            ]))
            elements.append(table)
            elements.append(Spacer(1, 0.3 * inch))

        # CV Results
        cv_results = state.get('cv_results', {})
        if cv_results:
            elements.append(Paragraph('Cross-Validation Results', heading_style))
            cv_data = [['Model', 'CV Score']]
            for model_name, score in cv_results.items():
                cv_data.append([model_name, f'{score:.4f}'])
            table = Table(cv_data, colWidths=[3 * inch, 3 * inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#16213e')),
                ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ffffff')),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1),
                 [HexColor('#f0f0f0'), HexColor('#ffffff')]),
            ]))
            elements.append(table)
            elements.append(Spacer(1, 0.3 * inch))

        # Charts
        chart_files = [
            'correlation_heatmap.png', 'target_distribution.png',
            'histograms.png', 'boxplots.png',
            'confusion_matrix.png', 'residual_plot.png',
            'shap_summary.png', 'shap_bar.png',
        ]
        chart_added = False
        for chart_name in chart_files:
            chart_path = os.path.join(CHART_DIR, chart_name)
            if os.path.exists(chart_path):
                if not chart_added:
                    elements.append(PageBreak())
                    elements.append(Paragraph('Visualizations', heading_style))
                    chart_added = True
                try:
                    img = Image(chart_path, width=6 * inch, height=4 * inch)
                    img.hAlign = 'CENTER'
                    elements.append(img)
                    elements.append(Spacer(1, 0.3 * inch))
                except Exception:
                    pass

        doc.build(elements)
        log_action(state, 'ReportAgent', 'pdf_generated', pdf_path)

    except Exception as e:
        state.setdefault('errors', []).append({
            'agent': 'ReportAgent', 'error': str(e),
        })

    return state
