import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from core.state import AgentState
from utils.audit import log_action

CHART_DIR = 'outputs/charts'
os.makedirs(CHART_DIR, exist_ok=True)


def visualizer_agent(state: AgentState) -> AgentState:
    """Generate dark-themed charts: histograms, boxplots, heatmap, target dist."""
    df     = state['cleaned_dataset']
    target = state.get('target_column', '')
    task   = state.get('task_type', 'classification')
    saved  = list(state.get('charts_saved', []))

    for fn, args in [
        (_histograms,          [df]),
        (_boxplots,            [df]),
        (_correlation_heatmap, [df]),
        (_target_distribution, [df, target, task]),
    ]:
        try:
            result = fn(*args)
            if isinstance(result, list):
                saved += result
            elif isinstance(result, str):
                saved.append(result)
        except Exception as e:
            state.setdefault('errors', []).append({
                'agent': 'VisualizerAgent', 'error': str(e),
            })

    log_action(state, 'VisualizerAgent', 'charts_saved',
               f'{len(saved)} charts')
    return {**state, 'charts_saved': saved}


def _histograms(df):
    cols = df.select_dtypes(include=[np.number]).columns[:8]
    if not len(cols):
        return []
    n_cols = min(len(cols), 4)
    n_rows = (len(cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    for i, col in enumerate(cols):
        axes[i].hist(df[col].dropna(), bins=30,
                     color='#00D4FF', edgecolor='#0A0F1E')
        axes[i].set_title(col, fontsize=10, color='white')
        axes[i].set_facecolor('#0D1526')
        axes[i].tick_params(colors='#94a3b8')
    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)
    fig.patch.set_facecolor('#0A0F1E')
    plt.tight_layout()
    path = f'{CHART_DIR}/histograms.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return [path]


def _boxplots(df):
    cols = df.select_dtypes(include=[np.number]).columns[:8]
    if not len(cols):
        return []
    n_cols = min(len(cols), 4)
    n_rows = (len(cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    for i, col in enumerate(cols):
        bp = axes[i].boxplot(
            df[col].dropna(), patch_artist=True,
            boxprops=dict(facecolor='#00D4FF', color='#00D4FF'),
            whiskerprops=dict(color='#94a3b8'),
            capprops=dict(color='#94a3b8'),
            medianprops=dict(color='#FF6B6B'),
        )
        axes[i].set_title(col, fontsize=10, color='white')
        axes[i].set_facecolor('#0D1526')
        axes[i].tick_params(colors='#94a3b8')
    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)
    fig.patch.set_facecolor('#0A0F1E')
    plt.tight_layout()
    path = f'{CHART_DIR}/boxplots.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return [path]


def _correlation_heatmap(df):
    corr = df.select_dtypes(include=[np.number]).corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#0A0F1E')
    ax.set_facecolor('#0D1526')
    sns.heatmap(
        corr, annot=True, fmt='.2f', cmap='coolwarm',
        center=0, ax=ax,
        annot_kws={'size': 8, 'color': 'white'},
        cbar_kws={'shrink': 0.8},
    )
    ax.set_title('Correlation Heatmap', color='white', fontsize=14)
    ax.tick_params(colors='white')
    plt.setp(ax.get_xticklabels(), color='#94a3b8')
    plt.setp(ax.get_yticklabels(), color='#94a3b8')
    path = f'{CHART_DIR}/correlation_heatmap.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def _target_distribution(df, target, task):
    if not target or target not in df.columns:
        return []
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0A0F1E')
    ax.set_facecolor('#0D1526')
    if task == 'classification':
        counts = df[target].value_counts()
        ax.bar(counts.index.astype(str), counts.values,
               color='#00D4FF', edgecolor='#0A0F1E')
    else:
        ax.hist(df[target].dropna(), bins=40, color='#00D4FF',
                edgecolor='#0A0F1E')
    ax.set_title(f'Target: {target}', color='white', fontsize=14)
    ax.tick_params(colors='#94a3b8')
    ax.set_xlabel(target, color='#94a3b8')
    ax.set_ylabel('Count', color='#94a3b8')
    path = f'{CHART_DIR}/target_distribution.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return [path]
