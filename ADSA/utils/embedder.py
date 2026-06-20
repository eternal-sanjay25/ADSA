from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'


def get_embeddings():
    """Return a HuggingFace embedding model for vector search."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True},
    )


def build_vector_store(state: dict) -> FAISS:
    """Build a FAISS vector store from analysis results for RAG chat."""
    texts = []
    eda = state.get('eda_summary', {})

    # Numeric column stats
    for col, stats in eda.get('numeric', {}).items():
        texts.append(Document(
            page_content=(
                f'Column {col}: mean={stats.get("mean")}, '
                f'median={stats.get("median")}, std={stats.get("std")}'
            ),
            metadata={'source': 'eda_numeric'},
        ))

    # Categorical distributions
    for col, freq in eda.get('categorical', {}).items():
        top = list(freq.items())[:3]
        texts.append(Document(
            page_content=(
                f'Column {col} top values: '
                + ', '.join([f'{k}({v:.1%})' for k, v in top])
            ),
            metadata={'source': 'eda_categorical'},
        ))

    # Top correlations
    for pair in eda.get('top_correlations', []):
        texts.append(Document(
            page_content=(
                f'Correlation {pair["col1"]} vs {pair["col2"]}: {pair["corr"]}'
            ),
            metadata={'source': 'correlation'},
        ))

    # Feature importance
    for feat, val in list(state.get('feature_importance', {}).items())[:10]:
        texts.append(Document(
            page_content=f'Feature importance: {feat} = {round(val, 4)}',
            metadata={'source': 'importance'},
        ))

    # Insights
    for i, ins in enumerate(state.get('insights', []), 1):
        texts.append(Document(
            page_content=f'Insight {i}: {ins}',
            metadata={'source': 'insight'},
        ))

    # Model metrics
    metrics = state.get('eval_metrics', {})
    texts.append(Document(
        page_content=(
            'Model performance: '
            + ', '.join([f'{k}={v}' for k, v in metrics.items()])
        ),
        metadata={'source': 'metrics'},
    ))
    texts.append(Document(
        page_content=f'Best model: {state.get("best_model_name", "")}',
        metadata={'source': 'model'},
    ))

    return FAISS.from_documents(texts, get_embeddings())
