from utils.audit import log_action


def chat_agent(state: dict, question: str) -> str:
    """RAG-based chat: retrieve relevant context from analysis, answer with LLM."""
    try:
        from utils.embedder import build_vector_store
        from utils.llm_client import get_llm

        # Build vector store from analysis results
        vector_store = build_vector_store(state)

        # Retrieve relevant documents
        docs = vector_store.similarity_search(question, k=5)
        context = '\n'.join([doc.page_content for doc in docs])

        # Generate answer
        llm = get_llm(temperature=0.3)
        prompt = f"""You are a data science assistant. Answer the user's question 
based on the analysis context below. Be specific, quantitative, and concise.

Context from analysis:
{context}

User question: {question}

Provide a clear, helpful answer in 2-4 sentences."""

        response = llm.invoke(prompt).content.strip()
        return response

    except ImportError:
        return _heuristic_chat(state, question)
    except Exception as e:
        return f'Sorry, I encountered an error: {str(e)}'


def _heuristic_chat(state: dict, question: str) -> str:
    """Simple keyword-based chat when LLM/embeddings are unavailable."""
    q = question.lower()

    if any(w in q for w in ['model', 'best', 'algorithm']):
        name    = state.get('best_model_name', 'Unknown')
        metrics = state.get('eval_metrics', {})
        return (
            f'The best model is {name} with metrics: '
            + ', '.join(f'{k}={v}' for k, v in metrics.items())
        )

    if any(w in q for w in ['feature', 'important', 'shap']):
        importance = state.get('feature_importance', {})
        top_5 = list(importance.keys())[:5]
        return (
            f'Top 5 most important features: {", ".join(top_5)}'
        )

    if any(w in q for w in ['insight', 'finding', 'conclusion']):
        insights = state.get('insights', [])
        return ' '.join(insights[:3]) if insights else 'No insights available.'

    if any(w in q for w in ['accuracy', 'f1', 'metric', 'performance']):
        metrics = state.get('eval_metrics', {})
        return ', '.join(f'{k}: {v}' for k, v in metrics.items())

    return (
        'I can answer questions about the model, features, '
        'metrics, and insights from this analysis.'
    )
