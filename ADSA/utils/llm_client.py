import os
from dotenv import load_dotenv

load_dotenv()


def get_llm(temperature=0.3):
    """Return an LLM instance with automatic fallback: Gemini → Groq → Ollama."""
    if os.getenv('GOOGLE_API_KEY'):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model='gemini-1.5-flash',
            temperature=temperature,
            google_api_key=os.getenv('GOOGLE_API_KEY'),
        )

    if os.getenv('GROQ_API_KEY'):
        from langchain_groq import ChatGroq
        return ChatGroq(
            model='llama-3.1-70b-versatile',
            temperature=temperature,
            groq_api_key=os.getenv('GROQ_API_KEY'),
        )

    try:
        from langchain_community.llms import Ollama
        return Ollama(model='llama3', temperature=temperature)
    except Exception:
        pass

    raise ValueError(
        'No LLM configured. Add GOOGLE_API_KEY to .env '
        '(free at aistudio.google.com)'
    )
