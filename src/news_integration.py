
import os
from langchain_community.utilities import SerpAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI


def fetch_news(query, num=5):
    """Fetch news snippets from Google News using SerpAPI"""
    search = SerpAPIWrapper()
    results = search.run(f"{query} stock news")
    return results


def summarize_with_gemini(text):
    """Summarize market news using Gemini via LangChain with fallback models"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "⚠️ Gemini API key missing. Showing raw news only."

    # Preferred model order
    model_candidates = [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ]

    last_error = None
    for model in model_candidates:
        try:
            llm = ChatGoogleGenerativeAI(model=model, temperature=0.3, api_key=api_key)
            response = llm.invoke(f"Summarize the market impact of this news:\n\n{text}")
            return response.content
        except Exception as e:
            last_error = e
            continue

    return f"⚠️ Gemini summarization failed: {last_error}. Showing raw news only."
