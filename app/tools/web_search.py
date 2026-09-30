from tavily import TavilyClient
from app.config import TAVILY_API_KEY


def web_search(query: str) -> str:
    """
    Search the web using Tavily.
    """

    if not TAVILY_API_KEY:
        return "TAVILY_API_KEY is not configured."

    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)

        response = client.search(
            query=query,
            max_results=5
        )

        results = response.get("results", [])

        if not results:
            return "No web results found."

        formatted_results = []

        for result in results:
            title = result.get("title", "")
            content = result.get("content", "")
            url = result.get("url", "")

            formatted_results.append(
                f"Title: {title}\n"
                f"Content: {content}\n"
                f"URL: {url}"
            )

        return "\n\n".join(formatted_results)
    except Exception as e:
        return f"Web search encountered an error: {str(e)}"