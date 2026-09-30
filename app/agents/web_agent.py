from app.tools.web_search import web_search


def web_agent(query: str) -> str:
    """
    Web agent handles current/external information.
    """

    result = web_search(query)

    return f"Web Search Result:\n{result}"