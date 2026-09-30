from app.config import invoke_gemini
from app.agents.calculator_agent import calculator_agent
from app.agents.web_agent import web_agent
from app.agents.rag_agent import rag_agent


# ---------------------------------------------------------
# Supervisor
# ---------------------------------------------------------

def choose_agent(query: str) -> str:
    """
    Ask Gemini to decide which specialized agent
    should handle the user query.
    """

    prompt = f"""
You are a supervisor for a multi-agent research assistant.

Choose exactly ONE agent for the user query.

Available agents:

1. RAG
   - Company documents
   - Internal/private information
   - Company policies
   - Uploaded documents

2. WEB
   - Latest information
   - Current information
   - News
   - External knowledge

3. CALCULATOR
   - Mathematical calculations
   - Arithmetic
   - Percentages
   - Numerical expressions

User query:
{query}

Return ONLY one word:

RAG
WEB
CALCULATOR
"""

    decision = invoke_gemini(prompt).strip().upper()

    if "CALCULATOR" in decision:
        return "CALCULATOR"

    if "WEB" in decision:
        return "WEB"

    return "RAG"


# ---------------------------------------------------------
# Route Query
# ---------------------------------------------------------

def supervisor_route(query: str) -> str:
    """
    Route the user query to the selected specialized agent.
    """

    agent = choose_agent(query)

    print(f"\nSupervisor selected: {agent}")

    if agent == "CALCULATOR":
        return calculator_agent(query)

    if agent == "WEB":
        return web_agent(query)

    return rag_agent(query)


def choose_agents(query: str) -> list[str]:
    """
    Ask Gemini to identify all agents required
    for the user query.
    """

    prompt = f"""
You are a supervisor for a multi-agent research assistant.

Identify ALL agents required to answer the user query.

Available agents:

RAG:
- Company documents
- Internal information
- Company policies
- Uploaded documents

WEB:
- Latest information
- Current information
- News
- External information

CALCULATOR:
- Mathematical calculations
- Arithmetic
- Percentages

User query:
{query}

Return ONLY the required agent names separated by commas.

Valid names:
RAG
WEB
CALCULATOR

Example:
RAG, WEB

If only one agent is required:
RAG
"""

    decision = invoke_gemini(prompt).strip().upper()

    agents = []

    if "RAG" in decision:
        agents.append("RAG")

    if "WEB" in decision:
        agents.append("WEB")

    if "CALCULATOR" in decision:
        agents.append("CALCULATOR")

    if not agents:
        agents.append("RAG")

    return agents