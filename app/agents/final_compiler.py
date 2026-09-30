from app.config import invoke_gemini


# ---------------------------------------------------------
# Compile Results
# ---------------------------------------------------------

def final_compiler(query: str, results: list[str]) -> str:
    """
    Combine results from multiple agents
    into one clear final answer.
    """

    combined_results = "\n\n".join(results)

    prompt = f"""
You are the final answer compiler for a multi-agent research assistant.

The user asked:

{query}

The specialized agents returned these results:

{combined_results}

Create one clear and concise final answer for the user.

Rules:
- Use only the information provided by the agents.
- Do not invent information.
- Combine relevant information from all agents.
- Avoid repeating the same information.
- If an agent found no relevant information, ignore it.
"""

    return invoke_gemini(prompt)