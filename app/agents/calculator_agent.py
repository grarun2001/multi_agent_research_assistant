import re
from app.tools.calculator import calculate
from app.config import invoke_gemini


def calculator_agent(query: str) -> str:
    """
    Calculator agent extracts the mathematical expression
    from a natural-language user query and evaluates it.
    """
    cleaned_query = query.lower()

    # Common natural-language math phrases
    replacements = [
        ("multiplied by", "*"),
        ("times", "*"),
        ("into", "*"),
        ("divided by", "/"),
        ("divide by", "/"),
        ("over", "/"),
        ("percent of", "* 0.01 *"),
        ("% of", "* 0.01 *"),
        ("percentage of", "* 0.01 *"),
        ("plus", "+"),
        ("added to", "+"),
        ("minus", "-"),
        ("subtract", "-"),
        ("percent", "* 0.01"),
        ("%", "* 0.01"),
        (" x ", " * "),
    ]

    for phrase, symbol in replacements:
        cleaned_query = cleaned_query.replace(phrase, symbol)

    # Remove common calculator-related filler words
    keywords = [
        "calculate",
        "what is",
        "compute",
        "find",
        "solve",
        "please",
        "the",
        "value",
        "result",
        "equal to",
        "equals",
    ]

    for keyword in keywords:
        cleaned_query = cleaned_query.replace(keyword, "")

    cleaned_query = cleaned_query.replace("?", "").replace("=", "").strip()

    # Keep only mathematical expression characters
    expression = re.sub(r"[^0-9+\-*/(). ]", "", cleaned_query)
    # Collapse multiple spaces
    expression = re.sub(r"\s+", " ", expression).strip()

    result = None
    if expression:
        calc_res = calculate(expression)
        if not calc_res.startswith("Calculation error") and not calc_res.startswith("Invalid"):
            result = calc_res

    # Fallback to LLM if direct parsing failed
    if result is None:
        try:
            prompt = (
                f"Extract only the mathematical formula from this query as a single valid expression "
                f"using only digits and +, -, *, /, (, ), . operators. Do not include any words, letters, or equal sign.\n"
                f"Query: {query}\n"
                f"Expression:"
            )
            extracted = invoke_gemini(prompt).strip()
            extracted = re.sub(r"[^0-9+\-*/(). ]", "", extracted).strip()
            if extracted:
                calc_res = calculate(extracted)
                if not calc_res.startswith("Calculation error") and not calc_res.startswith("Invalid"):
                    result = calc_res
        except Exception:
            pass

    if result is None:
        return "Calculator could not evaluate a mathematical expression."

    return f"Calculator Result: {result}"