from typing import Dict, Any, Optional

# Gemini pricing per million tokens (USD)
# Typical rates: Gemini 1.5/2.0/3.x Flash: ~$0.10 / 1M input, ~$0.40 / 1M output
INPUT_COST_PER_MILLION = 0.10
OUTPUT_COST_PER_MILLION = 0.40


def estimate_tokens(text: str) -> int:
    """
    Estimates token count using character/word heuristics.
    Approx 4 characters per token in English.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def calculate_cost(input_tokens: int, output_tokens: int) -> float:
    """
    Calculates estimated cost in USD based on input and output token counts.
    """
    in_cost = (input_tokens / 1_000_000.0) * INPUT_COST_PER_MILLION
    out_cost = (output_tokens / 1_000_000.0) * OUTPUT_COST_PER_MILLION
    return round(in_cost + out_cost, 7)


def get_token_usage_summary(
    prompt: str,
    response: str,
    usage_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Returns prompt tokens, completion tokens, total tokens, and USD cost.
    Prefers actual usage_metadata if returned by LLM, otherwise estimates.
    """
    if usage_metadata and isinstance(usage_metadata, dict):
        input_tokens = usage_metadata.get("input_tokens") or usage_metadata.get("prompt_token_count")
        output_tokens = usage_metadata.get("output_tokens") or usage_metadata.get("candidates_token_count")
        if input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens
            cost = calculate_cost(input_tokens, output_tokens)
            return {
                "input_tokens": int(input_tokens),
                "output_tokens": int(output_tokens),
                "total_tokens": int(total_tokens),
                "cost_usd": cost,
                "is_estimated": False
            }

    # Fallback to estimate
    in_tok = estimate_tokens(prompt)
    out_tok = estimate_tokens(response)
    total_tok = in_tok + out_tok
    cost = calculate_cost(in_tok, out_tok)

    return {
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "total_tokens": total_tok,
        "cost_usd": cost,
        "is_estimated": True
    }
