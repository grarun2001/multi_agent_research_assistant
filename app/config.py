import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
DEFAULT_MODEL = os.getenv("MODEL_NAME", "gemini-3.7-flash")
MODEL_NAME = DEFAULT_MODEL

FALLBACK_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    DEFAULT_MODEL,
    "gemini-3.5-flash",
    "gemini-3.8-flash",
]



def invoke_gemini(prompt: str) -> str:
    """
    Invoke Gemini with automatic model fallback across models
    if quota or rate limit is encountered.
    Automatically logs token and latency metrics to active trace context if present.
    """
    import time
    from app.observability.tracer import get_current_trace

    last_err = None
    start_time = time.time()

    for model in FALLBACK_MODELS:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model,
                google_api_key=GOOGLE_API_KEY,
                temperature=0,
                max_retries=1
            )
            response = llm.invoke(prompt)
            content = response.content
            if isinstance(content, list):
                res_str = " ".join(
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict)
                ).strip()
            else:
                res_str = str(content).strip()

            duration_ms = (time.time() - start_time) * 1000.0

            # Record into trace context if active
            trace = get_current_trace()
            if trace:
                meta = getattr(response, "response_metadata", {})
                usage = meta.get("usage_metadata") or meta.get("token_usage")
                trace.record_llm_call(
                    prompt=prompt,
                    response=res_str,
                    model=model,
                    duration_ms=duration_ms,
                    usage_metadata=usage
                )

            return res_str
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"All Gemini models exhausted. Last error: {last_err}")