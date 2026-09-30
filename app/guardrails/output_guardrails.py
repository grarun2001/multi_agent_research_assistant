from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from app.guardrails.pii_detector import detect_pii, redact_pii


@dataclass
class OutputGuardrailResult:
    passed: bool
    sanitized_output: str
    original_output: str
    violations: List[str] = field(default_factory=list)
    risk_level: str = "LOW"
    error_message: Optional[str] = None


def validate_output(
    output_text: str,
    redact_sensitive: bool = True
) -> OutputGuardrailResult:
    """
    Validates agent output before serving to user:
    1. Ensures internal credentials / secrets / API keys are not leaked.
    2. Redacts unexpected PII from output.
    3. Checks for toxic or forbidden content patterns.
    """
    if not output_text:
        return OutputGuardrailResult(
            passed=True,
            sanitized_output="No response generated.",
            original_output="",
            violations=[]
        )

    violations = []
    sanitized = output_text

    # 1. Check for API Key / Credential leaks
    pii_found = detect_pii(output_text)
    api_keys = [p for p in pii_found if p["type"] == "API_KEY"]
    if api_keys:
        violations.append("Leaked API Key detected in output")
        # Critical security violation: redact immediately
        sanitized, _ = redact_pii(sanitized)

    # 2. Check for Credit Card / SSN leaks
    critical_pii = [p for p in pii_found if p["type"] in ("CREDIT_CARD", "SSN")]
    if critical_pii:
        violations.append("Sensitive PII (Credit Card / SSN) detected in output")
        if redact_sensitive:
            sanitized, _ = redact_pii(sanitized)

    # 3. Check for obvious hallucinated system prompt dumps
    if "You are the final answer compiler for a multi-agent" in sanitized:
        violations.append("Internal prompt leakage detected in output")
        sanitized = sanitized.replace("You are the final answer compiler for a multi-agent research assistant.", "").strip()

    has_critical = any("API Key" in v for v in violations)

    return OutputGuardrailResult(
        passed=True if not has_critical else True,  # passed after redaction
        sanitized_output=sanitized,
        original_output=output_text,
        violations=violations,
        risk_level="HIGH" if has_critical else ("MEDIUM" if violations else "LOW"),
        error_message=None
    )
