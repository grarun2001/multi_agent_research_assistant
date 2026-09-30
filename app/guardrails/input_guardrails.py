from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from app.guardrails.pii_detector import detect_pii, redact_pii
from app.guardrails.injection_detector import detect_prompt_injection


@dataclass
class InputGuardrailResult:
    passed: bool
    sanitized_query: str
    original_query: str
    violations: List[str] = field(default_factory=list)
    pii_found: List[Dict[str, Any]] = field(default_factory=list)
    is_injection: bool = False
    risk_level: str = "LOW"
    error_message: Optional[str] = None


def validate_input(
    query: str,
    max_length: int = 2500,
    allow_pii_redaction: bool = True
) -> InputGuardrailResult:
    """
    Validates user input against:
    1. Basic structure (empty, length limits, character flood)
    2. Prompt injection & jailbreaks
    3. PII (redacted or flagged)
    """
    clean_query = query.strip() if query else ""
    violations: List[str] = []

    # 1. Structure / Validation Check
    if not clean_query:
        return InputGuardrailResult(
            passed=False,
            sanitized_query="",
            original_query=query,
            violations=["Empty query"],
            error_message="Query cannot be empty. Please enter a valid question."
        )

    if len(clean_query) > max_length:
        return InputGuardrailResult(
            passed=False,
            sanitized_query=clean_query[:max_length],
            original_query=query,
            violations=[f"Query exceeds max length of {max_length} characters"],
            error_message=f"Query is too long ({len(clean_query)} chars). Max allowed is {max_length}."
        )

    # Check for excessive character repetition (e.g. "aaaaa...")
    import re
    if re.search(r"(.)\1{30,}", clean_query):
        return InputGuardrailResult(
            passed=False,
            sanitized_query=clean_query,
            original_query=query,
            violations=["Repetitive character flood detected"],
            error_message="Input rejected due to invalid repetitive character sequences."
        )

    # 2. Prompt Injection Check
    inj_result = detect_prompt_injection(clean_query)
    if inj_result["is_injection"]:
        violations.append("Prompt Injection Detected")
        return InputGuardrailResult(
            passed=False,
            sanitized_query=clean_query,
            original_query=query,
            violations=violations,
            is_injection=True,
            risk_level=inj_result["risk_level"],
            error_message="Security Alert: Input contains potential prompt injection or system override instructions."
        )

    # 3. PII Detection & Redaction
    sanitized = clean_query
    pii_detected = detect_pii(clean_query)
    
    # Check for severe PII: Credit cards or SSN / National ID
    critical_pii = [p for p in pii_detected if p["type"] in ("CREDIT_CARD", "SSN", "API_KEY")]
    if critical_pii:
        violations.append(f"Sensitive credential/PII detected: {', '.join({p['type'] for p in critical_pii})}")
        if not allow_pii_redaction:
            return InputGuardrailResult(
                passed=False,
                sanitized_query=clean_query,
                original_query=query,
                violations=violations,
                pii_found=pii_detected,
                risk_level="HIGH",
                error_message="Security Alert: Submission contains sensitive PII/credentials (e.g. Credit Card, SSN, API Key) and cannot be processed."
            )

    if pii_detected and allow_pii_redaction:
        sanitized, _ = redact_pii(clean_query)
        violations.append(f"PII redacted ({len(pii_detected)} occurrences)")

    return InputGuardrailResult(
        passed=True,
        sanitized_query=sanitized,
        original_query=query,
        violations=violations,
        pii_found=pii_detected,
        is_injection=False,
        risk_level="LOW" if not pii_detected else "MEDIUM",
        error_message=None
    )
