import re
from typing import List, Dict, Tuple, Any

# Common PII Regex Patterns
EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)
PHONE_PATTERN = re.compile(
    r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
)
SSN_PATTERN = re.compile(
    r"\b\d{3}-\d{2}-\d{4}\b"
)
CREDIT_CARD_PATTERN = re.compile(
    r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
)
API_KEY_PATTERN = re.compile(
    r"\b(?:AIza[0-9A-Za-z-_]{35}|sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{36}|xox[baprs]-[0-9a-zA-Z]{10,48})\b"
)
AADHAAR_PATTERN = re.compile(
    r"\b\d{4}\s\d{4}\s\d{4}\b"
)


def detect_pii(text: str) -> List[Dict[str, Any]]:
    """
    Scans text for Personally Identifiable Information (PII)
    including Emails, Phone Numbers, Credit Cards, SSN/IDs, and API Keys.
    """
    findings = []

    # Check Emails
    for match in EMAIL_PATTERN.finditer(text):
        findings.append({
            "type": "EMAIL",
            "value": match.group(),
            "start": match.start(),
            "end": match.end()
        })

    # Check Phone Numbers
    for match in PHONE_PATTERN.finditer(text):
        val = match.group()
        # Avoid matching simple numbers like "1234567" unless formatted as phone
        if sum(c.isdigit() for c in val) >= 10:
            findings.append({
                "type": "PHONE",
                "value": val,
                "start": match.start(),
                "end": match.end()
            })

    # Check SSN
    for match in SSN_PATTERN.finditer(text):
        findings.append({
            "type": "SSN",
            "value": match.group(),
            "start": match.start(),
            "end": match.end()
        })

    # Check Credit Cards
    for match in CREDIT_CARD_PATTERN.finditer(text):
        val = match.group()
        digits = re.sub(r"\D", "", val)
        if len(digits) == 16:
            findings.append({
                "type": "CREDIT_CARD",
                "value": val,
                "start": match.start(),
                "end": match.end()
            })

    # Check API Keys / Secrets
    for match in API_KEY_PATTERN.finditer(text):
        findings.append({
            "type": "API_KEY",
            "value": match.group(),
            "start": match.start(),
            "end": match.end()
        })

    # Check Aadhaar / 12-digit formatted IDs
    for match in AADHAAR_PATTERN.finditer(text):
        findings.append({
            "type": "NATIONAL_ID",
            "value": match.group(),
            "start": match.start(),
            "end": match.end()
        })

    return findings


def redact_pii(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Detects and masks/redacts PII occurrences in the text.
    Returns (redacted_text, findings).
    """
    findings = detect_pii(text)
    if not findings:
        return text, []

    # Sort descending by start index to replace cleanly
    sorted_findings = sorted(findings, key=lambda x: x["start"], reverse=True)
    redacted = list(text)

    for item in sorted_findings:
        mask = f"[REDACTED_{item['type']}]"
        redacted[item["start"]:item["end"]] = list(mask)

    return "".join(redacted), findings
