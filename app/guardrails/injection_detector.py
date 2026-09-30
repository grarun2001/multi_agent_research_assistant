import re
from typing import Dict, Any, List

# Heuristic patterns for prompt injection and jailbreak attempts
INJECTION_PATTERNS = [
    r"(ignore|disregard|forget|override|bypass)\s+(all\s+)?(previous|prior|above|earlier|initial|existing)\s+(instructions?|prompts?|rules?|directives?|guidelines?)",
    r"system\s*prompt",
    r"reveal\s+(your|the)\s+(system|initial|hidden)\s+instructions?",
    r"print\s+(your|the)\s+(system|initial)\s+prompt",
    r"what\s+are\s+your\s+instructions",
    r"(act\s+as|you\s+are\s+now)\s+(dan|an\s+unrestricted|a\s+hacker|jailbroken|godmode)",
    r"\b(dan\s+mode|jailbreak|jailbroken|godmode)\b",
    r"developer\s+mode\s+(enabled|on)?",
    r"you\s+are\s+now\s+free\s+from\s+all\s+rules",
    r"bypass\s+(safety|guardrails?|content\s+filters?)",
    r"override\s+system\s+(policy|instructions?)",
    r"from\s+now\s+on\s+you\s+must",
    r"do\s+anything\s+now",
    r"simulate\s+an\s+evil\s+assistant",
    r"sudo\s+mode",
    r"base64\s+decode\s+and\s+execute",
]


COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


def detect_prompt_injection(text: str) -> Dict[str, Any]:
    """
    Analyzes input text for prompt injection, jailbreaking,
    and system prompt extraction attempts.
    """
    matched_patterns: List[str] = []

    for pattern in COMPILED_PATTERNS:
        match = pattern.search(text)
        if match:
            matched_patterns.append(match.group())

    is_injection = len(matched_patterns) > 0

    return {
        "is_injection": is_injection,
        "matched_patterns": matched_patterns,
        "risk_level": "HIGH" if len(matched_patterns) >= 2 else ("MEDIUM" if is_injection else "LOW"),
        "reason": f"Prompt injection pattern detected: {', '.join(matched_patterns)}" if is_injection else "Safe"
    }
