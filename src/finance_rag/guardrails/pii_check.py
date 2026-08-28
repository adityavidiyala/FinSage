import re

PII_PATTERNS = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "phone": r"\b(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "api_key": r"(api[_\s-]?key|token|secret)[:\s]+[A-Za-z0-9_\-]{10,}",
    "credit_card": r"\b\d{4}[\s-]\d{4}[\s-]\d{4}[\s-]\d{4}\b",
}


def detect_pii(message: str) -> list[str]:
    return [
        pii_type for pii_type, pattern in PII_PATTERNS.items()
        if re.search(pattern, message, re.IGNORECASE)
    ]