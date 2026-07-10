
from datetime import datetime


PROFILE_SCHEMA_VERSION = "1.0.0"


def blank_profile():
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "questions": [],
        "facts": {},
        "contexts": {},
        "interactions": {},
        "samples": {},
        "confidence": {},
        "evidence": {},
        "last_updated": datetime.utcnow().isoformat(),
    }


def confidence_from_sample(n, strong=100, medium=40, weak=15):
    if n >= strong:
        return "strong"
    if n >= medium:
        return "medium"
    if n >= weak:
        return "weak"
    return "low"


def safe_rate(numerator, denominator):
    if denominator in [0, None]:
        return None
    return numerator / denominator
