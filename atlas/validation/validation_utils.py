
def clamp(value, minimum=0.0, maximum=1.0):
    return max(minimum, min(maximum, float(value)))


def letter_grade(score):
    score = float(score)

    if score >= 0.97:
        return "A+"
    if score >= 0.93:
        return "A"
    if score >= 0.90:
        return "A-"
    if score >= 0.87:
        return "B+"
    if score >= 0.83:
        return "B"
    if score >= 0.80:
        return "B-"
    if score >= 0.77:
        return "C+"
    if score >= 0.73:
        return "C"
    if score >= 0.70:
        return "C-"
    if score >= 0.60:
        return "D"
    return "F"


def mean_confidence(reports):
    values = [
        report.get("confidence")
        for report in reports
        if report.get("confidence") is not None
    ]

    if not values:
        return None

    return sum(values) / len(values)
