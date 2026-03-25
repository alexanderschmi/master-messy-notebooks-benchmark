import radon.complexity as cc
import radon.metrics as mi
import radon.raw as raw


def analyze_script_complexity(file_path):
    """
    Analyzes a Python script using radon metrics and returns a complexity score (1-5).
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()
    except FileNotFoundError:
        return {"error": "File not found"}

    try:
        mi_score = mi.mi_visit(code, multi=True)
        cc_blocks = cc.cc_visit(code)

        if cc_blocks:
            avg_cc = sum(block.complexity for block in cc_blocks) / len(cc_blocks)
            max_cc = max(block.complexity for block in cc_blocks)
        else:
            avg_cc = 0
            max_cc = 0

        raw_metrics = raw.analyze(code)
        loc = raw_metrics.loc
    except Exception as e:
        return {"error": f"Error during radon analysis: {e}"}

    if mi_score >= 80:
        score = 5
    elif mi_score >= 60:
        score = 4
    elif mi_score >= 40:
        score = 3
    elif mi_score >= 20:
        score = 2
    else:
        score = 1

    if max_cc > 40:
        score = 1
    elif max_cc > 30:
        score = min(score, 2)
    elif max_cc > 20:
        score = min(score, 3)
    elif max_cc > 10:
        score -= 1

    final_score = max(1, min(5, score))

    return {
        "final_score": final_score,
        "details": {
            "maintainability_index": round(mi_score, 2),
            "average_complexity": round(avg_cc, 2),
            "max_complexity": max_cc,
            "loc": loc,
            "rating_description": _get_rating_desc(final_score),
        },
    }


def _get_rating_desc(score):
    descriptions = {
        5: "Excellent - Clean, simple, and highly maintainable.",
        4: "Good - Well structured, minor improvements possible.",
        3: "Fair - Functional but complex; consider refactoring.",
        2: "Poor - Hard to read/maintain; significant refactoring needed.",
        1: "Critical - High risk of bugs; immediate refactoring required.",
    }
    return descriptions.get(score, "Unknown")
