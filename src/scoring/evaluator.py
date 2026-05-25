import radon.complexity as cc
import radon.metrics as mi
import radon.raw as raw


# Maps the 1-5 integer score produced by analyze_script_complexity to a
# human-readable description used in report details.
_RATING_DESCRIPTIONS = {
    5: "Excellent — clean, simple, and highly maintainable.",
    4: "Good — well structured; minor improvements possible.",
    3: "Fair — functional but complex; consider refactoring.",
    2: "Poor — hard to read/maintain; significant refactoring needed.",
    1: "Critical — high risk of bugs; immediate refactoring required.",
}


def analyze_script_complexity(file_path) -> dict:
    """
    Analyze a Python script with radon and return a complexity score (1–5).

    Returns a dict with ``final_score`` and ``details`` on success,
    or ``{"error": "..."}`` on failure.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()
    except FileNotFoundError:
        return {"error": "File not found"}

    try:
        mi_score = mi.mi_visit(code, multi=True)
        cc_blocks = cc.cc_visit(code)
        raw_metrics = raw.analyze(code)
    except Exception as exc:
        return {"error": f"Error during radon analysis: {exc}"}

    avg_cc = (sum(b.complexity for b in cc_blocks) / len(cc_blocks)) if cc_blocks else 0
    max_cc = max((b.complexity for b in cc_blocks), default=0)

    # Derive score from Maintainability Index …
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

    # … then penalise for high cyclomatic complexity.
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
            "loc": raw_metrics.loc,
            "rating_description": _RATING_DESCRIPTIONS.get(final_score, "Unknown"),
        },
    }
