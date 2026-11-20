

# ---------------------------------------------------------------- trust checks

def quality_problems(phys, report, min_minutes=5.0):
    """Reasons not to trust the numbers enough for a pro comparison (empty = OK)."""
    problems = []
    if report is not None:
        if report.get("usable_share", 0) < 0.3:
            problems.append(f"the pitch was only located in {report['usable_share']:.0%} of checked "
                            "frames (need 30%+)")
        if report.get("mapped_share", 0) < 0.5:
            problems.append(f"he was only placed on the pitch in {report['mapped_share']:.0%} of his "
                            "frames (need 50%+)")
    if phys is None:
        problems.append("no usable pitch positions")
        return problems
    if phys["minutes_tracked"] < min_minutes:
        problems.append(f"only {phys['minutes_tracked']:.1f} min tracked on the pitch "
                        f"(need {min_minutes:.0f}+ for per-90 numbers)")
    if phys["top_speed"] > 37:
        problems.append(f"top speed {phys['top_speed']:.0f} km/h is faster than humanly possible, "
                        "so positions are jumping")
    if phys["total_distance_p90"] > 14000:
        problems.append(f"{phys['total_distance_p90'] / 1000:.1f} km per 90 is beyond any pro, "
                        "so positions are jittering")
    return problems
