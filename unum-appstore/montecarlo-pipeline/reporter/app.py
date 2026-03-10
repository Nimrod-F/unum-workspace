"""
Reporter Function — Final summary generation for the Monte Carlo Pipeline

Terminal function that receives aggregated results and produces a comprehensive
report with convergence analysis, confidence assessments, and formatted output.
No artificial delays.
"""
import datetime
import math


def format_confidence_interval(ci_dict):
    """Format a confidence interval dict as a readable string."""
    if not isinstance(ci_dict, dict):
        return "N/A"
    lower = ci_dict.get("ci_lower", 0.0)
    upper = ci_dict.get("ci_upper", 0.0)
    point = ci_dict.get("point_estimate", 0.0)
    return f"{point:.4f} [{lower:.4f}, {upper:.4f}]"


def compute_convergence_diagnostics(simulation_analysis):
    """
    Compute convergence diagnostics for the Monte Carlo simulations.
    """
    diagnostics = {}

    # Pi estimation convergence
    pi_error = simulation_analysis.get("pi_error", float("inf"))
    diagnostics["pi_converged"] = pi_error < 0.01
    diagnostics["pi_precision_digits"] = max(
        0, -int(math.floor(math.log10(pi_error + 1e-15)))
    ) if pi_error > 0 else 0

    # Option pricing precision
    option_se = simulation_analysis.get("option_std_error", float("inf"))
    option_price = simulation_analysis.get("option_price", 0.0)
    if option_price > 0:
        diagnostics["option_relative_error"] = option_se / option_price
        diagnostics["option_converged"] = (option_se / option_price) < 0.05
    else:
        diagnostics["option_relative_error"] = float("inf")
        diagnostics["option_converged"] = False

    # Random walk analysis
    walk_positive = simulation_analysis.get("walk_positive_pct", 50.0)
    diagnostics["market_bias"] = (
        "bullish" if walk_positive > 55 else "bearish" if walk_positive < 45 else "neutral"
    )

    return diagnostics


def compute_overall_quality(chain_analysis, simulation_analysis, cross_metrics):
    """
    Compute an overall quality score for the entire pipeline run.
    """
    scores = []

    # Chain quality (0-1)
    r_squared = chain_analysis.get("r_squared", 0.0)
    scores.append(min(1.0, r_squared))

    # CV stability (0-1)
    cv_std = chain_analysis.get("cv_std", 1.0)
    scores.append(max(0, 1.0 - cv_std * 5))  # cv_std < 0.2 → good

    # MC convergence (0-1)
    pi_error = simulation_analysis.get("pi_error", 1.0)
    scores.append(max(0, 1.0 - pi_error * 100))  # error < 0.01 → good

    # Cross-method consistency (0-1)
    quality_score = cross_metrics.get("quality_score", 0.0)
    scores.append(quality_score)

    if scores:
        overall = sum(scores) / len(scores)
    else:
        overall = 0.0

    if overall > 0.85:
        grade = "A"
    elif overall > 0.7:
        grade = "B"
    elif overall > 0.5:
        grade = "C"
    elif overall > 0.3:
        grade = "D"
    else:
        grade = "F"

    return {
        "overall_score": overall,
        "grade": grade,
        "component_scores": {
            "regression_fit": scores[0] if len(scores) > 0 else 0,
            "cv_stability": scores[1] if len(scores) > 1 else 0,
            "mc_convergence": scores[2] if len(scores) > 2 else 0,
            "cross_method": scores[3] if len(scores) > 3 else 0,
        },
    }


def generate_report_text(chain_analysis, simulation_analysis, diagnostics, quality):
    """
    Generate a human-readable summary report.
    """
    lines = [
        "=" * 60,
        "MONTE CARLO ANALYSIS PIPELINE — SUMMARY REPORT",
        "=" * 60,
        "",
        "1. STATISTICAL ANALYSIS CHAIN",
        f"   Model Quality: {chain_analysis.get('model_quality', 'unknown').upper()}",
        f"   R-squared: {chain_analysis.get('r_squared', 0.0):.4f}",
        f"   Cross-Validation Mean R²: {chain_analysis.get('cv_mean', 0.0):.4f} "
        f"(±{chain_analysis.get('cv_std', 0.0):.4f})",
        f"   Chi-sq p-value: {chain_analysis.get('chi_sq_p_value', 0.0):.4f}",
        "",
        "2. MONTE CARLO SIMULATIONS",
        f"   Pi Estimate: {simulation_analysis.get('pi_estimate', 0.0):.6f} "
        f"(error: {simulation_analysis.get('pi_error', 0.0):.6f})",
        f"   Pi Precision: {diagnostics.get('pi_precision_digits', 0)} decimal digits",
        f"   Option Price: ${simulation_analysis.get('option_price', 0.0):.4f} "
        f"(SE: {simulation_analysis.get('option_std_error', 0.0):.4f})",
        f"   Market Bias: {diagnostics.get('market_bias', 'unknown')}",
        f"   Positive Walk Returns: {simulation_analysis.get('walk_positive_pct', 0.0):.1f}%",
        "",
        "3. OVERALL ASSESSMENT",
        f"   Grade: {quality['grade']}",
        f"   Score: {quality['overall_score']:.2f}/1.00",
        f"   Regression Fit: {quality['component_scores']['regression_fit']:.2f}",
        f"   CV Stability: {quality['component_scores']['cv_stability']:.2f}",
        f"   MC Convergence: {quality['component_scores']['mc_convergence']:.2f}",
        f"   Cross-Method: {quality['component_scores']['cross_method']:.2f}",
        "",
        "=" * 60,
    ]
    return "\n".join(lines)


def lambda_handler(event, context):
    """
    Generate the final report for the Monte Carlo Analysis Pipeline.

    Input: Aggregated results from Aggregator
    Output: Comprehensive report with quality assessment
    """
    start_time = datetime.datetime.now()

    data = event if isinstance(event, dict) else {}

    chain_analysis = data.get("chain_analysis", {})
    simulation_analysis = data.get("simulation_analysis", {})
    cross_metrics = data.get("cross_method_metrics", {})

    # Convergence diagnostics
    diagnostics = compute_convergence_diagnostics(simulation_analysis)

    # Overall quality assessment
    quality = compute_overall_quality(
        chain_analysis, simulation_analysis, cross_metrics
    )

    # Generate readable report
    report_text = generate_report_text(
        chain_analysis, simulation_analysis, diagnostics, quality
    )

    end_time = datetime.datetime.now()
    report_time_us = (end_time - start_time) / datetime.timedelta(
        microseconds=1
    )

    return {
        "report": report_text,
        "quality": quality,
        "diagnostics": diagnostics,
        "chain_summary": chain_analysis,
        "simulation_summary": simulation_analysis,
        "cross_method_metrics": cross_metrics,
        "report_time_us": report_time_us,
        "pipeline": "montecarlo-pipeline",
        "status": "completed",
    }
