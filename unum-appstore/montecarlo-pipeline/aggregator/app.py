"""
Aggregator Function — Fan-in point for the Monte Carlo Analysis Pipeline

Receives results from both branches:
  - Branch A (chain): Validate → validation metrics, CIs, model quality
  - Branch B (single): Simulate → Monte Carlo results

This is the fan-in point where future-based execution provides the most benefit:
the aggregator is invoked as soon as the first branch completes, and uses futures
for the pending branch's result.
"""
import datetime
import math


def merge_analysis_results(chain_result, simulation_result):
    """
    Merge results from the statistical chain and Monte Carlo simulation.

    Args:
        chain_result: Output from Validate (cross-validation, chi-sq, bootstrap CIs)
        simulation_result: Output from Simulate (pi, random walks, option pricing)

    Returns:
        Combined analysis with cross-method metrics
    """
    combined = {
        "chain_analysis": {},
        "simulation_analysis": {},
        "cross_method_metrics": {},
    }

    # Extract chain analysis summary
    if isinstance(chain_result, dict) and chain_result.get("analysis_type") == "chain_validation":
        combined["chain_analysis"] = {
            "model_quality": chain_result.get("model_quality", "unknown"),
            "r_squared": chain_result.get("r_squared", 0.0),
            "cv_mean": chain_result.get("cross_validation", {}).get("cv_mean", 0.0),
            "cv_std": chain_result.get("cross_validation", {}).get("cv_std", 0.0),
            "chi_sq_p_value": chain_result.get("chi_squared_test", {}).get("p_value_approx", 1.0),
            "r2_ci": chain_result.get("bootstrap_confidence_intervals", {}).get(
                "r_squared_ci", {}
            ),
            "explained_variance": chain_result.get("explained_variance_ratios", []),
            "compute_time_us": chain_result.get("compute_time_us", 0),
        }
    elif isinstance(chain_result, dict):
        combined["chain_analysis"] = {"raw": str(chain_result)[:500]}

    # Extract simulation summary
    if isinstance(simulation_result, dict) and simulation_result.get("analysis_type") == "monte_carlo_simulation":
        pi_est = simulation_result.get("pi_estimation", {})
        walks = simulation_result.get("random_walks", {})
        option = simulation_result.get("option_pricing", {})

        combined["simulation_analysis"] = {
            "pi_estimate": pi_est.get("pi_estimate", 0.0),
            "pi_error": pi_est.get("absolute_error", 0.0),
            "pi_convergence_rate": _compute_convergence_rate(pi_est.get("convergence", [])),
            "walk_mean_return": walks.get("aggregate", {}).get("mean_return_pct", 0.0),
            "walk_positive_pct": walks.get("aggregate", {}).get("positive_return_pct", 0.0),
            "option_price": option.get("option_price", 0.0),
            "option_std_error": option.get("standard_error", 0.0),
            "option_delta": option.get("delta_approx", 0.0),
            "compute_time_us": simulation_result.get("compute_time_us", 0),
        }
    elif isinstance(simulation_result, dict):
        combined["simulation_analysis"] = {"raw": str(simulation_result)[:500]}

    # Cross-method consistency metrics
    combined["cross_method_metrics"] = _compute_cross_metrics(
        combined["chain_analysis"], combined["simulation_analysis"]
    )

    return combined


def _compute_convergence_rate(convergence_data):
    """Compute the convergence rate from Monte Carlo checkpoints."""
    if len(convergence_data) < 2:
        return 0.0

    # Rate = how fast error decreases with sample size
    first = convergence_data[0]
    last = convergence_data[-1]
    if first.get("error", 0) > 0 and last.get("error", 0) > 0:
        n_ratio = last["samples"] / max(first["samples"], 1)
        error_ratio = first["error"] / max(last["error"], 1e-15)
        if n_ratio > 1 and error_ratio > 1:
            return math.log(error_ratio) / math.log(n_ratio)
    return 0.0


def _compute_cross_metrics(chain, simulation):
    """Compute metrics that compare both analysis branches."""
    metrics = {}

    # Model confidence score: combines R² from chain with MC convergence
    r_squared = chain.get("r_squared", 0.0)
    pi_error = simulation.get("pi_error", 1.0)
    mc_score = max(0, 1.0 - pi_error)  # Higher is better

    metrics["combined_confidence"] = (r_squared + mc_score) / 2.0
    metrics["model_r_squared"] = r_squared
    metrics["mc_accuracy_score"] = mc_score

    # Compute time comparison
    chain_time = chain.get("compute_time_us", 0)
    sim_time = simulation.get("compute_time_us", 0)
    total_time = chain_time + sim_time
    if total_time > 0:
        metrics["chain_time_fraction"] = chain_time / total_time
        metrics["simulation_time_fraction"] = sim_time / total_time
    else:
        metrics["chain_time_fraction"] = 0.5
        metrics["simulation_time_fraction"] = 0.5

    # Overall quality assessment
    quality_indicators = []
    if r_squared > 0.7:
        quality_indicators.append("good_regression_fit")
    if pi_error < 0.01:
        quality_indicators.append("accurate_mc_convergence")
    if chain.get("cv_std", 1.0) < 0.1:
        quality_indicators.append("stable_cross_validation")

    option_se = simulation.get("option_std_error", float("inf"))
    option_price = simulation.get("option_price", 0.0)
    if option_price > 0 and option_se / option_price < 0.05:
        quality_indicators.append("precise_option_pricing")

    metrics["quality_indicators"] = quality_indicators
    metrics["quality_score"] = len(quality_indicators) / 4.0

    return metrics


def lambda_handler(event, context):
    """
    Aggregate results from the analysis chain and simulation branches.

    Input: Array of [chain_result (Validate), simulation_result (Simulate)]
    Output: Merged analysis with cross-method metrics
    """
    start_time = datetime.datetime.now()

    # Handle fan-in input (list of branch results)
    results = event if isinstance(event, list) else [event]

    # Identify which result is from which branch
    chain_result = None
    simulation_result = None

    for result in results:
        if not isinstance(result, dict):
            continue
        analysis_type = result.get("analysis_type", "")
        if analysis_type == "chain_validation":
            chain_result = result
        elif analysis_type == "monte_carlo_simulation":
            simulation_result = result
        elif "cross_validation" in result or "model_quality" in result:
            chain_result = result
        elif "pi_estimation" in result or "random_walks" in result:
            simulation_result = result
        else:
            # Assign to whichever is still None
            if chain_result is None:
                chain_result = result
            elif simulation_result is None:
                simulation_result = result

    # Merge results
    merged = merge_analysis_results(
        chain_result or {}, simulation_result or {}
    )

    end_time = datetime.datetime.now()
    aggregation_time_us = (end_time - start_time) / datetime.timedelta(
        microseconds=1
    )
    merged["aggregation_time_us"] = aggregation_time_us
    merged["branches_received"] = len(results)

    return merged
