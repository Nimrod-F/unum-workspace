"""
Simulate Function — Monte Carlo simulations

Parallel branch to the analysis chain (Transform → Estimate → Validate).
Performs genuine Monte Carlo computation: pi estimation, random walks,
and Black-Scholes option pricing. No artificial delays.

Inspired by Monte Carlo benchmarks in:
- Malawski et al., "Serverless execution of scientific workflows" (FGCS, 2020)
- SeBS (Serverless Benchmark Suite), Copik et al. (2021)
"""
import datetime
import math
from unum_streaming import StreamingPublisher, set_streaming_output
import random


def monte_carlo_pi(n_samples, seed=None):
    """
    Estimate π using Monte Carlo circle-in-square sampling.

    Throws n_samples random points in a unit square and counts how many
    fall inside the inscribed quarter-circle.

    Args:
        n_samples: Number of random samples

    Returns:
        dict with pi estimate, error, and convergence data
    """
    if seed is not None:
        random.seed(seed)

    inside = 0
    convergence_checkpoints = []

    for i in range(n_samples):
        x = random.random()
        y = random.random()
        if x * x + y * y <= 1.0:
            inside += 1

        # Record convergence at logarithmic intervals
        if (i + 1) in {100, 1000, 10000, 50000, 100000, 200000, 500000, n_samples}:
            pi_est = 4.0 * inside / (i + 1)
            convergence_checkpoints.append({
                "samples": i + 1,
                "estimate": pi_est,
                "error": abs(pi_est - math.pi),
            })

    pi_estimate = 4.0 * inside / n_samples

    return {
        "pi_estimate": pi_estimate,
        "pi_actual": math.pi,
        "absolute_error": abs(pi_estimate - math.pi),
        "relative_error": abs(pi_estimate - math.pi) / math.pi,
        "n_samples": n_samples,
        "convergence": convergence_checkpoints,
    }


def random_walks(n_walks, n_steps, initial_positions, drift_params, vol_params, seed=None):
    """
    Simulate geometric Brownian motion random walks.

    Each walk follows: S_{t+1} = S_t × exp((μ - σ²/2)Δt + σ√Δt × Z)
    where Z ~ N(0,1).

    Args:
        n_walks: Number of independent walks
        n_steps: Steps per walk
        initial_positions: Starting values
        drift_params: Per-walk drift (μ)
        vol_params: Per-walk volatility (σ)

    Returns:
        Summary statistics for each walk
    """
    if seed is not None:
        random.seed(seed)

    dt = 1.0 / n_steps
    sqrt_dt = math.sqrt(dt)
    walk_summaries = []

    for w in range(n_walks):
        pos = initial_positions[w] if w < len(initial_positions) else 1.0
        mu = drift_params[w] if w < len(drift_params) else 0.0
        sigma = vol_params[w] if w < len(vol_params) else 0.2

        positions = [pos]
        max_pos = pos
        min_pos = pos
        sum_pos = pos

        for step in range(n_steps):
            z = random.gauss(0, 1)
            pos = pos * math.exp((mu - 0.5 * sigma * sigma) * dt + sigma * sqrt_dt * z)
            max_pos = max(max_pos, pos)
            min_pos = min(min_pos, pos)
            sum_pos += pos
            positions.append(pos)

        # Compute walk statistics
        final = positions[-1]
        mean_pos = sum_pos / (n_steps + 1)
        # Compute standard deviation
        var_pos = sum((p - mean_pos) ** 2 for p in positions) / len(positions)

        walk_summaries.append({
            "initial": positions[0],
            "final": final,
            "max": max_pos,
            "min": min_pos,
            "mean": mean_pos,
            "std": math.sqrt(var_pos),
            "return_pct": (final - positions[0]) / abs(positions[0] + 1e-15) * 100,
        })

    # Aggregate across walks
    finals = [w["final"] for w in walk_summaries]
    returns = [w["return_pct"] for w in walk_summaries]

    return {
        "n_walks": n_walks,
        "n_steps": n_steps,
        "walk_summaries": walk_summaries[:10],  # First 10 for payload size
        "aggregate": {
            "mean_final": sum(finals) / len(finals),
            "std_final": math.sqrt(sum((f - sum(finals) / len(finals)) ** 2 for f in finals) / len(finals)),
            "mean_return_pct": sum(returns) / len(returns),
            "positive_return_pct": sum(1 for r in returns if r > 0) / len(returns) * 100,
        },
    }


def black_scholes_mc(spot, strike, rate, vol, T, n_simulations, seed=None):
    """
    Price a European call option via Monte Carlo simulation.

    S_T = S_0 × exp((r - σ²/2)T + σ√T × Z)
    Call payoff = max(S_T - K, 0)
    Price = e^(-rT) × E[payoff]

    Args:
        spot: Current stock price
        strike: Strike price
        rate: Risk-free rate
        vol: Volatility
        T: Time to expiry (years)
        n_simulations: Number of MC paths

    Returns:
        Option price, standard error, and Greeks approximations
    """
    if seed is not None:
        random.seed(seed)

    sqrt_T = math.sqrt(T)
    drift = (rate - 0.5 * vol * vol) * T

    payoffs = []
    for _ in range(n_simulations):
        z = random.gauss(0, 1)
        S_T = spot * math.exp(drift + vol * sqrt_T * z)
        payoff = max(S_T - strike, 0.0)
        payoffs.append(payoff)

    # Discounted expected payoff
    mean_payoff = sum(payoffs) / n_simulations
    price = math.exp(-rate * T) * mean_payoff

    # Standard error
    var_payoff = sum((p - mean_payoff) ** 2 for p in payoffs) / (n_simulations - 1)
    std_error = math.exp(-rate * T) * math.sqrt(var_payoff / n_simulations)

    # Approximate delta via finite difference
    eps = spot * 0.01
    payoffs_up = []
    random.seed(seed if seed else 789)
    for _ in range(min(n_simulations, 50000)):
        z = random.gauss(0, 1)
        S_T_up = (spot + eps) * math.exp(drift + vol * sqrt_T * z)
        payoffs_up.append(max(S_T_up - strike, 0.0))
    price_up = math.exp(-rate * T) * sum(payoffs_up) / len(payoffs_up)
    delta_approx = (price_up - price) / eps

    return {
        "option_price": price,
        "standard_error": std_error,
        "confidence_interval_95": [price - 1.96 * std_error, price + 1.96 * std_error],
        "delta_approx": delta_approx,
        "n_simulations": n_simulations,
        "parameters": {
            "spot": spot,
            "strike": strike,
            "rate": rate,
            "volatility": vol,
            "time_to_expiry": T,
        },
    }


def lambda_handler(event, context):
    """
    Run Monte Carlo simulations on the generated data.

    Input: Data + simulation parameters from DataGenerator
    Output: Pi estimate, random walk results, option pricing
    """

    # Streaming: Initialize publisher for incremental parameter streaming
    _streaming_session = (event.get('Session', '') if isinstance(event, dict) else '') or str(id(event))
    _streaming_publisher = StreamingPublisher(
        session_id=_streaming_session,
        source_function="SimulateFunction",
        field_names=["pi_estimation", "random_walks", "option_pricing", "compute_time_us"]
    )
    start_time = datetime.datetime.now()

    data = event if isinstance(event, dict) else {}
    sim_params = data.get("simulation_params", {})

    n_simulations = sim_params.get("n_simulations", 200000)
    n_walks = sim_params.get("n_walks", 100)
    n_steps = sim_params.get("walk_steps", 1000)
    initial_positions = sim_params.get("initial_positions", [1.0] * n_walks)
    drift_params = sim_params.get("drift_params", [0.0] * n_walks)
    vol_params = sim_params.get("volatility_params", [0.2] * n_walks)

    option_params = sim_params.get("option_pricing", {})

    # Step 1: Monte Carlo pi estimation
    pi_result = monte_carlo_pi(n_simulations, seed=42)
    _streaming_publisher.publish('pi_estimation', pi_result)
    # Streaming: Signal to runtime to invoke next function early with futures
    if _streaming_publisher.should_invoke_next():
        _streaming_payload = _streaming_publisher.get_streaming_payload()
        # Store payload for runtime to pick up and invoke continuation
        set_streaming_output(_streaming_payload)
        _streaming_publisher.mark_next_invoked()

    # Step 2: Random walk simulations
    walk_result = random_walks(
        n_walks, n_steps, initial_positions, drift_params, vol_params, seed=123
    )
    _streaming_publisher.publish('random_walks', walk_result)

    # Step 3: Black-Scholes Monte Carlo option pricing
    option_result = black_scholes_mc(
        spot=option_params.get("spot_price", 100),
        strike=option_params.get("strike_price", 105),
        rate=option_params.get("risk_free_rate", 0.03),
        vol=option_params.get("volatility", 0.25),
        T=option_params.get("time_to_expiry", 1.0),
        n_simulations=min(n_simulations, 500000),
        seed=456,
    )
    _streaming_publisher.publish('option_pricing', option_result)

    end_time = datetime.datetime.now()
    compute_time_us = (end_time - start_time) / datetime.timedelta(
        microseconds=1
    )
    _streaming_publisher.publish('compute_time_us', compute_time_us)

    return {
        "analysis_type": "monte_carlo_simulation",
        "pi_estimation": pi_result,
        "random_walks": walk_result,
        "option_pricing": option_result,
        "compute_time_us": compute_time_us,
    }
