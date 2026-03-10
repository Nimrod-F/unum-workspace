"""
Data Generator Function — Entry point for the Monte Carlo Analysis Pipeline

Generates random matrices and simulation parameters for downstream analysis.
Inspired by SeBS (Serverless Benchmark Suite) scientific computing benchmarks
and Malawski et al. "Serverless execution of scientific workflows" (FGCS, 2020).

No artificial delays — all computation is genuine random data generation.
"""
import datetime
import math
from unum_streaming import StreamingPublisher, set_streaming_output
import random


def generate_random_matrix(rows, cols, seed=None):
    """
    Generate a random matrix with values drawn from a standard normal distribution.
    Uses Box-Muller transform for Gaussian random numbers.

    Args:
        rows: Number of rows
        cols: Number of columns
        seed: Optional random seed for reproducibility

    Returns:
        List of lists representing the matrix
    """
    if seed is not None:
        random.seed(seed)

    matrix = []
    for i in range(rows):
        row = []
        for j in range(cols):
            # Box-Muller transform for standard normal
            u1 = random.random()
            u2 = random.random()
            z = math.sqrt(-2.0 * math.log(u1 + 1e-15)) * math.cos(2 * math.pi * u2)
            row.append(z)
        matrix.append(row)
    return matrix


def generate_simulation_parameters(n_simulations, n_walks, seed=None):
    """
    Generate parameters for Monte Carlo simulations.

    Args:
        n_simulations: Number of Monte Carlo samples
        n_walks: Number of random walks

    Returns:
        dict with simulation configuration and initial conditions
    """
    if seed is not None:
        random.seed(seed + 42)

    # Generate initial conditions for random walks
    initial_positions = [random.gauss(0, 1) for _ in range(n_walks)]

    # Generate drift and volatility parameters
    drift_params = [random.uniform(-0.05, 0.05) for _ in range(n_walks)]
    volatility_params = [random.uniform(0.1, 0.5) for _ in range(n_walks)]

    # Option pricing parameters (Black-Scholes Monte Carlo)
    spot_price = 100 + random.gauss(0, 10)
    strike_price = spot_price * random.uniform(0.9, 1.1)
    risk_free_rate = random.uniform(0.01, 0.05)
    volatility = random.uniform(0.15, 0.4)
    time_to_expiry = random.uniform(0.25, 2.0)

    return {
        "n_simulations": n_simulations,
        "n_walks": n_walks,
        "walk_steps": 1000,
        "initial_positions": initial_positions,
        "drift_params": drift_params,
        "volatility_params": volatility_params,
        "option_pricing": {
            "spot_price": spot_price,
            "strike_price": strike_price,
            "risk_free_rate": risk_free_rate,
            "volatility": volatility,
            "time_to_expiry": time_to_expiry,
        },
    }


def compute_basic_statistics(matrix):
    """Compute per-column statistics for validation downstream."""
    rows = len(matrix)
    cols = len(matrix[0])

    col_sums = [0.0] * cols
    col_sq_sums = [0.0] * cols
    col_mins = [float("inf")] * cols
    col_maxs = [float("-inf")] * cols

    for row in matrix:
        for j, val in enumerate(row):
            col_sums[j] += val
            col_sq_sums[j] += val * val
            col_mins[j] = min(col_mins[j], val)
            col_maxs[j] = max(col_maxs[j], val)

    col_means = [s / rows for s in col_sums]
    col_vars = [
        (col_sq_sums[j] / rows) - (col_means[j] ** 2) for j in range(cols)
    ]

    return {
        "col_means": col_means[:10],  # Truncate for payload size
        "col_variances": col_vars[:10],
        "col_ranges": [
            (col_mins[j], col_maxs[j]) for j in range(min(10, cols))
        ],
        "total_elements": rows * cols,
    }


def lambda_handler(event, context):
    """
    Generate random matrix data and simulation parameters.

    Input: Optional configuration (matrix_size, n_simulations, seed)
    Output: Matrix data + simulation parameters for downstream branches
    """

    # Streaming: Initialize publisher for incremental parameter streaming
    _streaming_session = (event.get('Session', '') if isinstance(event, dict) else '') or str(id(event))
    _streaming_publisher = StreamingPublisher(
        session_id=_streaming_session,
        source_function="DataGeneratorFunction",
        field_names=["dimensions", "config", "matrix", "simulation_params", "basic_stats", "generation_time_us"]
    )
    start_time = datetime.datetime.now()

    # Parse configuration from input
    config = event if isinstance(event, dict) else {}
    matrix_size = config.get("matrix_size", 150)
    _stream_dimensions = [matrix_size, matrix_size]
    _streaming_publisher.publish('dimensions', _stream_dimensions)
    # Streaming: Signal to runtime to invoke next function early with futures
    if _streaming_publisher.should_invoke_next():
        _streaming_payload = _streaming_publisher.get_streaming_payload()
        # Store payload for runtime to pick up and invoke continuation
        set_streaming_output(_streaming_payload)
        _streaming_publisher.mark_next_invoked()
    n_simulations = config.get("n_simulations", 200000)
    n_walks = config.get("n_walks", 100)
    seed = config.get("seed", None)
    _stream_config = {'matrix_size': matrix_size, 'n_simulations': n_simulations, 'n_walks': n_walks, 'seed': seed}
    _streaming_publisher.publish('config', _stream_config)

    # Generate random matrix (genuine computation)
    matrix = generate_random_matrix(matrix_size, matrix_size, seed=seed)
    _streaming_publisher.publish('matrix', matrix)

    # Generate simulation parameters
    sim_params = generate_simulation_parameters(
        n_simulations, n_walks, seed=seed
    )
    _streaming_publisher.publish('simulation_params', sim_params)

    # Compute basic statistics for the generated data
    basic_stats = compute_basic_statistics(matrix)
    _streaming_publisher.publish('basic_stats', basic_stats)

    end_time = datetime.datetime.now()
    generation_time_us = (end_time - start_time) / datetime.timedelta(
        microseconds=1
    )
    _streaming_publisher.publish('generation_time_us', generation_time_us)

    return {
        "matrix": matrix,
        "dimensions": _stream_dimensions,
        "simulation_params": sim_params,
        "basic_stats": basic_stats,
        "generation_time_us": generation_time_us,
        "config": _stream_config,
    }
