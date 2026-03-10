"""
Transform Function — Matrix normalisation and covariance computation

Part of the analysis chain: Transform → Estimate → Validate
This chain is fusible (all scalar, sequential, same runtime).

Performs genuine matrix operations: column centering, scaling, and O(n³) matrix
multiplication for the Gram matrix (X^T X), which approximates the covariance
structure. No artificial delays.
"""
import datetime
from unum_streaming import StreamingPublisher, set_streaming_output
import math


def column_means(matrix, rows, cols):
    """Compute the mean of each column."""
    means = [0.0] * cols
    for row in matrix:
        for j, val in enumerate(row):
            means[j] += val
    return [m / rows for m in means]


def column_stds(matrix, means, rows, cols):
    """Compute the standard deviation of each column."""
    sq_diffs = [0.0] * cols
    for row in matrix:
        for j, val in enumerate(row):
            sq_diffs[j] += (val - means[j]) ** 2
    stds = [math.sqrt(s / max(rows - 1, 1)) for s in sq_diffs]
    # Avoid zero division
    return [s if s > 1e-10 else 1.0 for s in stds]


def center_and_scale(matrix, means, stds, rows, cols):
    """Center each column (subtract mean) and scale (divide by std)."""
    normalised = []
    for row in matrix:
        normalised.append(
            [(row[j] - means[j]) / stds[j] for j in range(cols)]
        )
    return normalised


def matrix_multiply_transpose(matrix, rows, cols):
    """
    Compute X^T × X (Gram matrix / covariance proxy).
    Result is cols×cols. This is the computationally heavy step: O(rows × cols²).
    """
    result = [[0.0] * cols for _ in range(cols)]
    for i in range(cols):
        for j in range(i, cols):  # Exploit symmetry
            dot = 0.0
            for k in range(rows):
                dot += matrix[k][i] * matrix[k][j]
            result[i][j] = dot
            result[j][i] = dot  # Symmetric
    return result


def compute_column_norms(normalised, rows, cols):
    """Compute L2 norm of each column after normalisation (should be ~sqrt(n-1))."""
    norms = [0.0] * cols
    for row in normalised:
        for j, val in enumerate(row):
            norms[j] += val * val
    return [math.sqrt(n) for n in norms]


def lambda_handler(event, context):
    """
    Normalise the input matrix and compute the Gram matrix.

    Input: Matrix data from DataGenerator
    Output: Normalised matrix + Gram matrix for downstream estimation
    """

    # Streaming: Initialize publisher for incremental parameter streaming
    _streaming_session = (event.get('Session', '') if isinstance(event, dict) else '') or str(id(event))
    _streaming_publisher = StreamingPublisher(
        session_id=_streaming_session,
        source_function="TransformFunction",
        field_names=["simulation_params", "col_means", "col_stds", "normalised_matrix", "col_norms", "covariance_matrix", "compute_time_us"]
    )
    start_time = datetime.datetime.now()

    # Extract matrix data
    data = event if isinstance(event, dict) else {}
    _stream_simulation_params = data.get('simulation_params', {})
    _streaming_publisher.publish('simulation_params', _stream_simulation_params)
    # Streaming: Signal to runtime to invoke next function early with futures
    if _streaming_publisher.should_invoke_next():
        _streaming_payload = _streaming_publisher.get_streaming_payload()
        # Store payload for runtime to pick up and invoke continuation
        set_streaming_output(_streaming_payload)
        _streaming_publisher.mark_next_invoked()
    matrix = data.get("matrix", [])
    dims = data.get("dimensions", [len(matrix), len(matrix[0]) if matrix else 0])
    rows, cols = dims[0], dims[1]

    # Step 1: Compute column means
    means = column_means(matrix, rows, cols)
    _stream_col_means = means[:20]
    _streaming_publisher.publish('col_means', _stream_col_means)

    # Step 2: Compute column standard deviations
    stds = column_stds(matrix, means, rows, cols)
    _stream_col_stds = stds[:20]
    _streaming_publisher.publish('col_stds', _stream_col_stds)

    # Step 3: Center and scale (Z-score normalisation)
    normalised = center_and_scale(matrix, means, stds, rows, cols)
    _streaming_publisher.publish('normalised_matrix', normalised)

    # Step 4: Compute Gram matrix X^T × X — O(n³) operation
    gram_matrix = matrix_multiply_transpose(normalised, rows, cols)

    # Step 5: Column norms as a sanity check
    col_norms = compute_column_norms(normalised, rows, cols)
    _stream_col_norms = col_norms[:20]
    _streaming_publisher.publish('col_norms', _stream_col_norms)

    # Scale Gram matrix to get sample covariance
    n = max(rows - 1, 1)
    covariance = [
        [gram_matrix[i][j] / n for j in range(cols)] for i in range(cols)
    ]
    _streaming_publisher.publish('covariance_matrix', covariance)

    end_time = datetime.datetime.now()
    compute_time_us = (end_time - start_time) / datetime.timedelta(
        microseconds=1
    )
    _streaming_publisher.publish('compute_time_us', compute_time_us)

    # Pass forward what downstream needs (normalised data + covariance)
    return {
        "normalised_matrix": normalised,
        "covariance_matrix": covariance,
        "dimensions": [rows, cols],
        "col_means": _stream_col_means,
        "col_stds": _stream_col_stds,
        "col_norms": _stream_col_norms,
        "compute_time_us": compute_time_us,
        "simulation_params": _stream_simulation_params,
    }
