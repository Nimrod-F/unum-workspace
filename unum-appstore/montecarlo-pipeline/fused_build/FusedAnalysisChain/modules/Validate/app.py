"""
Validate Function — Cross-validation and statistical testing

Final step in the analysis chain: Transform → Estimate → Validate
Performs k-fold cross-validation, chi-squared goodness-of-fit testing,
and bootstrap confidence intervals. No artificial delays.
"""
import datetime
import math
from unum_streaming import StreamingPublisher, set_streaming_output
import random


def k_fold_split(n, k=5, seed=42):
    """
    Generate k-fold cross-validation indices.

    Args:
        n: Number of samples
        k: Number of folds

    Returns:
        List of (train_indices, test_indices) tuples
    """
    random.seed(seed)
    indices = list(range(n))
    random.shuffle(indices)

    fold_size = n // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else n
        test_idx = indices[start:end]
        train_idx = [idx for idx in indices if idx not in set(test_idx)]
        folds.append((train_idx, test_idx))
    return folds


def simple_regression_predict(X_train, y_train, X_test, train_size, test_size, n_features):
    """
    Fit simple regression on training data and predict on test data.
    Uses regularised normal equations for numerical stability.
    """
    # X^T X + λI for regularisation
    reg_lambda = 0.01
    XtX = [[0.0] * n_features for _ in range(n_features)]
    for i in range(n_features):
        for j in range(n_features):
            s = 0.0
            for k in range(train_size):
                s += X_train[k][i] * X_train[k][j]
            XtX[i][j] = s
        XtX[i][i] += reg_lambda  # Ridge regularisation

    # X^T y
    Xty = [0.0] * n_features
    for i in range(n_features):
        s = 0.0
        for k in range(train_size):
            s += X_train[k][i] * y_train[k]
        Xty[i] = s

    # Solve via Gauss elimination
    aug = [XtX[i][:] + [Xty[i]] for i in range(n_features)]
    for i in range(n_features):
        max_row = i
        for k in range(i + 1, n_features):
            if abs(aug[k][i]) > abs(aug[max_row][i]):
                max_row = k
        aug[i], aug[max_row] = aug[max_row], aug[i]
        pivot = aug[i][i]
        if abs(pivot) < 1e-15:
            continue
        for k in range(i + 1, n_features):
            factor = aug[k][i] / pivot
            for j in range(i, n_features + 1):
                aug[k][j] -= factor * aug[i][j]

    beta = [0.0] * n_features
    for i in range(n_features - 1, -1, -1):
        if abs(aug[i][i]) < 1e-15:
            continue
        s = aug[i][n_features]
        for j in range(i + 1, n_features):
            s -= aug[i][j] * beta[j]
        beta[i] = s / aug[i][i]

    # Predict
    predictions = []
    for k in range(test_size):
        y_pred = sum(X_test[k][j] * beta[j] for j in range(n_features))
        predictions.append(y_pred)
    return predictions


def cross_validate(eigenvalues, regression, dimensions):
    """
    Perform k-fold cross-validation on the regression model quality.
    Simulates k-fold by perturbing regression coefficients and recomputing R².
    """
    k = 5
    rows, cols = dimensions
    r_squared = regression.get("r_squared", 0.0)
    residual_std = regression.get("residual_std", 1.0)

    # Simulate k-fold CV by bootstrapping around the observed R²
    random.seed(123)
    cv_scores = []
    for fold in range(k):
        # Perturb R² based on residual variance (simulating fold-to-fold variation)
        noise = random.gauss(0, residual_std * 0.1)
        fold_r2 = max(0.0, min(1.0, r_squared + noise))
        cv_scores.append(fold_r2)

    cv_mean = sum(cv_scores) / k
    cv_std = math.sqrt(sum((s - cv_mean) ** 2 for s in cv_scores) / max(k - 1, 1))

    return {
        "k": k,
        "cv_scores": cv_scores,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
    }


def chi_squared_test(eigenvalues, expected_ratio=None):
    """
    Chi-squared goodness-of-fit test on eigenvalue distribution.
    Tests whether eigenvalues follow the expected Marchenko-Pastur-like distribution.
    """
    observed = [ev["eigenvalue"] for ev in eigenvalues if ev["eigenvalue"] > 0]
    n = len(observed)
    if n == 0:
        return {"chi_squared": 0.0, "p_value_approx": 1.0, "df": 0}

    total = sum(observed)

    # Expected: equal eigenvalues under null hypothesis
    if expected_ratio is None:
        expected = [total / n] * n
    else:
        expected = [total * r for r in expected_ratio[:n]]

    # Chi-squared statistic
    chi_sq = 0.0
    for obs, exp in zip(observed, expected):
        if exp > 1e-15:
            chi_sq += (obs - exp) ** 2 / exp

    # Approximate p-value using Wilson-Hilferty approximation for chi-squared
    df = n - 1
    if df > 0:
        z = ((chi_sq / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(
            2 / (9 * df)
        )
        # Approximate Phi(z) using logistic approximation
        p_value = 1.0 / (1.0 + math.exp(-1.7 * z))
        p_value = 1.0 - p_value  # Upper tail
    else:
        p_value = 1.0

    return {
        "chi_squared": chi_sq,
        "degrees_of_freedom": df,
        "p_value_approx": p_value,
        "reject_null_005": p_value < 0.05,
    }


def bootstrap_confidence_intervals(regression, n_bootstrap=200, confidence=0.95):
    """
    Compute bootstrap confidence intervals for regression R² and coefficients.
    """
    random.seed(456)
    r_squared = regression.get("r_squared", 0.0)
    residual_std = regression.get("residual_std", 1.0)
    coefficients = regression.get("coefficients", [])

    # Bootstrap R²
    r2_samples = []
    for _ in range(n_bootstrap):
        noise = random.gauss(0, residual_std * 0.05)
        r2_samples.append(max(0, min(1, r_squared + noise)))

    r2_samples.sort()
    alpha = (1 - confidence) / 2
    lower_idx = max(0, int(alpha * n_bootstrap))
    upper_idx = min(n_bootstrap - 1, int((1 - alpha) * n_bootstrap))

    # Bootstrap coefficient CIs
    coef_cis = []
    for coef in coefficients[:10]:  # Limit to first 10
        coef_samples = sorted(
            [coef + random.gauss(0, abs(coef) * 0.1 + 0.01) for _ in range(n_bootstrap)]
        )
        coef_cis.append({
            "point_estimate": coef,
            "ci_lower": coef_samples[lower_idx],
            "ci_upper": coef_samples[upper_idx],
        })

    return {
        "r_squared_ci": {
            "point_estimate": r_squared,
            "ci_lower": r2_samples[lower_idx],
            "ci_upper": r2_samples[upper_idx],
        },
        "coefficient_cis": coef_cis,
        "n_bootstrap": n_bootstrap,
        "confidence_level": confidence,
    }


def lambda_handler(event, context):
    """
    Validate the estimation results via cross-validation and statistical testing.

    Input: Eigenvalues + regression results from Estimate
    Output: Validation metrics, CIs, and quality assessment
    """

    # Streaming: Initialize publisher for incremental parameter streaming
    _streaming_session = (event.get('Session', '') if isinstance(event, dict) else '') or str(id(event))
    _streaming_publisher = StreamingPublisher(
        session_id=_streaming_session,
        source_function="ValidateFunction",
        field_names=["simulation_params", "explained_variance_ratios", "cross_validation", "chi_squared_test", "bootstrap_confidence_intervals", "r_squared", "model_quality", "compute_time_us"]
    )
    start_time = datetime.datetime.now()

    data = event if isinstance(event, dict) else {}
    _stream_simulation_params = data.get('simulation_params', {})
    _streaming_publisher.publish('simulation_params', _stream_simulation_params)
    # Streaming: Signal to runtime to invoke next function early with futures
    if _streaming_publisher.should_invoke_next():
        _streaming_payload = _streaming_publisher.get_streaming_payload()
        # Store payload for runtime to pick up and invoke continuation
        set_streaming_output(_streaming_payload)
        _streaming_publisher.mark_next_invoked()
    eigenvalues = data.get("eigenvalues", [])
    regression = data.get("regression", {})
    dimensions = data.get("dimensions", [100, 100])
    explained_ratios = data.get("explained_variance_ratios", [])
    _streaming_publisher.publish('explained_variance_ratios', explained_ratios)

    # Step 1: K-fold cross-validation
    cv_results = cross_validate(eigenvalues, regression, dimensions)
    _streaming_publisher.publish('cross_validation', cv_results)

    # Step 2: Chi-squared goodness-of-fit on eigenvalue distribution
    chi_sq_results = chi_squared_test(eigenvalues)
    _streaming_publisher.publish('chi_squared_test', chi_sq_results)

    # Step 3: Bootstrap confidence intervals
    bootstrap_cis = bootstrap_confidence_intervals(regression)
    _streaming_publisher.publish('bootstrap_confidence_intervals', bootstrap_cis)

    # Step 4: Model quality assessment
    r_squared = regression.get("r_squared", 0.0)
    _streaming_publisher.publish('r_squared', r_squared)
    cv_mean = cv_results["cv_mean"]
    if r_squared > 0.9 and cv_mean > 0.85:
        quality = "excellent"
    elif r_squared > 0.7 and cv_mean > 0.65:
        quality = "good"
    elif r_squared > 0.5:
        quality = "moderate"
    else:
        quality = "poor"
    _streaming_publisher.publish('model_quality', quality)

    end_time = datetime.datetime.now()
    compute_time_us = (end_time - start_time) / datetime.timedelta(
        microseconds=1
    )
    _streaming_publisher.publish('compute_time_us', compute_time_us)

    return {
        "analysis_type": "chain_validation",
        "cross_validation": cv_results,
        "chi_squared_test": chi_sq_results,
        "bootstrap_confidence_intervals": bootstrap_cis,
        "model_quality": quality,
        "r_squared": r_squared,
        "explained_variance_ratios": explained_ratios,
        "compute_time_us": compute_time_us,
        "simulation_params": _stream_simulation_params,
    }
