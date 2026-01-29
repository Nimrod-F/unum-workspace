"""
Data Generator - Generates synthetic dataset for ML training.

Creates a classification dataset that will be used by all parallel
model training functions.
"""
import json
import time
import random


def lambda_handler(event, context):
    """
    Generate synthetic classification dataset.
    
    Expected event:
    {
        "n_samples": 500,      # Number of samples
        "n_features": 20,      # Number of features
        "n_classes": 2         # Binary or multi-class
    }
    """
    n_samples = event.get('n_samples', 500)
    n_features = event.get('n_features', 20)
    n_classes = event.get('n_classes', 2)
    
    start_time = time.time()
    
    # Generate synthetic data
    random.seed(42)  # For reproducibility
    
    X = []
    y = []
    for i in range(n_samples):
        # Generate features
        features = [random.gauss(0, 1) for _ in range(n_features)]
        # Simple linear decision boundary + noise
        label = 1 if sum(features[:5]) > 0 else 0
        X.append(features)
        y.append(label)
    
    processing_time = int((time.time() - start_time) * 1000)
    
    # Return payloads for each model trainer
    # Each gets the same dataset but will train different models
    models = [
        {"model_type": "linear_regression", "delay_factor": 0.1},   # ~100ms - fastest
        {"model_type": "svm", "delay_factor": 2.0},                 # ~2-5s
        {"model_type": "random_forest", "delay_factor": 8.0},       # ~8-12s - slowest
        {"model_type": "gradient_boosting", "delay_factor": 5.0},   # ~5-8s
    ]
    
    payloads = []
    for model in models:
        payloads.append({
            "X": X,
            "y": y,
            "n_samples": n_samples,
            "n_features": n_features,
            "model_type": model["model_type"],
            "delay_factor": model["delay_factor"],
            "generator_time_ms": processing_time
        })
    
    return payloads


if __name__ == '__main__':
    result = lambda_handler({
        "n_samples": 100,
        "n_features": 10,
        "n_classes": 2
    }, None)
    print(f"Generated {len(result)} payloads for model training")
    for r in result:
        print(f"  - {r['model_type']}: {r['n_samples']} samples, {r['n_features']} features")
