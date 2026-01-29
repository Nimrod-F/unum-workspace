"""
Linear Regression Trainer - FAST model (~100-200ms)

This is intentionally the fastest model to train, completing
well before the other models.
"""
import json
import time
import random


def train_linear_regression(X, y):
    """Simple linear regression using gradient descent."""
    n_samples = len(X)
    n_features = len(X[0])
    
    # Initialize weights
    weights = [0.0] * n_features
    bias = 0.0
    lr = 0.01
    
    # Simple gradient descent (limited iterations for speed)
    for epoch in range(50):
        for i in range(n_samples):
            # Prediction
            pred = sum(w * x for w, x in zip(weights, X[i])) + bias
            error = pred - y[i]
            
            # Update weights
            for j in range(n_features):
                weights[j] -= lr * error * X[i][j] / n_samples
            bias -= lr * error / n_samples
    
    return weights, bias


def lambda_handler(event, context):
    """
    Train a linear regression model.
    """
    start_time = time.time()
    
    X = event.get('X', [])
    y = event.get('y', [])
    model_type = event.get('model_type', 'linear_regression')
    delay_factor = event.get('delay_factor', 0.1)
    
    # Simulate realistic training time variance
    base_delay = delay_factor * (0.8 + random.random() * 0.4)  # 80-120% of delay_factor
    time.sleep(base_delay)
    
    # Train the model
    weights, bias = train_linear_regression(X, y)
    
    # Calculate simple accuracy
    correct = 0
    for i in range(len(X)):
        pred = 1 if (sum(w * x for w, x in zip(weights, X[i])) + bias) > 0.5 else 0
        if pred == y[i]:
            correct += 1
    accuracy = correct / len(X) if X else 0
    
    training_time_ms = int((time.time() - start_time) * 1000)
    
    return {
        "model_type": model_type,
        "model_name": "LinearRegression",
        "accuracy": round(accuracy, 4),
        "n_samples": len(X),
        "n_features": len(X[0]) if X else 0,
        "training_time_ms": training_time_ms,
        "weights_sum": sum(weights),
        "bias": bias,
        "timestamp": time.time()
    }


if __name__ == '__main__':
    # Test locally
    X = [[random.gauss(0, 1) for _ in range(10)] for _ in range(100)]
    y = [1 if sum(x[:5]) > 0 else 0 for x in X]
    
    result = lambda_handler({
        "X": X,
        "y": y,
        "model_type": "linear_regression",
        "delay_factor": 0.1
    }, None)
    print(json.dumps(result, indent=2))
