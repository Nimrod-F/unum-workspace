"""
SVM Trainer - MEDIUM model (~2-5 seconds)

Simulates training an SVM classifier which takes moderate time.
"""
import json
import time
import random


def train_svm(X, y, C=1.0, max_iter=100):
    """Simple SVM using SGD (simplified for serverless)."""
    n_samples = len(X)
    n_features = len(X[0])
    
    # Convert labels to -1/+1
    y_svm = [1 if label == 1 else -1 for label in y]
    
    # Initialize weights
    weights = [0.0] * n_features
    bias = 0.0
    lr = 0.001
    
    for epoch in range(max_iter):
        for i in range(n_samples):
            # Hinge loss gradient
            margin = y_svm[i] * (sum(w * x for w, x in zip(weights, X[i])) + bias)
            
            if margin < 1:
                for j in range(n_features):
                    weights[j] += lr * (y_svm[i] * X[i][j] - 2 * (1/C) * weights[j])
                bias += lr * y_svm[i]
            else:
                for j in range(n_features):
                    weights[j] -= lr * 2 * (1/C) * weights[j]
    
    return weights, bias


def lambda_handler(event, context):
    """
    Train an SVM classifier.
    """
    start_time = time.time()
    
    X = event.get('X', [])
    y = event.get('y', [])
    model_type = event.get('model_type', 'svm')
    delay_factor = event.get('delay_factor', 2.0)
    
    # Simulate realistic training time variance (2-5 seconds)
    base_delay = delay_factor * (0.8 + random.random() * 0.4)
    time.sleep(base_delay)
    
    # Train the model
    weights, bias = train_svm(X, y)
    
    # Calculate accuracy
    correct = 0
    for i in range(len(X)):
        score = sum(w * x for w, x in zip(weights, X[i])) + bias
        pred = 1 if score > 0 else 0
        if pred == y[i]:
            correct += 1
    accuracy = correct / len(X) if X else 0
    
    training_time_ms = int((time.time() - start_time) * 1000)
    
    return {
        "model_type": model_type,
        "model_name": "SVM",
        "accuracy": round(accuracy, 4),
        "n_samples": len(X),
        "n_features": len(X[0]) if X else 0,
        "training_time_ms": training_time_ms,
        "support_vectors_estimate": int(len(X) * 0.3),  # Simplified
        "C": 1.0,
        "timestamp": time.time()
    }


if __name__ == '__main__':
    X = [[random.gauss(0, 1) for _ in range(10)] for _ in range(100)]
    y = [1 if sum(x[:5]) > 0 else 0 for x in X]
    
    result = lambda_handler({
        "X": X,
        "y": y,
        "model_type": "svm",
        "delay_factor": 2.0
    }, None)
    print(json.dumps(result, indent=2))
