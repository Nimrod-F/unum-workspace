"""
Gradient Boosting Trainer - MEDIUM-SLOW model (~5-8 seconds)

Simulates training a Gradient Boosting classifier.
"""
import json
import time
import random


class WeakLearner:
    """Simple weak learner for gradient boosting."""
    def __init__(self):
        self.feature_idx = 0
        self.threshold = 0.0
        self.left_value = 0.0
        self.right_value = 0.0
    
    def fit(self, X, residuals):
        """Fit learner to minimize residuals."""
        n_features = len(X[0])
        self.feature_idx = random.randint(0, n_features - 1)
        
        values = [x[self.feature_idx] for x in X]
        self.threshold = sum(values) / len(values)
        
        # Calculate mean residual on each side
        left_res = [residuals[i] for i in range(len(X)) if X[i][self.feature_idx] <= self.threshold]
        right_res = [residuals[i] for i in range(len(X)) if X[i][self.feature_idx] > self.threshold]
        
        self.left_value = sum(left_res) / len(left_res) if left_res else 0
        self.right_value = sum(right_res) / len(right_res) if right_res else 0
    
    def predict(self, x):
        if x[self.feature_idx] <= self.threshold:
            return self.left_value
        return self.right_value


def train_gradient_boosting(X, y, n_estimators=10, learning_rate=0.1):
    """Train gradient boosting model."""
    n_samples = len(X)
    
    # Initialize predictions
    predictions = [0.5] * n_samples
    estimators = []
    
    for _ in range(n_estimators):
        # Compute residuals
        residuals = [y[i] - predictions[i] for i in range(n_samples)]
        
        # Fit weak learner to residuals
        learner = WeakLearner()
        learner.fit(X, residuals)
        estimators.append(learner)
        
        # Update predictions
        for i in range(n_samples):
            predictions[i] += learning_rate * learner.predict(X[i])
    
    return estimators, learning_rate


def lambda_handler(event, context):
    """
    Train a Gradient Boosting classifier.
    """
    start_time = time.time()
    
    X = event.get('X', [])
    y = event.get('y', [])
    model_type = event.get('model_type', 'gradient_boosting')
    delay_factor = event.get('delay_factor', 5.0)
    n_estimators = event.get('n_estimators', 10)
    
    # Simulate realistic training time (5-8 seconds)
    base_delay = delay_factor * (0.8 + random.random() * 0.4)
    time.sleep(base_delay)
    
    # Train the model
    estimators, learning_rate = train_gradient_boosting(X, y, n_estimators)
    
    # Calculate accuracy
    correct = 0
    for i in range(len(X)):
        score = 0.5
        for est in estimators:
            score += learning_rate * est.predict(X[i])
        pred = 1 if score > 0.5 else 0
        if pred == y[i]:
            correct += 1
    accuracy = correct / len(X) if X else 0
    
    training_time_ms = int((time.time() - start_time) * 1000)
    
    return {
        "model_type": model_type,
        "model_name": "GradientBoosting",
        "accuracy": round(accuracy, 4),
        "n_samples": len(X),
        "n_features": len(X[0]) if X else 0,
        "training_time_ms": training_time_ms,
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "timestamp": time.time()
    }


if __name__ == '__main__':
    X = [[random.gauss(0, 1) for _ in range(10)] for _ in range(100)]
    y = [1 if sum(x[:5]) > 0 else 0 for x in X]
    
    result = lambda_handler({
        "X": X,
        "y": y,
        "model_type": "gradient_boosting",
        "delay_factor": 5.0
    }, None)
    print(json.dumps(result, indent=2))
