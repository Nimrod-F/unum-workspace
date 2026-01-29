"""
Random Forest Trainer - SLOWEST model (~8-12 seconds)

Simulates training a Random Forest which takes the longest time.
This creates the maximum stagger for demonstrating future-based benefits.
"""
import json
import time
import random


class DecisionStump:
    """Simple decision stump for RF."""
    def __init__(self):
        self.feature_idx = 0
        self.threshold = 0.0
        self.left_class = 0
        self.right_class = 1
    
    def fit(self, X, y, feature_subset):
        """Fit stump on a random feature."""
        if not feature_subset:
            feature_subset = list(range(len(X[0])))
        
        self.feature_idx = random.choice(feature_subset)
        values = [x[self.feature_idx] for x in X]
        self.threshold = sum(values) / len(values)
        
        # Determine majority class on each side
        left_y = [y[i] for i in range(len(X)) if X[i][self.feature_idx] <= self.threshold]
        right_y = [y[i] for i in range(len(X)) if X[i][self.feature_idx] > self.threshold]
        
        self.left_class = max(set(left_y), key=left_y.count) if left_y else 0
        self.right_class = max(set(right_y), key=right_y.count) if right_y else 1
    
    def predict(self, x):
        if x[self.feature_idx] <= self.threshold:
            return self.left_class
        return self.right_class


def train_random_forest(X, y, n_trees=10, max_features=None):
    """Train a simple random forest."""
    n_features = len(X[0])
    max_features = max_features or int(n_features ** 0.5)
    
    trees = []
    for _ in range(n_trees):
        # Bootstrap sample
        indices = [random.randint(0, len(X)-1) for _ in range(len(X))]
        X_boot = [X[i] for i in indices]
        y_boot = [y[i] for i in indices]
        
        # Random feature subset
        feature_subset = random.sample(range(n_features), min(max_features, n_features))
        
        # Train stump
        stump = DecisionStump()
        stump.fit(X_boot, y_boot, feature_subset)
        trees.append(stump)
    
    return trees


def lambda_handler(event, context):
    """
    Train a Random Forest classifier - SLOWEST model.
    """
    start_time = time.time()
    
    X = event.get('X', [])
    y = event.get('y', [])
    model_type = event.get('model_type', 'random_forest')
    delay_factor = event.get('delay_factor', 8.0)
    n_trees = event.get('n_trees', 10)
    
    # Simulate realistic training time variance (8-12 seconds)
    base_delay = delay_factor * (0.8 + random.random() * 0.4)
    time.sleep(base_delay)
    
    # Train the forest
    trees = train_random_forest(X, y, n_trees)
    
    # Calculate accuracy using majority voting
    correct = 0
    for i in range(len(X)):
        votes = [tree.predict(X[i]) for tree in trees]
        pred = max(set(votes), key=votes.count)
        if pred == y[i]:
            correct += 1
    accuracy = correct / len(X) if X else 0
    
    training_time_ms = int((time.time() - start_time) * 1000)
    
    return {
        "model_type": model_type,
        "model_name": "RandomForest",
        "accuracy": round(accuracy, 4),
        "n_samples": len(X),
        "n_features": len(X[0]) if X else 0,
        "training_time_ms": training_time_ms,
        "n_trees": n_trees,
        "max_features": int(len(X[0]) ** 0.5) if X else 0,
        "timestamp": time.time()
    }


if __name__ == '__main__':
    X = [[random.gauss(0, 1) for _ in range(10)] for _ in range(100)]
    y = [1 if sum(x[:5]) > 0 else 0 for x in X]
    
    result = lambda_handler({
        "X": X,
        "y": y,
        "model_type": "random_forest",
        "delay_factor": 8.0,
        "n_trees": 10
    }, None)
    print(json.dumps(result, indent=2))
