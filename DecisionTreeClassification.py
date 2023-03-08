import numpy as np
import math

class DecisionTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    def predict(self, X):
        return np.array([self.predict_one(row, self.tree) for row in X])

    def predict_one(self, row, tree):
        if tree['is_leaf']:
            return tree['class']
        else:
            if row[tree['feature']] <= tree['threshold']:
                return self.predict_one(row, tree['left'])
            else:
                return self.predict_one(row, tree['right'])

    def build_tree(self, X, y, depth=0):
        if depth == self.max_depth or len(set(y)) == 1:
            return {
                'is_leaf': True,
                'class': max(set(y), key=list(y).count)
            }
        else:
            num_features = X.shape[1]
            best_feature, best_threshold = None, None
            best_gain = -np.inf
            for feature in range(num_features):
                thresholds = sorted(set(X[:, feature]))
                for threshold in thresholds:
                    gain = self.information_gain(X, y, feature, threshold)
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_threshold = threshold

            left_indices = X[:, best_feature] <= best_threshold
            right_indices = X[:, best_feature] > best_threshold

            return {
                'is_leaf': False,
                'feature': best_feature,
                'threshold': best_threshold,
                'left': self.build_tree(X[left_indices], y[left_indices], depth+1),
                'right': self.build_tree(X[right_indices], y[right_indices], depth+1)
            }

    def information_gain(self, X, y, feature, threshold):
        parent_entropy = self.entropy(y)
        left_indices = X[:, feature] <= threshold
        right_indices = X[:, feature] > threshold
        left_entropy = self.entropy(y[left_indices])
        right_entropy = self.entropy(y[right_indices])
        n_left, n_right, n = len(y[left_indices]), len(y[right_indices]), len(y)
        return parent_entropy - (n_left/n) * left_entropy - (n_right/n) * right_entropy

    def entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities))