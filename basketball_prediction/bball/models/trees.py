import numpy as np
from sklearn import linear_model
from sklearn import tree

class decision_tree:
    def __init__(self, max_depth=None):
        self.model = tree.DecisionTreeClassifier(criterion="entropy", max_depth=max_depth)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self, deep=True):
        return {}