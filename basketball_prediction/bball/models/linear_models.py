import pdb

import numpy as np
from sklearn import linear_model

class linear:
    def __init__(self):
        self.model = linear_model.LinearRegression(fit_intercept=True)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self, deep=True):
        return {}

class log_linear:
    def __init__(self):
        self.model = linear_model.LinearRegression(fit_intercept=True)

    def fit(self, X, y):
        #filter out zeros
        nonzero_rows = (y != 0)
        y_nonzero = y[nonzero_rows]
        X_nonzero = X[nonzero_rows]

        #log-transform y
        y_log = np.log(y_nonzero.astype('float'))

        self.model.fit(X_nonzero, y_log)

    def predict(self, X):
        return np.exp(self.model.predict(X))

    def get_params(self, deep=True):
        return {}