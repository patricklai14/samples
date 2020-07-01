import pdb

from sklearn import linear_model

from bball.models import linear_models, trees

class predictor:
    #TODO: accept custom arguments for different model types
    def __init__(self, model_type):
        self.model_type = model_type

        if model_type == "linear":
            self.model = linear_models.linear()
        elif model_type == "log_linear":
            self.model = linear_models.log_linear()
        elif model_type == "decision_tree":
            self.model = trees.decision_tree()
        else:
            print("Unknown model type: {}".format(model_type))
            self.model = None

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self, deep=True):
        return {"model_type": self.model_type}