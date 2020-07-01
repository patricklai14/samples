from sklearn import model_selection
import pdb

from bball import predictor, preprocessor, stats

#To run: in top-level directory: python -m test.validate_models

def cross_validate(model_types):
    train_data = stats.read_data(2003, 2019)
    X, y = train_data.get_data()

    #cross validation parameters
    NUM_FOLDS = 5

    #metrics: ["neg_mean_absolute_error", "r2"]
    SCORING_METRIC = "r2"

    for model_type in model_types:
        X_transformed = preprocessor.transform_features(X)
        pdb.set_trace()

        #fit model
        model = predictor.predictor(model_type)
        scores = model_selection.cross_val_score(model, X_transformed, y, scoring=SCORING_METRIC, cv=NUM_FOLDS)
        print("{} for {}: {}".format(SCORING_METRIC, model_type, scores))

def main():
    model_types = ["linear", "log_linear"]
    cross_validate(model_types)

if __name__ == "__main__":
    main()