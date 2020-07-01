import numpy as np

from matplotlib import pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit

from parse_data import read_diabetic_data, read_digits_data
from score_classifier import generate_learning_curves

import pdb

def plot_hidden_layer_size_and_get_best(estimator, cv, x, y, data_set_name, x_min, x_max, y_min, y_max):
    scoring = {'Accuracy': make_scorer(accuracy_score), 'AUC': 'roc_auc'}
    
    gs = GridSearchCV(estimator, param_grid={'hidden_layer_sizes': [(n,) for n in range(x_min, x_max)]}, scoring=scoring, 
                      cv=cv, refit='AUC', return_train_score=True)
    gs.fit(x, y)
    results = gs.cv_results_

    #plotting
    plt.figure(figsize=(13, 13))
    plt.title("Neural Network Performance vs Number of Neurons in Hidden Layer - " + data_set_name, fontsize=16)

    plt.xlabel("Number of neurons")
    plt.ylabel("Score")

    ax = plt.gca()
    ax.set_xlim(x_min - 1, x_max + 1)
    ax.set_xticks(range(x_min - 1, x_max + 1))
    ax.set_ylim(y_min, y_max)

    # Get the regular numpy array from the MaskedArray
    X_axis = np.array([size[0] for size in results['param_hidden_layer_sizes'].data], dtype=float)

    for scorer, color in zip(sorted(scoring), ['g', 'k']):
        for sample, style in (('train', '--'), ('test', '-')):
            sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
            sample_score_std = results['std_%s_%s' % (sample, scorer)]
            ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                            sample_score_mean + sample_score_std,
                            alpha=0.1 if sample == 'test' else 0, color=color)
            ax.plot(X_axis, sample_score_mean, style, color=color,
                    alpha=1 if sample == 'test' else 0.7,
                    label="%s (%s)" % (scorer, sample))

        best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
        best_score = results['mean_test_%s' % scorer][best_index]

        # Plot a dotted vertical line at the best score for that scorer marked by x
        ax.plot([X_axis[best_index], ] * 2, [0, best_score],
                linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

        # Annotate the best score for that scorer
        ax.annotate("%0.4f" % best_score,
                    (X_axis[best_index], best_score + 0.0005))

    plt.legend(loc="best")
    plt.grid('off')
    plt.show()

    return gs.best_estimator_

def plot_learning_rate_and_get_best(estimator, cv, x, y, data_set_name, x_range, y_min, y_max):
    scoring = {'Accuracy': make_scorer(accuracy_score), 'AUC' : 'roc_auc'}
    
    gs = GridSearchCV(estimator, param_grid={'learning_rate_init': x_range}, scoring=scoring, cv=cv, 
                      refit='AUC', return_train_score=True)
    gs.fit(x, y)
    results = gs.cv_results_

    best_auc_index = np.nonzero(results['rank_test_AUC'] == 1)[0][0]
    print("Training time: {}".format(results['mean_fit_time'][best_auc_index]))
    print("Testing time: {}".format(results['mean_score_time'][best_auc_index]))

    #plotting
    plt.figure(figsize=(13, 13))
    plt.title("Neural Network Performance vs Learning Rate - " + data_set_name, fontsize=16)

    plt.xlabel("Learning Rate")
    plt.ylabel("Score")

    ax = plt.gca()
    ax.set_xlim(x_range[0], x_range[-1])
    ax.set_xticks(x_range)
    ax.set_ylim(y_min, y_max)

    # Get the regular numpy array from the MaskedArray
    X_axis = np.array(results['param_learning_rate_init'].data, dtype=float)

    for scorer, color in zip(sorted(scoring), ['g', 'k']):
        for sample, style in (('train', '--'), ('test', '-')):
            sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
            sample_score_std = results['std_%s_%s' % (sample, scorer)]
            ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                            sample_score_mean + sample_score_std,
                            alpha=0.1 if sample == 'test' else 0, color=color)
            ax.plot(X_axis, sample_score_mean, style, color=color,
                    alpha=1 if sample == 'test' else 0.7,
                    label="%s (%s)" % (scorer, sample))

        best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
        best_score = results['mean_test_%s' % scorer][best_index]

        # Plot a dotted vertical line at the best score for that scorer marked by x
        ax.plot([X_axis[best_index], ] * 2, [0, best_score],
                linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

        # Annotate the best score for that scorer
        ax.annotate("%0.4f" % best_score,
                    (X_axis[best_index], best_score + 0.0005))

    plt.legend(loc="best")
    plt.grid('off')
    #plt.savefig("DT_maxdepth_" + data_set_name + ".png")
    plt.show()

    return gs.best_estimator_


def main():
    x, y = read_diabetic_data("diabetic_data.txt")
    y = np.ravel(y)

    cv = StratifiedShuffleSplit(n_splits=30, test_size=0.25, random_state=0)
    base_estimator = MLPClassifier(activation='logistic', random_state=42)

    #set neurons in hidden layer
    layer_estimator = plot_hidden_layer_size_and_get_best(base_estimator, cv, x, y, "Diabetic Retinopathy", 
                                                          10, 30, 0.7, 1.0)
    #set learning rate
    learning_estimator = plot_learning_rate_and_get_best(layer_estimator, cv, x, y, "Diabetic Retinopathy", 
                                                         [0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004], 
                                                         0.7, 1.0)

    pdb.set_trace()
    title = "Neural Network (Diabetic Retinopathy)"
    generate_learning_curves(learning_estimator, title, x, y, 0.7, 1.01)




    x, y = read_digits_data("digits_data.txt")
    y = np.ravel(y)

    cv = StratifiedShuffleSplit(n_splits=30, test_size=0.25, random_state=0)
    base_estimator = MLPClassifier(activation='logistic', random_state=42)

    #set neurons in hidden layer
    layer_estimator = plot_hidden_layer_size_and_get_best(base_estimator, cv, x, y, "Digits", 
                                                          10, 30, 0.98, 1.01)
    #set learning rate
    learning_estimator = plot_learning_rate_and_get_best(layer_estimator, cv, x, y, "Digits", 
                                                         [0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004], 
                                                         0.98, 1.01)

    title = "Neural Network (Digits)"
    generate_learning_curves(learning_estimator, title, x, y, 0.9, 1.01)

if __name__ == "__main__":
    main()