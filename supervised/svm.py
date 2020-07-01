import numpy as np
from matplotlib import pyplot as plt

from sklearn.svm import SVC
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit

from parse_data import read_diabetic_data, read_digits_data
from score_classifier import generate_learning_curves

import pdb

def plot_c_and_get_best(estimator, cv, x, y, data_set_name, c_range, y_min, y_max, kernel):
    scoring = {'Accuracy': make_scorer(accuracy_score), 'AUC' : 'roc_auc'}
    
    gs = GridSearchCV(estimator, param_grid={'C': c_range}, scoring=scoring, cv=cv, 
                      refit='AUC', return_train_score=True)
    gs.fit(x, y)
    results = gs.cv_results_

    best_auc_index = np.nonzero(results['rank_test_AUC'] == 1)[0][0]
    print("Training time: {}".format(results['mean_fit_time'][best_auc_index]))
    print("Testing time: {}".format(results['mean_score_time'][best_auc_index]))

    #plotting
    plt.figure(figsize=(13, 13))
    plt.title("SVM (" + kernel + ") Performance vs C - " + data_set_name, fontsize=16)

    plt.xlabel("C")
    plt.ylabel("Score")
    plt.xscale("log")

    ax = plt.gca()
    ax.set_xlim(c_range[0], c_range[-1])
    ax.set_xticks(c_range)
    ax.set_ylim(y_min, y_max)

    # Get the regular numpy array from the MaskedArray
    X_axis = np.array(results['param_C'].data, dtype=float)

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

def base_10_exp(x_list):
    return [10**x for x in x_list]

def main():
    LINEAR_KERNEL = 'linear'
    POLY_KERNEL = 'poly'

    x, y = read_diabetic_data("diabetic_data.txt")
    y = np.ravel(y)
    cv = StratifiedShuffleSplit(n_splits=30, test_size=0.25, random_state=0)

    linear_estimator = SVC(kernel = LINEAR_KERNEL, max_iter = 50000)
    best_estimator = plot_c_and_get_best(linear_estimator, cv, x, y, "Diabetic Retinopathy", 
                                         base_10_exp([-5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1]), 
                                         0.6, 1.01, LINEAR_KERNEL)

    title = "SVM - Linear Kernel (Diabetic Retinopathy)"
    generate_learning_curves(best_estimator, title, x, y, 0.6, 1.01)

    poly_estimator = SVC(kernel = POLY_KERNEL, max_iter = 50000)
    best_estimator = plot_c_and_get_best(poly_estimator, cv, x, y, "Diabetic Retinopathy", 
                                         base_10_exp([-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2]), 
                                         0.5, 1.01, POLY_KERNEL)

    title = "SVM - Polynomial Kernel (Diabetic Retinopathy)"
    generate_learning_curves(best_estimator, title, x, y, 0.5, 1.01)



    x, y = read_digits_data("digits_data.txt")
    y = np.ravel(y)
    cv = StratifiedShuffleSplit(n_splits=30, test_size=0.25, random_state=0)

    linear_estimator = SVC(kernel = LINEAR_KERNEL, max_iter = 50000)
    best_estimator = plot_c_and_get_best(linear_estimator, cv, x, y, "Digits", 
                                         base_10_exp([-5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1]), 
                                         0.9, 1.01, LINEAR_KERNEL)

    title = "SVM - Linear Kernel (Digits)"
    generate_learning_curves(best_estimator, title, x, y, 0.9, 1.01)

    poly_estimator = SVC(kernel = POLY_KERNEL, max_iter = 50000)
    best_estimator = plot_c_and_get_best(poly_estimator, cv, x, y, "Digits", 
                                         base_10_exp([-12, -11.5, -11, -10.5, -10, -9.5, -9, -8.5, -8, -7.5, -7, -6.5, -6, -5.5, -5]), 
                                         0.98, 1.01, POLY_KERNEL)

    title = "SVM - Polynomial Kernel (Digits)"
    generate_learning_curves(best_estimator, title, x, y, 0.9, 1.01)


if __name__ == "__main__":
    main()