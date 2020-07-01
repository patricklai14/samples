import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier

from copy import deepcopy
import pdb
from time import time

from parse_data import read_diabetic_data, read_digits_data

AUC_SCORER = "roc_auc"
HOMOGENEITY = "homogeneity_score"
ADJUSTED_MUTUAL_INFO = "adjusted_mutual_info_score"

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, scorer=None, scorer_name = "Accuracy", 
                        train_sizes=np.linspace(.1, 1.0, 7)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title + '- ' + scorer_name)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel(scorer_name)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=scorer)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Testing score")

    ax = plt.gca()
    for i in range(len(train_sizes)):
        if (train_scores_mean[i] > test_scores_mean[i]):
            train_offset = 0.0005
            test_offset = -0.0005
        else:
            train_offset = -0.0005
            test_offset = +0.0005


        ax.annotate("%0.4f" % train_scores_mean[i],
                    (train_sizes[i], train_scores_mean[i] + train_offset))
        ax.annotate("%0.4f" % test_scores_mean[i],
                    (train_sizes[i], test_scores_mean[i] + test_offset))


    plt.legend(loc="best")
    plt.show()
    #plt.savefig(title + '-' + scorer_name + ".png")




def generate_learning_curves(estimator, title, X, y, y_min, y_max):
    cv = StratifiedShuffleSplit(n_splits=30, test_size=0.2, random_state=0)
    plot_learning_curve(estimator, title, X, y, ylim=(y_min, y_max), cv=cv, n_jobs=4, 
                        scorer=ADJUSTED_MUTUAL_INFO, scorer_name = "Adjusted Mutual Info")
    plot_learning_curve(estimator, title, X, y, ylim=(y_min, y_max), cv=cv, n_jobs=4,
                        scorer = HOMOGENEITY, scorer_name = "Homogeneity")
