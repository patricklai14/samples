This project applies a number of well-known supervised learning algorithms to two prediction problems from the UCI Machine Learning Repository: a handwritten digits recognition problem and a diabetic retinopathy prediction problem. We perform model selection, train, and evaluate/visualize the performance of the following machine learning models:
- Decision Trees
- Boosted Decision Trees
- Neural Networks
- K Nearest Neighbors
- Support Vector Machines

For each model, optimal parameters are chosen by conducting an exhaustive "grid search" and using the parameter values that yield the highest cross-validated test score.

To run the code for a particular model (on both data sets):
Decision Tree: python decision_tree.py
Neural Network: python neural_network.py
Boosting: python boosting.py
KNN: python knn.py
SVM: python svm.py

Packages used:
python 3.6.8
numpy 1.15.4
matplotlib 3.0.2
scikit-learn 0.20.2

References:
- parameter optimization code/visualization: https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html
- learning curve generation: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html