import subprocess
import numpy as np
import tempfile
import os
from sklearn.base import BaseEstimator
from sklearn.datasets import dump_svmlight_file
from utils.constants import constants, LIBSVM_DIR


class ProbabilityOfInclusionSVM(BaseEstimator):
    """
    The PI-SVM is a multi-class classifier that combines the separation capabilities of binary classifiers with an option to reject unknown classes.
    The basic idea is to fit a single-class probability model over the positive class scores from a discriminative binary classifier.
    The binary classification model discriminates the positive class from the known negative classes, while the single-class probability model adjusts the decision boundary so unknown classes are not frequently misclassified as belonging to the positive class.
    Since the decision boundary is defined by the training samples that are effectively extremes, the PI-SVM calibrates the posterior probabilities based on extreme value theory.
    The PI-SVM models the un-normalized posterior probability of inclusion by fitting an extreme value distribution (Weibull distribution) to decision scores from positive classes. Analogously to other open-set classifiers, it does not use negative samples for probabilistic modeling.

    Reference: L. Jain, W. Scheirer, T. Boult, "Multi-class Open Set Recognition Using Probability of Inclusion". ECCV, 2014.
    See also https://www.ic.unicamp.br/~rocha/pub/2015-wifs/wifs2015-tutorial-open-set.pdf#page=87
    """
    def __init__(self, C=1.0):
        """
        :param C: set the cost parameter C of C-SVC
        """
        self.model_file = None
        self.C = C

    def fit(self, X, y):
        """
        Train the PI-SVM model
        :param X: training samples of shape [num_samples, num_features]
        :param y: categorical labels of shape [num_samples], tested with 0, 1, 2
        """
        self.model_file = tempfile.NamedTemporaryFile(suffix=".svm", delete=False)
        with tempfile.NamedTemporaryFile(suffix=".svmlight") as f_train:
            dump_svmlight_file(X, y, f=f_train.name)

            # Train SVM
            # -s 10: 1-vs-rest binary SVMs for PI-SVM
            # -t 2: RBF kernel
            libsvm_train = os.path.join(constants[LIBSVM_DIR], "svm-train")
            subprocess.run([libsvm_train, "-s", "10", "-t", "2", "-c", str(self.C), f_train.name, self.model_file.name], check=True, stdout=subprocess.DEVNULL)

    def predict(self, X):
        """
        Predict using the PI-SVM model
        :param X: test samples of shape [num_samples, num_features]
        :return: (y_pred, y_prob), where y_pred is the predicted class label, and y_prob is the probability of inclusion (lower values indicate unknown classes and should be rejected)
        """
        with tempfile.NamedTemporaryFile(suffix=".svmlight") as f_test:
            # svmlight file must contain labels for computing the accuracy
            y_dummy = np.zeros(len(X), dtype=np.int)
            dump_svmlight_file(X, y_dummy, f=f_test.name)

            with tempfile.NamedTemporaryFile(suffix=".svmlight") as f_pred:
                libsvm_predict = os.path.join(constants[LIBSVM_DIR], "svm-predict")
                subprocess.run([libsvm_predict, "-P", "0", f_test.name, self.model_file.name, f_pred.name], check=True, stdout=subprocess.DEVNULL)

                y_pred_prob = np.genfromtxt(f_pred.name)

                y_pred = y_pred_prob[:, 0]
                y_prob = y_pred_prob[:, 1]

                return y_pred, y_prob
