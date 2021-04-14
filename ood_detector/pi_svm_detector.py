from ood_detector.out_of_distribution_detector import OutOfDistributionDetector
from classifier.pi_svm import ProbabilityOfInclusionSVM
from utils.constants import PI_SVM
from utils.open_set_grid_search import OpenSetGridSearch, PredictWithProbScorer
from sklearn.metrics import accuracy_score, roc_auc_score
from utils.metrics import normalized_accuracy_with_prob
import numpy as np


class PISVMOutOfDistributionDetector(OutOfDistributionDetector):
    def __init__(self):
        super(PISVMOutOfDistributionDetector, self).__init__()

        self.clf = None

    @classmethod
    def name(cls):
        return PI_SVM

    def fit(self, X, y, **fit_kwargs):
        estimator = ProbabilityOfInclusionSVM()
        param_grid = {
            "C": [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
        }

        scoring = PredictWithProbScorer(score_func=normalized_accuracy_with_prob, sign=1, kwargs={})
        self.clf = OpenSetGridSearch(
            estimator=estimator,
            n_splits=100,
            param_grid=param_grid,
            scoring=scoring,
            verbose=0,
            n_jobs=-1,
        )

        self.clf.fit(X, y)

    def predict(self, X, **kwargs):
        y_pred, y_prob = self.clf.predict(X)
        return y_pred, y_prob

    def eval_ind_accuracy(self, X, y):
        y_pred, _ = self.predict(X)
        return accuracy_score(y, y_pred)

    def eval_ood_auc(self, X_known, X_unknown):
        _, y_prob_known = self.clf.predict(X_known)
        _, y_prob_unknown = self.clf.predict(X_unknown)

        # Known samples should receive higher probability of inclusion
        return roc_auc_score(
            y_true=np.concatenate([np.ones(len(X_known)), np.zeros(len(X_unknown))]),
            y_score=np.concatenate([y_prob_known, y_prob_unknown]))
