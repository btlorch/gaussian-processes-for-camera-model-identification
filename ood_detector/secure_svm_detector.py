from ood_detector.out_of_distribution_detector import OutOfDistributionDetector
from classifier.secure_svm_classifier import SecureSVMClassifier
from utils.open_set_grid_search import OpenSetGridSearch
from utils.metrics import normalized_accuracy
from utils.constants import SECURE_SVM
from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score
import numpy as np


class SecureSVMOutOfDistributionDetector(OutOfDistributionDetector):
    def __init__(self):
        super(SecureSVMOutOfDistributionDetector, self).__init__()
        self.clf = None

    @classmethod
    def name(cls):
        return SECURE_SVM

    def fit(self, X, y, **fit_kwargs):
        # Grid search over the best parameters.
        estimator = SecureSVMClassifier()
        param_grid = {
            "C": [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
            "nu": [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.2, 0.5]
        }
        scoring = make_scorer(normalized_accuracy)
        self.clf = OpenSetGridSearch(estimator=estimator, n_splits=100, param_grid=param_grid, scoring=scoring, verbose=0, n_jobs=-1)
        self.clf.fit(X, y)

    def predict(self, X, **kwargs):
        return self.clf.predict(X)

    def eval_ind_accuracy(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def eval_ood_auc(self, X_known, X_unknown):
        y_pred_known = self.clf.predict(X_known)
        y_pred_unknown = self.clf.predict(X_unknown)

        y_score_known = (y_pred_known >= 0).astype(np.float)
        y_score_unknown = (y_pred_unknown >= 0).astype(np.float)

        return roc_auc_score(
            y_true=np.concatenate([np.ones(len(y_pred_known)), np.zeros(len(y_pred_unknown))]),
            y_score=np.concatenate([y_score_known, y_score_unknown])
        )
