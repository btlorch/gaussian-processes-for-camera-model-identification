from abc import ABC, abstractmethod


class OutOfDistributionDetector(ABC):
    def __init__(self):
        pass

    @classmethod
    @abstractmethod
    def name(cls):
        pass

    @staticmethod
    def fit(self, X, y, **kwargs):
        pass

    @abstractmethod
    def predict(self, X, **kwargs):
        pass

    @staticmethod
    def eval_ind_accuracy(self, X, y):
        pass

    @abstractmethod
    def eval_ood_auc(self, X_known, X_unknown):
        pass

    def eval_additional_scores(self, **kwargs):
        return {}
