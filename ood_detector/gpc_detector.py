from ood_detector.out_of_distribution_detector import OutOfDistributionDetector
from classifier.gp_classification_model import InducingGPClassificationModel
from utils.constants import GAUSSIAN_PROCESS_CLASSIFIER
import torch
import gpytorch
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np


class GaussianProcessClassifierOutOfDistributionDetector(OutOfDistributionDetector):
    def __init__(self,
                 model_file,
                 num_inducing_points,
                 num_likelihood_samples,
                 lengthscale_prior_alpha=None,
                 lengthscale_prior_beta=None,
                 enable_ard=True):
        super(GaussianProcessClassifierOutOfDistributionDetector, self).__init__()

        self.model_file = model_file
        self.num_inducing_points = num_inducing_points
        self.num_likelihood_samples = num_likelihood_samples
        self.lengthscale_prior_alpha = lengthscale_prior_alpha
        self.lengthscale_prior_beta = lengthscale_prior_beta
        self.enable_ard = enable_ard

        self.model = None
        self.likelihood = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def name(cls):
        return GAUSSIAN_PROCESS_CLASSIFIER

    def fit(self, X, y, **kwargs):
        num_features = X.shape[1]
        num_classes = len(set(y))

        # Create dummy inducing locations
        inducing_points = torch.zeros([self.num_inducing_points, num_features])
        ard_num_dims = num_features if self.enable_ard is True else None
        model = InducingGPClassificationModel(
            inducing_points=inducing_points,
            num_classes=num_classes,
            ard_num_dims=ard_num_dims,
            lengthscale_prior_alpha=self.lengthscale_prior_alpha,
            lengthscale_prior_beta=self.lengthscale_prior_beta,
        )
        likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_classes=num_classes, mixing_weights=False)
        state_dict = torch.load(self.model_file, map_location=self.device)
        model.load_state_dict(state_dict)

        self.model = model
        self.likelihood = likelihood

        self.model.to(self.device)
        self.likelihood.to(self.device)

    def predict(self, X, **kwargs):
        self.model.eval()
        self.likelihood.eval()

        X_torch = torch.from_numpy(X).to(self.device)
        with gpytorch.settings.num_likelihood_samples(self.num_likelihood_samples):
            f_pred = self.model(X_torch)
            y_pred = self.likelihood(f_pred)

        probs = y_pred.probs.mean(dim=0).detach().cpu().numpy()
        y_var = y_pred.probs.var(dim=0).detach().cpu().numpy().mean(axis=1)
        y_entropy = y_pred.entropy().mean(dim=0).detach().cpu().numpy()

        if kwargs.get("return_entropy", None):
            return probs, y_var, y_entropy

        return probs, y_var

    def eval_ind_accuracy(self, X, y):
        y_pred, _ = self.predict(X)
        y_pred = np.argmax(y_pred, axis=1)
        return accuracy_score(y, y_pred)

    def eval_ood_auc(self, X_known, X_unknown):
        assert len(X_known) == len(X_unknown), "Expected known and unknown sets to have same length"

        _, y_var_known = self.predict(X_known)
        _, y_var_unknown = self.predict(X_unknown)

        return roc_auc_score(
            y_true=np.concatenate([np.zeros(len(X_known)), np.ones(len(X_unknown))]),
            y_score=np.concatenate([y_var_known, y_var_unknown]))

    def eval_additional_scores(self, **kwargs):
        """
        Compute evidence lower bound on the training data
        :param kwargs: must contains X_train and y_train
        :return: dict with (positive) ELBO that is maximized during training
        """
        self.model.eval()
        self.likelihood.eval()

        X_train_torch = torch.from_numpy(kwargs["X_train"]).to(self.device)
        y_train_torch = torch.from_numpy(kwargs["y_train"]).to(self.device)
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=y_train_torch.numel())

        with torch.no_grad(), gpytorch.settings.num_likelihood_samples(self.num_likelihood_samples):
            f_pred = self.model(X_train_torch)
            elbo = mll(f_pred, y_train_torch).item()

        return {
            "elbo": elbo
        }
