import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import IndependentMultitaskVariationalStrategy
from utils.logger import setup_custom_logger
import os


log = setup_custom_logger(os.path.basename(__file__))


class InducingGPClassificationModel(ApproximateGP):
    def __init__(self,
                 inducing_points,
                 num_classes,
                 ard_num_dims=None,
                 initial_lengthscale=None,
                 initial_outputscale=None,
                 learn_inducing_locations=True,
                 lengthscale_prior_alpha=None,
                 lengthscale_prior_beta=None):
        """
        Initializes a variational Gaussian process classifier with the given inducing points and whitening.

        :param inducing_points: Tensor containing the inducing point locations, shape [num_inducing_points, num_features]
        :param num_classes: number of classes
        :param ard_num_dims: number of feature dimensions for ARD prior, or None for shared lengthscale
        :param initial_lengthscale: initial value for all length scales
        :param initial_outputscale: initial value for all output scales
        :param learn_inducing_locations: whether or not to adjust the inducing locations
        :param lengthscale_prior_alpha: concentration parameter alpha of Gamma distribution used as lengthscale prior
        :param lengthscale_prior_beta: rate parameter beta of Gamma distribution used as lengthscale prior
        """

        # Ensure both alpha and beta, or none of them are set
        assert (lengthscale_prior_alpha is not None) == (lengthscale_prior_beta is not None), "Either both alpha and beta or none of them must be set"

        # Set up variational distribution with full covariance
        num_inducing_points = len(inducing_points)
        variational_distribution = CholeskyVariationalDistribution(num_inducing_points=num_inducing_points, batch_shape=torch.Size([num_classes]))

        base_variational_strategy = gpytorch.variational.VariationalStrategy(
            model=self,
            inducing_points=inducing_points,
            variational_distribution=variational_distribution,
            learn_inducing_locations=learn_inducing_locations
        )

        # Produce a `MultitaskMultivariateNormal` distribution with one task per class
        variational_strategy = IndependentMultitaskVariationalStrategy(base_variational_strategy, num_tasks=num_classes)

        super(InducingGPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()

        # Lengthscale prior
        lengthscale_prior = None
        if lengthscale_prior_alpha is not None:
            log.info(f"Setting lengthscale prior with alpha = {lengthscale_prior_alpha} and beta = {lengthscale_prior_beta}")
            lengthscale_prior = gpytorch.priors.GammaPrior(lengthscale_prior_alpha, lengthscale_prior_beta)

        # Covariance kernel: ScaleKernel + RBFKernel
        rbf_kernel = gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_classes]), ard_num_dims=ard_num_dims, lengthscale_prior=lengthscale_prior)
        scale_kernel = gpytorch.kernels.ScaleKernel(rbf_kernel, batch_shape=torch.Size([num_classes]))

        # Set initial length scale
        if initial_lengthscale is not None:
            log.info(f"Initializing lengthscale to {initial_lengthscale}")
            rbf_kernel.lengthscale = initial_lengthscale

        # Set initial scale
        if initial_outputscale is not None:
            log.info(f"Initializing outputscale to {initial_outputscale}")
            scale_kernel.outputscale = initial_outputscale

        self.covar_module = scale_kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred
