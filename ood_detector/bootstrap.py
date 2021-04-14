from ood_detector.secure_svm_detector import SecureSVMOutOfDistributionDetector
from ood_detector.pi_svm_detector import PISVMOutOfDistributionDetector
from ood_detector.gpc_detector import GaussianProcessClassifierOutOfDistributionDetector


def set_up_detector(detector_name, **detector_args):
    if GaussianProcessClassifierOutOfDistributionDetector.name() == detector_name:
        return GaussianProcessClassifierOutOfDistributionDetector(
            model_file=detector_args["model_file"],
            num_inducing_points=detector_args["num_inducing_points"],
            num_likelihood_samples=detector_args.get("num_likelihood_samples", 100),
            lengthscale_prior_alpha=detector_args.get("lengthscale_prior_alpha", None),
            lengthscale_prior_beta=detector_args.get("lengthscale_prior_beta", None),
            enable_ard=detector_args["enable_ard"],
        )

    elif SecureSVMOutOfDistributionDetector.name() == detector_name:
        return SecureSVMOutOfDistributionDetector()

    elif PISVMOutOfDistributionDetector.name() == detector_name:
        return PISVMOutOfDistributionDetector()

    else:
        raise ValueError("Unknown ood_detector name")
