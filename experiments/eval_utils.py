from utils.constants import GAUSSIAN_PROCESS_CLASSIFIER
from ood_detector.bootstrap import set_up_detector
from experiments.data.data_setup import load_data
from utils.logger import setup_custom_logger
import numpy as np
import json
import os


log = setup_custom_logger(os.path.basename(__file__))


def eval_detector(data, detector):
    # Destruct data
    num_classes = data["num_classes"]
    num_features = data["num_features"]
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    X_ood = data["X_ood"]

    # Fit
    detector.fit(X_train, y_train)

    # Evaluate in-distribution test accuracy
    ind_accuracy = detector.eval_ind_accuracy(X_test, y_test)

    # Evaluate out-of-distribution AUC
    # The number of test and out-of-distribution images may be different.
    # Choose the maximum number of images available in both sets
    min_num_test_ood_images = min(len(X_test), len(X_ood))
    X_ood_subset = X_ood[np.random.permutation(len(X_ood))[:min_num_test_ood_images]]
    X_test_subset = X_test[np.random.permutation(len(X_test))[:min_num_test_ood_images]]
    ood_auc = detector.eval_ood_auc(X_test_subset, X_ood_subset)

    scores = {
        "detector_name": detector.name(),
        "ind_accuracy": ind_accuracy,
        "ood_auc": ood_auc,
    }

    # Update with additional scores
    additional_scores = detector.eval_additional_scores(X_train=X_train, y_train=y_train)
    scores.update(additional_scores)

    return scores


def eval_gpc_detector(model_dir, features_file, dresden_dir, num_likelihood_samples):
    # Restore model args
    args_file = os.path.join(model_dir, "args.json")
    if not os.path.exists(args_file):
        log.warning("args.json does not exist for model directory \"{}\"".format(model_dir))
        return None

    with open(args_file, "r") as f:
        model_args = json.load(f)

    # Load training and evaluation data
    model_selection_rng = None
    if model_args["model_selection_seed"]:
        model_selection_rng = np.random.RandomState(model_args["model_selection_seed"])

    other_rng = None
    if model_args["seed"]:
        other_rng = np.random.RandomState(model_args["seed"])
        if model_selection_rng is None:
            model_selection_rng = other_rng

    data = load_data(
        features_file=features_file,
        dresden_dir=dresden_dir,
        fraction_motives=model_args["fraction_motives"],
        num_known_models=model_args["num_known_models"],
        model_selection_rng=model_selection_rng,
        other_rng=other_rng)

    # Restore ood_detector
    model_file = os.path.join(model_dir, "gpc_multiclass_model.pth")

    num_inducing_points = model_args["num_inducing_points"]
    detector = set_up_detector(GAUSSIAN_PROCESS_CLASSIFIER,
                               model_file=model_file,
                               num_inducing_points=num_inducing_points,
                               num_likelihood_samples=num_likelihood_samples,
                               lengthscale_prior_alpha=model_args["lengthscale_prior_alpha"],
                               lengthscale_prior_beta=model_args["lengthscale_prior_beta"],
                               enable_ard=model_args["enable_ard"])

    # Evaluate ood_detector
    scores = eval_detector(data, detector)

    # Store some additional metadata
    scores.update({
        "num_inducing_points": num_inducing_points,
        "model_dir": os.path.basename(model_dir),
        "model_selection_seed": model_args["model_selection_seed"],
        "seed": model_args["seed"],
        "torch_seed": model_args["torch_seed"],
        "enable_ard": model_args["enable_ard"],
        "lengthscale_prior_alpha": model_args["lengthscale_prior_alpha"],
        "lengthscale_prior_beta": model_args["lengthscale_prior_beta"],
    })

    return scores
