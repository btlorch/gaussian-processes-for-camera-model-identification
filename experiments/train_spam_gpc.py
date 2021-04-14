from classifier.gp_classification_model import InducingGPClassificationModel
from utils.parse import nullable_int, nullable_float
from experiments.data.data_setup import load_data
from utils.logger import setup_custom_logger
from timeit import default_timer as timer
from datetime import timedelta
from sklearn.metrics import roc_auc_score
from distutils.util import strtobool
from tensorboardX import SummaryWriter
import numpy as np
import argparse
import torch
import gpytorch
import time
import json
import os


log = setup_custom_logger(os.path.basename(__file__))


def train_eval_gp(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args["no_cuda"] else "cpu")

    # Load data
    # Set up random number generator for random selection of camera models
    model_selection_rng = None
    if args["model_selection_seed"]:
        model_selection_rng = np.random.RandomState(args["model_selection_seed"])

    # Set up another random number generator for everything else
    other_rng = None
    if args["seed"]:
        other_rng = np.random.RandomState(args["seed"])
        if model_selection_rng is None:
            model_selection_rng = other_rng

    data = load_data(
        features_file=args["features_file"],
        dresden_dir=args["dresden_dir"],
        fraction_motives=args["fraction_motives"],
        num_known_models=args["num_known_models"],
        model_selection_rng=model_selection_rng,
        other_rng=other_rng)

    # Destruct data dict
    num_classes = data["num_classes"]
    num_features = data["num_features"]
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    X_ood = data["X_ood"]

    # Convert to Torch tensors
    X_train_torch = torch.from_numpy(X_train)
    y_train_torch = torch.from_numpy(y_train)
    X_test_torch = torch.from_numpy(X_test)
    X_ood_torch = torch.from_numpy(X_ood)

    # Initialize inducing points by selecting a random subset of the training points
    num_training_points = len(X_train)
    if args["torch_seed"]:
        torch.manual_seed(args["torch_seed"])

    permutation = torch.randperm(num_training_points)
    num_inducing_points = min(args["max_num_inducing_points"], num_training_points)
    log.info(f"Number of training points: {num_training_points}, number of inducing points: {num_inducing_points}")
    Z_torch = X_train_torch[permutation[:num_inducing_points]]

    # Store the number of inducing points alongside with the model
    args["num_inducing_points"] = num_inducing_points

    # Move to GPU if available
    X_train_torch = X_train_torch.to(device)
    y_train_torch = y_train_torch.to(device)
    X_test_torch = X_test_torch.to(device)
    X_ood_torch = X_ood_torch.to(device)
    Z_torch = Z_torch.to(device)

    # Set up model and likelihood
    ard_num_dims = num_features if args["enable_ard"] is True else None
    model = InducingGPClassificationModel(
        inducing_points=Z_torch,
        num_classes=num_classes,
        ard_num_dims=ard_num_dims,
        learn_inducing_locations=args["learn_inducing_locations"],
        lengthscale_prior_alpha=args["lengthscale_prior_alpha"],
        lengthscale_prior_beta=args["lengthscale_prior_beta"])
    likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_classes=num_classes, mixing_weights=False)

    model.to(device)
    likelihood.to(device)

    # Summary writer
    logdir = os.path.join(args["logdir"], time.strftime("%Y_%m_%d_%H_%M_%S") + "-gpc")
    if args["logdir_suffix"] is not None:
        logdir = logdir + "-" + args["logdir_suffix"]
    model_path = os.path.join(logdir, "gpc_multiclass_model.pth")

    # Create log directory
    os.mkdir(logdir)
    writer = SummaryWriter(logdir=logdir, flush_secs=30)

    # Dump args to log directory
    args_file = os.path.join(logdir, "args.json")
    with open(args_file, "w") as f:
        json.dump(args, f, indent=4, sort_keys=True)

    # Find optimal model hyper-parameters
    model.train()
    likelihood.train()

    # Use the Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=1000)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=y_train_torch.numel())

    # Stop training when any of the two stopping criteria are reached:
    # - Maximum number of iterations exceeded
    # - Loss has not improved for `initial_patience` epochs
    best_loss = np.inf
    patience = args["initial_patience"]

    i = 0
    start_time = timer()
    while i < args["max_train_iters"] and patience >= 0:
        # Go back into training mode
        model.train()
        likelihood.train()

        # Zero backpropped gradients from previous iteration
        optimizer.zero_grad()

        # Get predictive output
        output = model(X_train_torch)

        # Calc loss and backprop gradients
        with gpytorch.settings.num_likelihood_samples(args["num_likelihood_samples"]):
            loss = -mll(output, y_train_torch)

        loss.backward()
        writer.add_scalar("train/loss", loss.item(), global_step=i)
        writer.add_scalar("learning_rate", optimizer.param_groups[0]['lr'], global_step=i)

        optimizer.step()

        # Evaluate model every couple of training steps
        if i % 20 == 0:
            # If training loss reached a global minimum, save the trained model
            if loss.item() < best_loss:
                torch.save(model.state_dict(), model_path)
                best_loss = loss.item()
                patience = args["initial_patience"]
            else:
                patience -= 1

            print('Iter %d/%d - Loss: %.3f' % (i, args["max_train_iters"], loss.item()))

        if i % args["xval_frequency"] == 0:
            # Switch to evaluation mode
            model.eval()
            likelihood.eval()

            # Log gradients
            for name, param in model.named_parameters():
                writer.add_histogram(name, param, global_step=i)
                if param.grad is not None:
                    writer.add_histogram(name + "_grad", param.grad, global_step=i)

        scheduler.step(loss.item())
        i += 1

    end_time = timer()
    train_time_seconds = end_time - start_time
    writer.close()
    log.info(f"Saved model to {model_path}")
    log.info(f"Train time: {timedelta(seconds=train_time_seconds)}")

    #
    # Evaluate
    #
    model.eval()
    likelihood.eval()

    with gpytorch.settings.num_likelihood_samples(args["num_likelihood_samples"]):
        y_pred = likelihood(model(X_test_torch))

    probs = y_pred.probs.mean(dim=0)
    y_var = y_pred.probs.var(dim=0).detach().cpu().numpy().mean(axis=1)
    accuracy = np.mean(np.argmax(probs.detach().cpu().numpy(), axis=1) == y_test)
    print("Test accuracy: {:3.2f}".format(accuracy))

    with gpytorch.settings.num_likelihood_samples(args["num_likelihood_samples"]):
        y_pred_ood = likelihood(model(X_ood_torch))

    y_var_ood = y_pred_ood.probs.var(dim=0).detach().cpu().numpy().mean(axis=1)

    # Note that len(X_test) != len(X_ood)
    num_common_test_samples = min(len(y_var), len(y_var_ood))
    y_var = y_var[:num_common_test_samples]
    y_var_ood = y_var_ood[:num_common_test_samples]

    auc = roc_auc_score(
        y_true=np.concatenate([np.zeros(len(y_var)), np.ones(len(y_var_ood))]),
        y_score=np.concatenate([y_var, y_var_ood]))
    print("Out-of-distribution AUC = {:3.2f}".format(auc))

    # Dump training records with preliminary results to JSON file
    training_records = {
        "train_time_seconds": train_time_seconds,
        "test_accuracy": accuracy,
        "ood_auc": auc,
    }
    training_records_file = os.path.join(logdir, "train_records.json")
    with open(training_records_file, "w") as f:
        json.dump(training_records, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--dresden_dir", required=True, type=str, help="Path to folder containing Dresden images and dresden.csv")
    parser.add_argument("--features_file", required=True, type=str, help="Path to features")
    parser.add_argument("--fraction_motives", type=float, help="Fraction of motives to include in the train/validation set", default=0.8)
    parser.add_argument("--num_known_models", type=int, help="Restrict number of known camera models", default=10)
    parser.add_argument("--seed", type=nullable_int, help="Optional seed for random state")
    parser.add_argument("--model_selection_seed", type=nullable_int, help="Seed for random state to select known camera models")

    # Model
    parser.add_argument("--max_num_inducing_points", type=int, help="Maximum number of inducing points", default=512)
    parser.add_argument("--learn_inducing_locations", type=lambda x: bool(strtobool(str(x))), help="Whether to learn inducing locations", default=True)
    parser.add_argument("--num_likelihood_samples", type=int, help="Number of likelihood samples", default=100)
    parser.add_argument("--lengthscale_prior_alpha", type=nullable_float, help="Alpha parameter of lengthscale prior")
    parser.add_argument("--lengthscale_prior_beta", type=nullable_float, help="Beta parameter of lengthscale prior")
    parser.add_argument("--enable_ard", type=lambda x: bool(strtobool(str(x))), help="Whether to enable ARD", default=False)

    # Training
    parser.add_argument("--no_cuda", default=False, action="store_true", help="Explicitly disable CUDA (default: Use CUDA)")
    parser.add_argument("--logdir", required=True, type=str, help="Path to logdir")
    parser.add_argument("--logdir_suffix", type=str, help="Suffix to append to log directory name")
    parser.add_argument("--initial_patience", type=int, help="If model does not improve for this number of steps, the training terminates", default=1000)
    parser.add_argument("--max_train_iters", type=int, help="Maximum number of training steps", default=1000000)
    parser.add_argument("--torch_seed", type=nullable_int, help="Seed for torch random number generator")
    parser.add_argument("--xval_frequency", type=int, help="Extended evaluation after the given number of steps", default=20)

    args = vars(parser.parse_args())

    train_eval_gp(args)
