from utils.constants import GAUSSIAN_PROCESS_CLASSIFIER, FULL_RESOLUTION
from utils.postprocessing import JpegPostprocessing, GaussianBlurPostprocessing, AdditiveNoisePostprocessing
from ood_detector.bootstrap import set_up_detector
from spam_features.ddimgdb_extract_features import extract_features
from experiments.data.data_setup import load_data
import argparse
import time
import os
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
import h5py
import tempfile
import json


def eval_postprocessing(test_df, dresden_dir, postprocessing, scaler, detector):
    num_test_samples = len(test_df)

    with tempfile.NamedTemporaryFile(suffix=".h5") as f:
        extract_features(
            dresden_df=test_df,
            dresden_dir=dresden_dir,
            output_filepath=f.name,
            crop=FULL_RESOLUTION,
            postprocessing=postprocessing,
        )

        with h5py.File(f.name, "r") as g:
            filenames = g["filename"][()]
            features = g["features"][()].astype(np.float32)

    features_preprocessed = scaler.transform(features)
    y_pred, y_var, y_entropy = detector.predict(features_preprocessed, return_entropy=True)
    y_pred = np.argmax(y_pred, axis=1)

    return pd.DataFrame({
        "filename": filenames,
        "y_pred": y_pred,
        "y_var": y_var,
        "y_entropy": y_entropy,
        "y_true": test_df["y_true"].to_list(),
        "postprocessing": [postprocessing.name()] * num_test_samples,
        "postprocessing_details": [str(postprocessing.details())] * num_test_samples,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required args
    parser.add_argument("--model_dir", required=True, type=str, help="Path to save model file")
    parser.add_argument("--dresden_dir", required=True, type=str, help="Path to folder containing Dresden images and dresden.csv")
    parser.add_argument("--features_file", required=True, type=str, help="Path to features HDF5 file")

    # Optional args
    parser.add_argument("--num_likelihood_samples", type=int, help="Number of Monte Carlo draws", default=1000)

    args = vars(parser.parse_args())

    # Set up output file
    output_filename = time.strftime("%Y_%m_%d") + "-eval_postprocessing.csv"
    results_dir = os.path.join(args["model_dir"], "results")
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    output_csv = os.path.join(results_dir, output_filename)

    #
    # Set up model
    #
    # Restore model args
    args_file = os.path.join(args["model_dir"], "args.json")
    with open(args_file, "r") as f:
        model_args = json.load(f)

    model_file = os.path.join(args["model_dir"], "gpc_multiclass_model.pth")
    detector = set_up_detector(
        GAUSSIAN_PROCESS_CLASSIFIER,
        model_file=model_file,
        num_inducing_points=model_args["num_inducing_points"],
        num_likelihood_samples=args["num_likelihood_samples"],
        lengthscale_prior_alpha=model_args["lengthscale_prior_alpha"],
        lengthscale_prior_beta=model_args["lengthscale_prior_beta"],
        enable_ard=model_args["enable_ard"],
    )

    #
    # Load data
    #
    model_selection_rng = None
    if model_args["model_selection_seed"]:
        model_selection_rng = np.random.RandomState(model_args["model_selection_seed"])

    other_rng = None
    if model_args["seed"]:
        other_rng = np.random.RandomState(model_args["seed"])
        if model_selection_rng is None:
            model_selection_rng = other_rng

    data = load_data(
        features_file=args["features_file"],
        dresden_dir=args["dresden_dir"],
        fraction_motives=model_args["fraction_motives"],
        num_known_models=model_args["num_known_models"],
        model_selection_rng=model_selection_rng,
        other_rng=other_rng,
    )

    # Destruct data
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    test_df = data["test_df"].copy()
    X_ood = data["X_ood"]
    scaler = data["scaler"]

    # Add y_true label
    test_df["y_true"] = y_test

    # Restore trained parameters
    detector.fit(X_train, y_train)

    # As a sanity check, print the in-distribution classification accuracy and out-of-distribution AUC
    # Note that computing the results from the csv file can give slightly different results to calling detector.eval(), because separate calls to predict() involve some randomness.
    ind_accuracy = detector.eval_ind_accuracy(X_test, y_test)
    print(f"In-distribution accuracy: {ind_accuracy:3.2f}")

    # Evaluate out-of-distribution AUC
    # The number of test and out-of-distribution images may be different.
    # Choose the maximum number of images available in both sets
    min_num_test_ood_images = min(len(X_test), len(X_ood))
    X_ood_subset = X_ood[np.random.permutation(len(X_ood))[:min_num_test_ood_images]]
    X_test_subset = X_test[np.random.permutation(len(X_test))[:min_num_test_ood_images]]
    ood_auc = detector.eval_ood_auc(X_test_subset, X_ood_subset)
    print(f"Out-of-distribution AUC: {ood_auc:3.2f}")

    #
    # Evaluation time
    #
    buffer = []
    num_test_samples = len(X_test)

    # Evaluate detector on in-distribution test data
    y_pred, y_var, y_entropy = detector.predict(X_test, return_entropy=True)
    y_pred = np.argmax(y_pred, axis=1)

    buffer.append(pd.DataFrame({
        "filename": test_df["filename"].to_list(),
        "y_pred": y_pred,
        "y_var": y_var,
        "y_entropy": y_entropy,
        "y_true": y_test,
        "postprocessing": ["none"] * num_test_samples
    }))

    # Evaluate on different types of post-processing

    # JPEG compression
    for quality_factor in tqdm(np.arange(15, 105, 5), desc="Evaluating JPEG compression"):
        res = eval_postprocessing(
            test_df=test_df,
            dresden_dir=args["dresden_dir"],
            postprocessing=JpegPostprocessing(quality_factor=quality_factor),
            scaler=scaler,
            detector=detector,
        )

        buffer.append(res)

    # Gaussian blur
    for sigma in tqdm(np.linspace(0, 2, num=20), desc="Evaluating Gaussian blur"):
        res = eval_postprocessing(
            test_df=test_df,
            dresden_dir=args["dresden_dir"],
            postprocessing=GaussianBlurPostprocessing(sigma=sigma),
            scaler=scaler,
            detector=detector
        )

        buffer.append(res)

    # Additive noise
    for sigma in tqdm(np.linspace(0, 4, num=40), desc="Evaluating additive noise"):
        res = eval_postprocessing(
            test_df=test_df,
            dresden_dir=args["dresden_dir"],
            postprocessing=AdditiveNoisePostprocessing(sigma=sigma),
            scaler=scaler,
            detector=detector
        )

        buffer.append(res)

    # Store results to csv file
    df = pd.concat(buffer)
    df.to_csv(output_csv, index=False)
