from utils.constants import GAUSSIAN_PROCESS_CLASSIFIER, SECURE_SVM, PI_SVM, FULL_RESOLUTION
from experiments.data.data_setup import load_data
from experiments.eval_utils import eval_detector
from ood_detector.bootstrap import set_up_detector
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import argparse
import time
import json
import re
import os


def eval_detectors(model_dir, **kwargs):
    args_file = os.path.join(model_dir, "args.json")
    with open(args_file, "r") as f:
        model_args = json.load(f)

    features_file = os.path.join(kwargs["features_dir"], os.path.basename(model_args["features_file"]))

    # Extract crop from filename
    crop = re.search("_crop_([a-z_0-9]+).h5$", features_file).group(1)

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
        dresden_dir=kwargs["dresden_dir"],
        fraction_motives=model_args["fraction_motives"],
        num_known_models=model_args["num_known_models"],
        model_selection_rng=model_selection_rng,
        other_rng=other_rng)

    buffer = []

    # Gaussian process classifier ood_detector
    model_file = os.path.join(model_dir, "gpc_multiclass_model.pth")
    detector = set_up_detector(
        GAUSSIAN_PROCESS_CLASSIFIER,
        model_file=model_file,
        num_inducing_points=model_args["num_inducing_points"],
        num_likelihood_samples=kwargs["num_likelihood_samples"],
        enable_ard=model_args["enable_ard"],
    )
    scores = eval_detector(data, detector)
    scores["crop"] = crop
    scores["num_inducing_points"] = model_args["num_inducing_points"]
    scores["model_dir"] = os.path.basename(model_dir)
    buffer.append(scores)

    # CCF
    detector = set_up_detector(SECURE_SVM)
    scores = eval_detector(data, detector)
    scores["crop"] = crop
    buffer.append(scores)

    # PI-SVM
    detector = set_up_detector(PI_SVM)
    scores = eval_detector(data, detector)
    scores["crop"] = crop
    buffer.append(scores)

    return buffer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required args
    parser.add_argument("--output_dir", required=True, type=str, help="Where to store resulting csv file")
    parser.add_argument("--dresden_dir", required=True, type=str, help="Path to folder containing Dresden images and dresden.csv")
    parser.add_argument("--features_dir", required=True, type=str, help="Path to folder containing the features HDF5 file")
    parser.add_argument("--model_base_dir", required=True, type=str, help="Path to model directories")

    # Optional args
    parser.add_argument("--num_likelihood_samples", type=int, help="Number of Monte Carlo draws", default=1000)

    args = vars(parser.parse_args())

    # Set up output file
    output_filename = time.strftime("%Y_%m_%d") + "-eval_and_compare_multiple.csv"
    output_csv = os.path.join(args["output_dir"], output_filename)

    # Find all models
    model_dirs = sorted([dirname for dirname in os.listdir(args["model_base_dir"]) if os.path.isdir(os.path.join(args["model_base_dir"], dirname))])

    # Loop over models
    buffer = []
    for model_dir in tqdm(model_dirs, desc="Iterating data splits/models"):
        buffer.extend(eval_detectors(model_dir=os.path.join(args["model_base_dir"], model_dir), **args))

    df = pd.DataFrame(buffer)
    df.to_csv(output_csv, index=False)
