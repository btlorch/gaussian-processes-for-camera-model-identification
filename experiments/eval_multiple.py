from experiments.eval_utils import eval_gpc_detector
import os
import argparse
import time
from tqdm.auto import tqdm
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required args
    parser.add_argument("--output_dir", required=True, type=str, help="Where to store resulting csv file")
    parser.add_argument("--dresden_dir", required=True, type=str, help="Path to folder containing Dresden images and dresden.csv")
    parser.add_argument("--features_file", required=True, type=str, help="Path to features HDF5 file")
    parser.add_argument("--model_base_dir", required=True, type=str, help="Where to find model directories")

    # Optional args
    parser.add_argument("--num_likelihood_samples", type=int, help="Number of Monte Carlo draws", default=1000)
    args = vars(parser.parse_args())

    # Set up output file
    output_filename = time.strftime("%Y_%m_%d") + "-eval_multiple.csv"
    output_csv = os.path.join(args["output_dir"], output_filename)

    # Find models
    model_dirs = sorted([dirname for dirname in os.listdir(args["model_base_dir"]) if os.path.isdir(os.path.join(args["model_base_dir"], dirname))])

    # Loop over models
    buffer = []
    for model_dir in tqdm(model_dirs, desc="Iteration models"):
        scores = eval_gpc_detector(
            model_dir=os.path.join(args["model_base_dir"], model_dir),
            features_file=args["features_file"],
            dresden_dir=args["dresden_dir"],
            num_likelihood_samples=args["num_likelihood_samples"],
        )

        if scores is not None:
            # A warning has already been logged
            buffer.append(scores)

    df = pd.DataFrame(buffer)
    df.to_csv(output_csv, index=False)
