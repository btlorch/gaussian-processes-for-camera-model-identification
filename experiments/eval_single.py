from experiments.eval_utils import eval_gpc_detector
import pandas as pd
import argparse
import time
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required arguments with reasonable default values
    parser.add_argument("--dresden_dir", required=True, type=str, help="Path to folder containing Dresden images and dresden.csv")
    parser.add_argument("--features_file", required=True, type=str, help="Path to features HDF5 file")
    parser.add_argument("--model_dir", required=True, type=str, help="Path to save model file")
    parser.add_argument("--num_likelihood_samples", type=int, help="Number of Monte Carlo draws", default=1000)
    args = vars(parser.parse_args())

    scores = eval_gpc_detector(**args)
    scores_df = pd.DataFrame([scores])

    # Set up output file
    output_filename = time.strftime("%Y_%m_%d") + "-eval_single.csv"
    results_dir = os.path.join(args["model_dir"], "results")
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    output_csv = os.path.join(results_dir, output_filename)
    scores_df.to_csv(output_csv, index=False)
