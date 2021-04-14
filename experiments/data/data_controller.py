import h5py
import numpy as np
import pandas as pd
import os
from utils.logger import setup_custom_logger


log = setup_custom_logger(os.path.basename(__file__))


def read_dresden_df(dresden_csv):
    """
    Read dresden-csv file and drop a number of blacklisted images
    :param dresden_csv: path to dresden.csv
    :return: data frame
    """

    # Blacklist some images
    blacklist = [
        "Samsung_L74wide_0_43583.JPG",
    ]

    dresden_df = pd.read_csv(dresden_csv)
    num_total_images = len(dresden_df)
    dresden_df = dresden_df[~dresden_df["filename"].isin(blacklist)]
    log.info(f"Filtered {num_total_images-len(dresden_df):d} blacklisted image(s) from the Dresden data set")

    # Reset index such that the next index does not contain any gaps
    # Keep old index as column "index_org"
    dresden_df = dresden_df.reset_index().rename(columns={"index": "index_org"})

    # Ensure that the new index does not contain any gaps
    assert np.allclose(dresden_df.index, np.arange(len(dresden_df)))
    assert dresden_df["index_org"].is_monotonic
    return dresden_df


class DataController(object):
    def __init__(self, features_file, dresden_dir):
        self.features_file = features_file
        self.dresden_dir = dresden_dir

        #
        # Init
        #

        # Read in dresden.csv
        dresden_csv = os.path.join(dresden_dir, "dresden.csv")
        dresden_df = read_dresden_df(dresden_csv)

        # Load features
        with h5py.File(features_file, "r") as f:
            features = f["features"][()]

            # Ensure that the features are in the same order as the csv file
            assert f["filename"][()].astype(np.str).tolist() == dresden_df["filename"].to_list()

        # Append camera id column
        dresden_df["brand_model"] = dresden_df["brand"] + " " + dresden_df["model"]

        self.dresden_df = dresden_df
        self.features = features

    def filter_subset(self, selection_df):
        # Move index to column, merge data frames, then restore original index
        selected_dresden_df = self.dresden_df.reset_index().merge(selection_df, how="inner").set_index("index")
        return selected_dresden_df
