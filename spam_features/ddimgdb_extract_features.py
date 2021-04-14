from spam_features.residual_based_local_features import residual_based_local_features
from utils.color import color2gray
from utils.writer import BufferedWriter
from utils.constants import FULL_RESOLUTION
from utils.logger import setup_custom_logger
from utils.postprocessing import IdentityPostprocessing
from experiments.data.data_controller import read_dresden_df
import multiprocessing as mp
from tqdm.auto import tqdm
from imageio import imread
import numpy as np
import functools
import argparse
import time
import os


log = setup_custom_logger(os.path.basename(__file__))


def process_row(row, dresden_dir, crop=FULL_RESOLUTION, postprocessing=IdentityPostprocessing()):
    """
    Processes one row of the dresden dataframe.
    :param row: row from dresden dataframe
    :param dresden_dir: directory where the ddimgdb images are located
    :param crop: FULL_RESOLUTION or integer of crop size for center crop
    :param postprocessing: instance of Postprocessing. Default is IdentityPostprocessing
    :return: single-instance batch, or None on failure
    """
    row = row.iloc[0]
    filename = os.path.join(dresden_dir, row["filename"])
    img = imread(filename)

    # Crop image if desired
    if crop != FULL_RESOLUTION:
        assert np.issubdtype(int, np.integer)

        height, width = img.shape[:2]
        if height < crop or width < crop:
            log.warning(f"Image \"{filename}\" is too small for crop window")
            return None

        offset_y = height - crop // 2
        offset_x = width - crop // 2

        img = img[offset_y:offset_y + height, offset_x:offset_x + width]

    gray = color2gray(img)

    # Apply post-processing
    gray_postprocessed = postprocessing(gray)

    features = residual_based_local_features(gray_postprocessed)

    ret = {
        "filename": [row["filename"]],
        "brand": [row["brand"]],
        "model": [row["model"]],
        "instance": [row["instance"]],
        "crop": [crop],
        "features": np.expand_dims(features, axis=0),
        "postprocessing": [postprocessing.name()]
    }

    for key, value in postprocessing.details().items():
        ret[key] = [value]

    return ret


def extract_features(dresden_df, dresden_dir, output_filepath, crop=FULL_RESOLUTION, postprocessing=IdentityPostprocessing()):
    writer = BufferedWriter(output_filepath)

    num_threads = mp.cpu_count() * 2
    with mp.Pool(processes=num_threads) as p:
        for result in tqdm(p.imap(functools.partial(process_row, dresden_dir=dresden_dir, crop=crop, postprocessing=postprocessing), np.array_split(dresden_df, len(dresden_df))), desc="Feature extraction", total=len(dresden_df)):
            # Process the result right away when it is ready

            # Skip failures
            if result is None:
                continue

            writer.write(result)

    writer.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required
    parser.add_argument("--output_dir", type=str, help="Where to store resulting features", required=True)

    # Optional
    parser.add_argument("--dresden_dir", type=str, help="Path to folder containing Dresden images and dresden.csv", default="/mnt/nfs/DresdenImageDB")
    parser.add_argument("--crop", type=int, help="Crop images to central patch of given size")
    args = vars(parser.parse_args())

    np.random.seed(91058)

    dresden_csv = os.path.join(args["dresden_dir"], "dresden.csv")
    dresden_df = read_dresden_df(dresden_csv)

    crop = args["crop"]
    if crop is None:
        crop = FULL_RESOLUTION

    output_filename = time.strftime("%Y_%m_%d") + "-dresden_spam_features_crop_{}.h5".format(str(crop))
    output_filepath = os.path.join(args["output_dir"], output_filename)

    extract_features(dresden_df=dresden_df,
                     dresden_dir=args["dresden_dir"],
                     output_filepath=output_filepath,
                     crop=crop)
