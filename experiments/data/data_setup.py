from experiments.data.data_controller import DataController
from sklearn.preprocessing import StandardScaler, LabelEncoder
from utils.logger import setup_custom_logger
import numpy as np
import os


log = setup_custom_logger(os.path.basename(__file__))


def load_data(features_file, dresden_dir, normalize_mean=True, normalize_std=True, fraction_motives=0.5, num_known_models=None, model_selection_rng=None, other_rng=None):
    """
    Replicate training setup from [1].
    As known camera models, select all camera models for which more than one instance is available.
    Randomly select one of the instances for training.
    Split the available motives into 50% seen during training (and validation) and 50% for testing.
    Determine the maximum number of images available for all of the selected training camera instances.
    Take 80% of those images for training, and 20% for validation.

    For testing, randomly select 50 images from the known camera models but unseen instances and unseen motives.
    In addition, randomly select 50 images from unknown camera models and return them as out-of-distribution data.

    Note that len(X_test) != len(X_ood).

    [1] L. Bondi, L. Baroffio, D. Güera, P. Bestagini, E. J. Delp and S. Tubaro, "First Steps Toward Camera Model Identification With Convolutional Neural Networks," in IEEE Signal Processing Letters, vol. 24, no. 3, pp. 259-263, March 2017.
    :param features_file: pre-extracted features HDF5 file
    :param dresden_dir: path to ddimgdb folder containing the "dresden.csv"
    :param normalize_mean: if True, normalize the data such that the mean of the training data is zero.
    :param normalize_std: if True, normalize the data such that standard deviation of the individual feature dimension in the training data is 1.
    :param fraction_motives: split motive names into two disjoint sets, one for training and validation, the other one for testing This number indicates the fraction of motives to include in the train/validation set.
    :param num_known_models: Number of camera models to include in the known set. The default is all camera models that have more than 1 instance (= 18 camera models).
    :param model_selection_rng: random state created by np.random.RandomState() used to sample the known camera model instances
    :param other_rng: random state for all other random ops. Default is the global state
    :return: dict with train, validation, test, and out-of-distribution data
    """
    if other_rng is None:
        other_rng = np.random.random.__self__

    # Sanitize input args
    if model_selection_rng is None:
        model_selection_rng = other_rng

    # Set up data controller to holds references to Dresden table and features array
    data_controller = DataController(features_file=features_file, dresden_dir=dresden_dir)

    data_df = data_controller.dresden_df.copy()

    #
    # Preparation and cleaning
    #
    # Combine D70 and D70s
    # Create a new column "clean_model" that combines D70 and D70s
    data_df["clean_model"] = data_df["model"].apply(lambda x: "D70" if x == "D70s" else x)

    # How many instances ot the D70 do we have?
    d70_instances = np.array(data_df[data_df["model"] == "D70"]["instance"].drop_duplicates())
    num_d70_instances = len(d70_instances)
    assert np.allclose(d70_instances, np.arange(num_d70_instances))

    # Treat D70s as additional instances of the D70
    data_df.loc[data_df["model"] == "D70s", "instance"] += num_d70_instances

    # Update "brand_model" column accordingly
    data_df["brand_model"] = data_df["brand"] + " " + data_df["clean_model"]

    # Select all those camera models of which more than 1 instance are available
    instance_count = data_df[["brand", "clean_model", "instance"]].drop_duplicates().groupby(["brand", "clean_model"]).count()
    model_selection_df = instance_count[instance_count["instance"] > 1].reset_index()[["brand", "clean_model"]]

    if num_known_models is not None:
        # Randomly select `num_known_models` out of the given ones
        model_selection_df = model_selection_df.sample(num_known_models, replace=False, random_state=model_selection_rng)

    # Apply the filter: Keep images of selection camera models only
    known_dresden_df = data_df.reset_index().merge(model_selection_df, how="inner").set_index("index")

    # For each camera model, randomly select 1 of the instances
    train_instances_df = known_dresden_df.groupby(["brand", "clean_model"]).sample(1, random_state=other_rng)[["brand", "clean_model", "instance"]].reset_index(drop=True)
    # Obtain all images that match the selected camera instances
    train_instances_imgs_df = data_df.reset_index().merge(train_instances_df, how="inner").set_index("index")

    # Sort by scenes
    all_motive_names = train_instances_imgs_df["motive_name"].unique()
    other_rng.shuffle(all_motive_names)

    num_train_motives = int(np.ceil(len(all_motive_names) * fraction_motives))
    train_motive_names = all_motive_names[:num_train_motives]
    test_motive_names = all_motive_names[num_train_motives:]

    # Filter selected images by motive number
    train_instances_motives_imgs_df = train_instances_imgs_df[train_instances_imgs_df["motive_name"].isin(train_motive_names)]

    # Determine minimum number of images available for all instances
    num_train_images_available = train_instances_motives_imgs_df.groupby(["brand", "clean_model"]).count()["filename"].min()

    # From each group, select training images
    train_df = train_instances_motives_imgs_df.groupby(["brand", "clean_model"]).sample(num_train_images_available, random_state=other_rng)
    assert len(set(train_df.index).intersection(train_instances_motives_imgs_df.index)) == len(train_df)

    # For testing, retain all images taken by known cameras, but discard
    # - Instances seen during training
    test_df = known_dresden_df[~known_dresden_df.index.isin(train_instances_imgs_df.index)]
    # - Motives seen during training
    test_df = test_df[test_df["motive_name"].isin(test_motive_names)]

    assert len(train_df) + len(test_df) <= len(known_dresden_df)
    assert len(set(test_df.index).intersection(known_dresden_df.index)) == len(test_df), "Test set indices are wrong"
    assert len(set(test_df.index).intersection(train_df.index)) == 0, "Training and test sets are not disjoint"

    X_train = data_controller.features[train_df.index]
    num_features = X_train.shape[1]
    X_train = X_train.astype(np.float32)
    scaler = StandardScaler(with_mean=normalize_mean, with_std=normalize_std)
    X_train = scaler.fit_transform(X_train)

    # Categorical encoding of labels
    le = LabelEncoder()
    y_train = le.fit_transform(train_df["brand_model"])
    num_classes = len(train_df["brand_model"].unique())

    # Set up test data
    num_test_images_available = test_df.groupby(["brand", "clean_model"])["filename"].count().min()
    num_test_images = min(num_test_images_available, 50)
    log.info(f"Randomly selecting {num_test_images} test images for each known camera model")
    sub_test_df = test_df.groupby(["brand", "clean_model"]).sample(num_test_images, random_state=other_rng)

    #
    # Select unknown camera models
    #
    X_test = data_controller.features[sub_test_df.index]
    X_test = X_test.astype(np.float32)
    X_test = scaler.transform(X_test)
    y_test = le.transform(sub_test_df["brand_model"])

    # Select random images from other cameras
    unknown_dresden_df = data_df[~data_df.index.isin(known_dresden_df.index)]
    assert len(unknown_dresden_df) + len(known_dresden_df) == len(data_df)

    # Select unknown motives only
    unknown_dresden_df = unknown_dresden_df[~unknown_dresden_df["motive_name"].isin(train_motive_names)]
    assert len(set(unknown_dresden_df["motive_name"]).intersection(train_instances_motives_imgs_df["motive_name"])) == 0, "Motive overlap"

    # Group by clean model
    unknown_dresden_grouped = unknown_dresden_df.groupby(["brand_model"])
    num_ood_images_per_camera_model_available = unknown_dresden_grouped.count()["filename"].min()
    ood_df = unknown_dresden_grouped.sample(num_ood_images_per_camera_model_available, random_state=other_rng)

    # All camera models should now be part of either the training or the out-of-distribution set
    assert set(data_df["brand_model"]) == (set(train_df["brand_model"]).union(set(ood_df["brand_model"])))

    X_ood = data_controller.features[ood_df.index]
    X_ood = X_ood.astype(np.float32)
    X_ood = scaler.transform(X_ood)
    y_ood = -1 * np.ones(len(X_ood))

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "X_ood": X_ood,
        "y_ood": y_ood,
        "num_classes": num_classes,
        "num_features": num_features,
        "train_df": train_df,
        "test_df": sub_test_df,
        "ood_df": ood_df,
        "scaler": scaler,
    }
