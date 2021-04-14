"""
Implements metrics from P. Mendes Junior et al. "In-Depth Study on Open-Set Camera Model Identification"
"""
import numpy as np
from sklearn.metrics import accuracy_score, roc_curve


def accuracy_known_samples(y_true, y_pred):
    """
    Accuracy in correctly attributing images from known models to the actual model.
    This metric encompasses two kinds of misclassification errors:
    - known-model images attributed to the unknown class (false unknown)
    - known-model images attributed to wrong known classes (misclassification)

    :param y_true: ground truth labels. Use value -1 for unknown models
    :param y_pred: predicted labels. Unrecognized models are encoded as the value -1.
    :return: accuracy on known samples, or None if there are no known samples
    """
    # Only consider known camera models
    mask = y_true >= 0
    if not np.any(mask):
        return None

    return accuracy_score(y_true[mask], y_pred[mask])


def accuracy_unknown_samples(y_true, y_pred):
    """
    Accuracy in correctly classifying as unknown the images from unknown camera models
    :param y_true: ground truth labels. Use value -1 for unknown models
    :param y_pred: predicted labels. Unrecognized models are encoded as the value -1.
    :return: accuracy on unknown samples, or None if there are no unknown samples
    """

    # Only consider unknown camera models
    mask = y_true < 0
    if not np.any(mask):
        return None

    return accuracy_score(y_true[mask], y_pred[mask])


def normalized_accuracy(y_true, y_pred):
    """
    Average between accuracy on known samples and accuracy on unknown samples.
    Provides an overall view of a classifier performance in terms of both open- and closed-set scenarios.
    :param y_true: ground truth labels. Use value -1 for unknown models.
    :param y_pred: predicted labels. Unrecognized models are encoded as the value -1.
    :return: average between accuracy on known samples and accuracy on unknown samples
    """

    acc_known_samples = accuracy_known_samples(y_true, y_pred)
    acc_unknown_samples = accuracy_unknown_samples(y_true, y_pred)

    if acc_known_samples is None:
        return acc_unknown_samples

    if acc_unknown_samples is None:
        return acc_known_samples

    return (acc_known_samples + acc_unknown_samples) / 2.


def normalized_accuracy_with_prob(y_true, y_pred, y_prob):
    """
    Average between accuracy on known samples and accuracy on unknown samples.
    Compared to `normalized_accuracy()`, this method uses posterior probabilities on a continuous scale.

    :param y_true: ground truth labels. Use value -1 for unknown models.
    :param y_pred: predicted labels
    :param y_prob: posterior probability for the corresponding predicted label
    :return: average between accuracy on known samples and accuracy on unknown samples
    """
    acc_known_samples = accuracy_known_samples(y_true, y_pred)

    # How to equally balance accuracy and AUC?
    # 1) Compute the ROC curve
    # 2) Pick the threshold where the curve is closest to the top-left corner
    # 3) Binarize continuous predictions (are they above or below the threshold). Unknown classes are assigned the label -1.

    # Set labels to {-1, 1}
    labels = np.copy(y_true)
    labels[labels >= 0] = 1
    fpr, tpr, thresholds = roc_curve(labels, y_prob)

    # Pick the threshold closest to the top left corner
    distances_to_top_left_corner = np.sqrt((1 - tpr) ** 2 + fpr ** 2)
    threshold = thresholds[np.argmin(distances_to_top_left_corner)]

    # Compute the accuracy based on this threshold
    # Rejected samples are assigned the label - 1
    y_prob_binary = (y_prob >= threshold).astype(np.int) - 1
    acc_unknown_samples = accuracy_unknown_samples(y_true, y_prob_binary)

    if acc_known_samples is None:
        return acc_unknown_samples

    if acc_unknown_samples is None:
        return acc_known_samples

    return (acc_known_samples + acc_unknown_samples) / 2.
