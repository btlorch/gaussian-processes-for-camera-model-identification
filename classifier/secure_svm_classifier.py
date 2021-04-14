from sklearn.svm import SVC, OneClassSVM
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator


class SecureSVMClassifier(BaseEstimator):
    """
    Combined classification framework (CCF) proposed in [1].

    [1] Wang et al. "Source Camera Identification Using Support Vector Machines", International Conference on Digital Forensics, 2009. https://link.springer.com/chapter/10.1007/978-3-642-04155-6_8
    """

    def __init__(self, C=1.0, nu=0.5, scale_mean=False, scale_std=False):
        """
        Initialize the secure SVM classifier
        :param C: Regularization hyper-parameter of the multi-class SVM (default = 1.0)
        :param nu: Hyper-parameter for all one-class SVMs. An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors. Should be in the interval (0, 1]. (Default = 0.5)
        :param scale_mean: whether to normalize the data to zero-mean
        :param scale_std: whether to normalize the data to unit-variance
        """
        self.C = C
        self.nu = nu
        self.scale_mean = scale_mean
        self.scale_std = scale_std

        self.preprocessors = dict()
        self.anomaly_svms = dict()
        self.classes = None

        # Preserve training data to create new multi-class models on-the-fly
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Fit individual one-class SVM to each class
        :param X: ndarray of shape [num_samples, num_features]
        :param y: targets of shape [num_samples]. Classes do not need need to be labeled from 0 through C-1, but can take on any non-negative integers.
        """
        assert y.dtype == np.int and np.all(y >= 0), "Class labels must be non-negative integers"
        self.classes = np.sort(np.unique(y))

        # Fit one-class SVMs for each class
        for c in self.classes:
            mask = y == c
            X_class = X[mask]
            scaler = StandardScaler(with_mean=self.scale_mean, with_std=self.scale_std)
            X_class_transformed = scaler.fit_transform(X_class)

            clf = OneClassSVM(kernel="rbf", nu=self.nu)
            clf.fit(X_class_transformed)

            self.preprocessors[c] = scaler
            self.anomaly_svms[c] = clf

        # Preserve training data
        self.X_train = X
        self.y_train = y

    def _new_multiclass_svm_predict(self, classes, X_test):
        mask = np.isin(self.y_train, classes)
        X_relevant = self.X_train[mask]
        # No need to map these class labels to an increasing list of classes starting from 0
        y_relevant = self.y_train[mask]

        scaler = StandardScaler(with_mean=self.scale_mean, with_std=self.scale_std)
        X_relevant_transformed = scaler.fit_transform(X_relevant)
        clf = SVC(kernel="rbf", C=self.C)
        clf.fit(X_relevant_transformed, y_relevant)

        X_test_transformed = scaler.transform(X_test)
        return clf.predict(X_test_transformed)

    def predict(self, X, return_decision_function=False):
        """
        There are three possible one-class SVM outputs for a test image:
        (1) Outlier: Return -1
        (2) One positive result: Return the corresponding class label.
        (3) Multiple positive results: Train new SVM one the fly to distinguish between the positive classes.
        :param X: ndarray of shape [num_samples, num_features]
        :param return_decision_function: If True, return the maximum (over the C one-class SVMs) signed distance to the separating hyper-plane
        :return: ndarray of shape [num_samples] with the corresponding sparse class label. Outliers use the value -1.
            If return_decision_function is True, additionally return an ndarray of shape [num_samples] with the maximum signed distance to the separating hyper-plane
        """

        # Initialize predictions with dummy value -2 corresponding to an uninitialized value
        y_pred = -2 * np.ones(len(X), dtype=np.int)

        # Feed examples through one-class SVMs
        oc_preds = []
        oc_decision_functions = []
        for c in self.classes:
            X_transformed = self.preprocessors[c].transform(X)
            oc_preds.append(self.anomaly_svms[c].predict(X_transformed))
            oc_decision_functions.append((self.anomaly_svms[c].decision_function(X_transformed)))

        # Stack predictions of one-class SVMs
        oc_preds = np.stack(oc_preds, axis=1)
        oc_decision_functions = np.stack(oc_decision_functions, axis=1)

        # For each example we can distinguish between three cases
        # (1) Outlier
        is_outlier_mask = np.all(oc_preds == -1, axis=1)
        y_pred[is_outlier_mask] = -1

        # (2) One positive result
        single_positive_mask = np.sum(oc_preds == 1, axis=1) == 1
        y_pred[single_positive_mask] = self.classes[np.where(oc_preds[single_positive_mask] == 1)[1]]

        # (3) Multiple positive results
        # In these cases, we need to train a new multi-class model on-the-fly for the positive classes
        multiple_positives_mask = np.sum(oc_preds == 1, axis=1) > 1
        if not np.any(multiple_positives_mask):
            return y_pred

        # Convert mask to indices
        multiple_positives_indices = np.where(multiple_positives_mask)[0]

        # Collect all combinations of classifiers that we need to create
        # Interpret `oc_preds` as a boolean array, where each column corresponds to a power of 2
        # Convert to boolean array
        multiple_positives_bits = oc_preds[multiple_positives_mask] == 1
        # Create array with powers of 2, i.e., [2 ** (num_classes - 1), ..., 2 ** 1, 2 ** 0]
        powers = 2 ** np.arange(0, len(self.classes))[::-1]
        # Convert boolean array to a number
        all_required_clfs_numbers = np.sum(powers[np.newaxis, :] * multiple_positives_bits, axis=1)
        # Filter out unique classifiers
        unique_required_clfs_numbers = np.unique(all_required_clfs_numbers)

        # Loop over the classifiers that we need to create
        for clf_number in unique_required_clfs_numbers:
            # Get all examples that were recognized by this specific combination of one-class SVMs
            oc_combo_mask = all_required_clfs_numbers == clf_number
            # Obtain their indices (within the multiple positives)
            sample_indices = np.where(oc_combo_mask)[0]
            assert len(sample_indices) > 0, "There must be at least one such example"
            # Transform the set of SVMs that recognized the example into their respective class labels
            required_classes = self.classes[multiple_positives_bits[sample_indices[0]]]
            # Train a multi-class SVM for the selected classes, and evaluate the multi-class SVM on the specific test examples
            mc_pred = self._new_multiclass_svm_predict(required_classes, X[multiple_positives_indices[oc_combo_mask]])
            # Copy the predicted class labels over to the final output
            y_pred[multiple_positives_indices[oc_combo_mask]] = mc_pred

        # Ensure that we left no value uninitialized
        assert not np.any(-2 == y_pred)

        if return_decision_function:
            return y_pred, np.max(oc_decision_functions, axis=1)

        return y_pred

    def score(self, X, y, sample_weight=None):
        """
        Return the mean accuracy on the given test data and labels
        :param X: test samples of shape [num_samples, num_features]
        :param y: true labels for X, of shape [num_samples]
        :param sample_weight: array-like of shape [num_samples], default=None
        :return: mean accuracy of ``self.predict(X)`` w.r.t. ``y`.
        """
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
