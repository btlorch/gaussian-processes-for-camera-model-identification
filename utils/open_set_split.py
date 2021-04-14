import numpy as np


class OpenSetShuffleSplit(object):
    def __init__(self, n_splits, random_state=np.random.RandomState(), train_percentage=0.5):
        self.n_splits = n_splits
        self.rng = random_state
        self.train_percentage = train_percentage

    def split(self, X, y):
        """
        Splits the given data into several training and test sets.

        Unknown samples are indicated by the target value -1.
        Note that the labels within a set may not be numbered from 0 through the number of classes in the set. Instead, they will be a subset of the possible classes in y.

        :param X: ndarray of shape [num_samples, num_features]
        :param y: targets of shape [num_samples]
        :return: generator that yields (X_train, y_train, X_test, y_test)
        """

        # Obtain set of class labels
        classes = np.unique(y)
        num_classes = len(classes)
        # 50 percent of these classes shall be known, the other 50 percent shall be unknown
        num_known_classes = int(np.ceil(num_classes / 2))
        num_unknown_classes = num_classes - num_known_classes

        # Generate n_splits splits into known and unknown classes
        known_classes_per_split = [self.rng.choice(classes, size=num_known_classes, replace=False) for i in range(self.n_splits)]

        # Iterate over splits
        for i in range(self.n_splits):
            # For the current split, determine the known and unknown classes
            known_classes = known_classes_per_split[i]
            unknown_classes = np.array(list(set(classes).difference(known_classes)))

            # Determine samples that belong to the known and unknown sets, respectively
            known_samples = np.isin(y, known_classes)
            unknown_samples = np.isin(y, unknown_classes)
            assert np.all(np.logical_xor(known_samples, unknown_samples))

            # Convert mask to indices
            known_indices = np.where(known_samples)[0]
            unknown_indices = np.where(unknown_samples)[0]

            # Split the known samples into train and test sets
            # Randomly permute known indices
            known_indices = self.rng.permutation(known_indices)
            # Determine how many samples go into the train set
            num_known_train_samples = int(np.ceil(len(known_indices) * self.train_percentage))
            known_train_indices = known_indices[:num_known_train_samples]
            known_test_indices = known_indices[num_known_train_samples:]

            # From the unknown classes, select the same number of images for the test set
            num_unknown_test_samples = len(known_test_indices)
            # From all unknown samples, randomly choose without replacement
            unknown_test_indices = self.rng.choice(unknown_indices, size=num_unknown_test_samples, replace=False)
            # The labels for unknown samples are -1.
            unknown_test_labels = -1 * np.ones(len(known_test_indices))

            assert len(set(known_train_indices).intersection(unknown_indices)) == 0, "Training and test set must not overlap"

            # Obtain training data
            X_train = X[known_train_indices]
            y_train = y[known_train_indices]

            # Concatenate test data from known and unknown classes
            test_indices = np.concatenate([known_test_indices, unknown_test_indices], axis=0)
            X_test = X[test_indices]
            y_test = np.concatenate([y[known_test_indices], unknown_test_labels], axis=0)

            # Return train and test sets
            yield X_train, y_train, X_test, y_test
