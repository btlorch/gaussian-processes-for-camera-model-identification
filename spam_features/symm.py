import numpy as np


def symm_single(f, T, order):
    """
    Marginaliztion by sign and directional symmetry
    :param f: 1-D array of shape (2T + 1) ** order
    :param T: threshold
    :param order: co-occurrence order
    :return: 1-D array with reduced feature dimensionality
    """

    num_features = len(f)
    B = 2 * T + 1
    c = B ** order
    assert num_features == c

    f = np.reshape(f, [B] * order)
    return symm(f, T=T, order=order)


def symm_batch(f, T, order):
    """
    Marginalization by sign and directional symmetry for a feature vector stored as as (2T + 1)^order-dimensional array.
    The input feature f is assumed to be [num_samples, num_features] matrix of features stored as rows.
    :param f: ndarray of shape [num_samples, num_features]
    :param T:
    :param order:
    :return:
    """

    num_samples, num_features = f.shape
    B = 2 * T + 1
    c = B ** order
    assert num_features == c

    # Reduced dimensionality for a c-dimensional feature vector
    if 1 == order:
        num_reduced_features = T + 1
    elif 2 == order:
        num_reduced_features = (T + 1) ** 2
    elif 3 == order:
        num_reduced_features = 1 + 3 * T + 4 * T ** 2 + 2 * T ** 3
    elif 4 == order:
        num_reduced_features = B ** 2 + 4 * T ** 2 * (T + 1) ** 2
    elif 5 == order:
        num_reduced_features = 1/4 * (B ** 2 + 1) * (B ** 3 + 1)

    fsym = np.zeros((num_samples, num_reduced_features))

    for n in range(num_samples):
        if 1 == order:
            cube = f[n]
        elif 2 == order:
            cube = np.reshape(f[n], [B, B])
        elif 3 == order:
            cube = np.reshape(f[n], [B, B, B])
        elif 4 == order:
            cube = np.reshape(f[n], [B, B, B, B])
        elif 5 == order:
            cube = np.reshape(f[n], [B, B, B, B, B])

        fsym[n] = symm(cube, T, order)

    return fsym


def symm(A, T, order):
    """
    Symmetry marginalization routine. The purpose is to reduce the feature dimensionality and make the features more populated.
    It can be applied to 1D -- 5D co-occurrence matrices (order \in {1,2,3,4,5}) with sign and directional symmetries (explained below).

    Marginalization by symmetry pertains to the fact that, fundamentally, the differences between consecutive pixels in a natural image (both cover and stego) d1, d2, d3, ..., have the same probability of occurrence as the triple -d1, -d2, -d3, ...
    Directional marginalization pertains to the fact that the differences d1, d2, d3, ... in a natural (cover and stego) image are as likely to occur as ..., d3, d2, d1.
    :param A: ndarray of shape (2T + 1)^order
    :param T: threshold
    :param order: co-occurrence order
    :return: symmetrized 1-D array of shape [num_ouput_features]
    """

    # Skip index 0, where the origin is stored
    next_idx = 1
    B = 2 * T + 1

    if 1 == order:
        assert np.prod(A.shape) == 2 * T + 1
        As = np.zeros(T + 1)
        # The only non-marginalized bin is the origin 0
        As[0] = A[T]
        As[1:] = A[:T] + A[T + 1:]
        return As

    elif 2 == order:
        assert np.prod(A.shape) == (2 * T + 1) ** 2
        As = np.zeros((T + 1) ** 2)
        # The only non-marginalized bin is the origin [0, 0]
        As[0] = A[T, T]

        done = np.zeros_like(A, dtype=np.bool)
        for i in range(-T, T + 1):
            for j in range(-T, T + 1):
                if (not done[i + T, j + T]) and (abs(i) + abs(j) != 0):
                    As[next_idx] = A[i + T, j + T] + A[T - i, T - j]
                    done[i + T, j + T] = True
                    done[T - i, T - j] = True
                    # Flip j and i indices
                    if (i != j) and (not done[j + T, i + T]):
                        As[next_idx] += A[j + T, i + T] + A[T - j, T, - i]
                        done[j + T, i + T] = True
                        done[T - j, T - i] = True

                    next_idx += 1

        return As

    elif 3 == order:
        assert np.prod(A.shape) == B ** 3
        As = np.zeros(1 + 3 * T + 4 * T**2 + 2 * T**3)
        # The only non-marginalized bin is the origin [0, 0, 0]
        As[0] = A[T, T, T]

        done = np.zeros_like(A, dtype=np.bool)
        for i in range(-T, T + 1):
            for j in range(-T, T + 1):
                for k in range(-T, T + 1):
                    if (not done[i + T, j + T, k + T]) and (abs(i) + abs(j) + abs(k) != 0):
                        As[next_idx] = A[i + T, j + T, k + T] + A[T - i, T - j, T - k]
                        done[i + T, j + T, k + T] = True
                        done[T - i, T - j, T - k] = True
                        if (i != k) and (not done[k + T, j + T, i + T]):
                            As[next_idx] += A[k + T, j + T, i + T] + A[T - k, T - j, T - i]
                            done[k + T, j + T, i + T] = True
                            done[T - k, T - j, T - i] = True

                        next_idx += 1

        return As

    elif 4 == order:
        assert np.prod(A.shape) == B ** 4
        As = np.zeros(B ** 2 + 4 * T ** 2 * (T + 1) ** 2)
        # The only non-marginalized bin is the origin [0, 0, 0]
        As[0] = A[T, T, T, T]

        done = np.zeros_like(A, dtype=np.bool)
        for i in range(-T, T + 1):
            for j in range(-T, T + 1):
                for k in range(-T, T + 1):
                    for n in range(-T, T + 1):
                        if (not done[i + T, j + T, k + T, n + T]) and (abs(i) + abs(j) + abs(k) + abs(n) != 0):
                            As[next_idx] = A[i + T, j + T, k + T, n + T] + A[T - i, T - j, T - k, T - n]
                            done[i + T, j + T, k + T, n + T] = True
                            done[T - i, T - j, T - k, T - n] = True
                            if ((i != n) or (j != k)) and (not done[n + T, k + T, j + T, i + T]):
                                As[next_idx] += A[n + T, k + T, j + T, i + T] + A[T - n, T - k, T - j, T - i]
                                done[n + T, k + T, j + T, i + T] = True
                                done[T - n, T - k, T - j, T - i] = True

                            next_idx += 1

        return As
