import numpy as np


def cooc(residuals, order, type, T):
    """
    Co-occurrence operator to be applied to a 2D array of residuals D in [-T, T]
    :param residuals: residuals in range [-T, T]
    :param order: co-occurrence order in {1, 2, 3, 4, 5}
    :param type: co-occurrence type in {"horizontal", "vertical", "diag"}
    :param T: threshold
    :return: array of shape (2T + 1)^order
    """

    B = 2 * T + 1
    assert np.all((residuals <= T) & (residuals >= -T)), "Residual out of range"
    assert type in {"horizontal", "vertical", "diag"}

    if 1 == order:
        # Bin edges includes the right outer edge
        bin_edges = np.arange(-T, T + 2)
        f = np.histogram(residuals.flatten(), bins=bin_edges)[0]

    elif 2 == order:
        f = np.zeros((B, B))
        if "horizontal" == type:
            L = residuals[:, :-1]
            R = residuals[:, 1:]
        elif "vertical" == type:
            L = residuals[:-1, :]
            R = residuals[1:, :]
        elif "diag" == type:
            L = residuals[:-1, :-1]
            R = residuals[1:, 1:]

        for i in range(-T, T + 1):
            R2 = R[L == i]
            for j in range(-T, T + 1):
                f[i + T, j + T] = np.sum(R2 == j)

    elif 3 == order:
        f = np.zeros((B, B, B))
        if "horizontal" == type:
            L = residuals[:, :-2]
            C = residuals[:, 1:-1]
            R = residuals[:, 2:]
        elif "vertical" == type:
            L = residuals[:-2, :]
            C = residuals[1:-1, :]
            R = residuals[2:, :]
        elif "diag" == type:
            L = residuals[:-2, :-2]
            C = residuals[1:-1, 1:-1]
            R = residuals[2:, 2:]

        for i in range(-T, T + 1):
            C2 = C[L == i]
            R2 = R[L == i]
            for j in range(-T, T + 1):
                R3 = R2[C2 == j]
                for k in range(-T, T + 1):
                    f[i + T, j + T, k + T] = np.sum(R3 == k)

    elif 4 == order:
        f = np.zeros((B, B, B, B))
        if "horizontal" == type:
            L = residuals[:, :-3]
            C = residuals[:, 1:-2]
            E = residuals[:, 2:-1]
            R = residuals[:, 3:]
        elif "vertical" == type:
            L = residuals[:-3, :]
            C = residuals[1:-2, :]
            E = residuals[2:-1, :]
            R = residuals[3:, :]
        elif "diag" == type:
            L = residuals[:-3, :-3]
            C = residuals[1:-2, 1:-2]
            E = residuals[2:-1, 2:-1]
            R = residuals[3:, 3:]

        for i in range(-T, T + 1):
            ind = L == i
            C2 = C[ind]
            E2 = E[ind]
            R2 = R[ind]

            for j in range(-T, T + 1):
                ind = C2 == j
                E3 = E2[ind]
                R3 = R2[ind]

                for k in range(-T, T + 1):
                    R4 = R3[E3 == k]

                    for l in range(-T, T + 1):
                        f[i + T, j + T, k + T, l + T] = np.sum(R4 == l)

    else:
        raise ValueError("Not implemented yet. See Matlab implementation")

    # Normalization
    f_sum = np.sum(f)
    if f_sum > 0:
        f = f / f_sum

    return f
