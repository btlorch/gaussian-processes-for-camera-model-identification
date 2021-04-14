import numpy as np


def residual(X, order, type):
    """
    Computes the noise residual of a given type and order from MxN image X
    !!!!!!!!!!!!! Use order = 2 with KB and all edge residuals !!!!!!!!!!!!!

    :param X: ndarray of shape M x N
    :param order: residual order in {1, 2, 3, 4, 5, 6}
    :param type: type in {"horizontal", "vertical", "diag"}
    :return: The resulting residual is an (M - b) x (N - b) array of the specified order, where b = ceil(order / 2). This cropping is a little more than it needs to be to make sure all the residuals are easily "synchronized".

    """
    M, N = X.shape

    y_start = int(np.ceil(order / 2))
    y_stop = M - int(np.ceil(order / 2))

    x_start = int(np.ceil(order / 2))
    x_stop = N - int(np.ceil(order / 2))

    # Set up slices
    sy_m3 = slice(y_start - 3, y_stop - 3)
    sy_m2 = slice(y_start - 2, y_stop - 2)
    sy_m1 = slice(y_start - 1, y_stop - 1)
    sy    = slice(y_start    , y_stop    )
    sy_p1 = slice(y_start + 1, y_stop + 1)
    sy_p2 = slice(y_start + 2, y_stop + 2)
    sy_p3 = slice(y_start + 3, y_stop + 3)

    sx_m3 = slice(x_start - 3, x_stop - 3)
    sx_m2 = slice(x_start - 2, x_stop - 2)
    sx_m1 = slice(x_start - 1, x_stop - 1)
    sx    = slice(x_start    , x_stop    )
    sx_p1 = slice(x_start + 1, x_stop + 1)
    sx_p2 = slice(x_start + 2, x_stop + 2)
    sx_p3 = slice(x_start + 3, x_stop + 3)

    if "horizontal" == type:
        if 1 == order:
            return -X[sy, sx] + X[sy, sx_p1]
        elif 2 == order:
            return X[sy, sx_m1] - 2 * X[sy, sx] + X[sy, sx_p1]
        elif 3 == order:
            return X[sy, sx_m1] - 3 * X[sy, sx] + 3 * X[sy, sx_p1] - X[sy, sx_p2]
        elif 4 == order:
            return -X[sy, sx_m2] + 4 * X[sy, sx_m1] - 6 * X[sy, sx] + 4 * X[sy, sx_p1] - X[sy, sx_p2]
        elif 5 == order:
            return -X[sy, sx_m2] + 5 * X[sy, sx_m1] - 10 * X[sy, sx] + 10 * X[sy, sx_p1] - 5 * X[sy, sx_p2] + X[sy, sx_p3]
        elif 6 == order:
            return X[sy, sx_m3] - 6 * X[sy, sx_m2] + 15 * X[sy, sx_m1] - 20 * X[sy, sx] + 15 * X[sy, sx_p1] - 6 * X[sy, sx_p2] + X[sy, sx_p3]

    elif "vertical" == type:
        if 1 == order:
            return - X[sy, sx] + X[sy_p1, sx]
        elif 2 == order:
            return X[sy_m1, sx] - 2 * X[sy, sx] + X[sy_p1, sx]
        elif 3 == order:
            return X[sy_m1, sx] - 3 * X[sy, sx] + 3 * X[sy_p1, sx] - X[sy_p2, sx]
        elif 4 == order:
            return -X[sy_m2, sx] + 4 * X[sy_m1, sx] - 6 * X[sy, sx] + 4 * X[sy_p1, sx] - X[sy_p2, sx]
        elif 5 == order:
            return -X[sy_m2, sx] + 5 * X[sy_m1, sx] - 10 * X[sy, sx] + 10 * X[sy_p1, sx] - 5 * X[sy_p2, sx] + X[sy_p3, sx]
        elif 6 == order:
            return X[sy_m3, sx] - 6 * X[sy_m2, sx] + 15 * X[sy_m1, sx] - 20 * X[sy, sx] + 15 * X[sy_p1, sx] - 6 * X[sy_p2, sx] + X[sy_p3, sx]

    elif "diag" == type:
        if 1 == order:
            return - X[sy, sx] + X[sy_p1, sx_p1]
        elif 2 == order:
            return X[sy_m1, sx_m1] - 2 * X[sy, sx] + X[sy_p1, sx_p1]
        elif 3 == order:
            return X[sy_m1, sx_m1] - 3 * X[sy, sx] + 3 * X[sy_p1, sx_p1] - X[sy_p2, sx_p2]
        elif 4 == order:
            return -X[sy_m2, sx_m2] + 4 * X[sy_m1, sy_m1] - 6 * X[sy, sx] + 4 * X[sy_p1, sx_p1] - X[sy_p2, sx_p2]
        elif 5 == order:
            return -X[sy_m2, sx_m2] + 5 * X[sy_m1, sx_m1] - 10 * X[sy, sx] + 10 * X[sy_p1, sx_p1] - 5 * X[sy_p2, sx_p2] + X[sy_p3, sx_p3]
        elif 6 == order:
            return X[sy_m3, sx_m3] - 6 * X[sy_m2, sx_m2] + 15 * X[sy_m1, sx_m1] - 20 * X[sy, sx] + 15 * X[sy_p1, sx_p1] - 6 * X[sy_p2, sx_p2] + X[sy_p3, sx_p3]

    else:
        raise ValueError("Not supported. Look up official Matlab implementation.")
