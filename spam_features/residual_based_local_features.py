import numpy as np
from spam_features.residual import residual
from spam_features.cooccurrence import cooc
from spam_features.symm import symm


def quantize_truncate(X, q, T):
    """
    Quantization routine
    :param X: variable to be quantized and truncated
    :param q: scalar quantization step
    :param T: threshold
    :return:
    """

    matlab_round = lambda x: np.sign(x) * np.floor(np.abs(x) + 0.5)

    # Round and truncate to [-T, T]
    return np.maximum(-T, np.minimum(T, matlab_round(X / q)))


def residual_based_local_features(img, q=1, T=2, cooc_order=4):

    residual_order = 3
    residual_factor = 3 # This is the largest coefficient of the filter kernel
    residual_h = residual(img, order=residual_order, type="horizontal")
    residual_v = residual(img, order=residual_order, type="vertical")

    residual_h_quantized = quantize_truncate(residual_h, q=q * residual_factor, T=T)
    residual_v_quantized = quantize_truncate(residual_v, q=q * residual_factor, T=T)

    # Co-occurrences
    spam14h_1 = cooc(residuals=residual_h_quantized, order=cooc_order, type="horizontal", T=T)
    spam14h_2 = cooc(residuals=residual_v_quantized, order=cooc_order, type="vertical", T=T)
    spam14v_1 = cooc(residuals=residual_h_quantized, order=cooc_order, type="vertical", T=T)
    spam14v_2 = cooc(residuals=residual_v_quantized, order=cooc_order, type="horizontal", T=T)

    # Reduce by symmetry
    spam14h_1 = symm(spam14h_1, T=T, order=cooc_order)
    spam14h_2 = symm(spam14h_2, T=T, order=cooc_order)
    spam14v_1 = symm(spam14v_1, T=T, order=cooc_order)
    spam14v_2 = symm(spam14v_2, T=T, order=cooc_order)

    spam14h = spam14h_1 + spam14h_2
    spam14v = spam14v_1 + spam14v_2

    return np.concatenate([spam14h, spam14v])
