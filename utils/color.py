import numpy as np


def color2gray(img):
    img = img.astype(np.float32)
    coeffs = np.array([0.299, 1 - 0.299 - 0.114, 0.114], dtype=np.float32)
    return img @ coeffs
