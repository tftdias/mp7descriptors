__author__ = 'Tiago Dias'


import numpy as np
import cv2
from cv2 import cv

import hmmd_quantifiers_generators as hmmd


def subsample_image(image):
    H, W = image.shape[0:2]
    p = max(0, np.floor(np.log2(np.sqrt(W*H) - 7.5)))
    K = np.power(2, p)

    new_W = W / K
    new_H = H / K

    return cv2.resize(image, (new_H, new_W))


def rgb_to_hmmd(image):
    Max = np.maximum(image[..., 0], np.maximum(image[..., 1], image[..., 2]))
    Min = np.minimum(image[..., 0], np.minimum(image[..., 1], image[..., 2]))

    Diff = Max - Min
    Sum = (Max+Min) / 2
    Hue = (cv2.cvtColor(image, cv.CV_BGR2HSV)[..., 0] / 180.0 * 255).astype(int)

    return np.array([Hue, Diff, Sum])


def quantify_image_256(image):
    quantifier = hmmd.get_hmmd_quantifier_256()
    return quantifier[image[0, ...], image[1, ...], image[2, ...]]


def quantify_image_128(image):
    quantifier = hmmd.get_hmmd_quantifier_128()
    return quantifier[image[0, ...], image[1, ...], image[2, ...]]


def quantify_image_64(image):
    quantifier = hmmd.get_hmmd_quantifier_64()
    return quantifier[image[0, ...], image[1, ...], image[2, ...]]


def quantify_image_32(image):
    quantifier = hmmd.get_hmmd_quantifier_32()
    return quantifier[image[0, ...], image[1, ...], image[2, ...]]


def get_structured_histogram(image, M=256):
    hist = np.zeros(256, int)

    for i in range(image.shape[0]-8):
        for j in range(image.shape[1]-8):
            struct_elem = image[i:i+8, j:j+8]
            hist[np.unique(struct_elem)] += 1
    return hist


def unify_histogram_beans(hist, M):
    if len(hist) == 256 and M < 256:
        factor = 256 / M
        hist = np.sum(hist.reshape([M, factor]), axis=1)
    return hist


def normalize_histogram(hist):
    total = np.sum(hist)
    norm_hist = hist / np.float(total)
    return norm_hist


def quantify_histogram(hist):
    quant_hist = np.zeros(len(hist))
    # quant_hist[hist >= 0 and hist < 0.000000001] = 0
    quant_hist[hist >= 0.000000001 and hist < 0.037] = 1
    quant_hist[hist >= 0.037 and hist < 0.08] = 2
    quant_hist[hist >= 0.08 and hist < 0.195] = 3
    quant_hist[hist >= 0.195 and hist < 0.32] = 4
    quant_hist[hist >= 0.32 and hist < 1.0] = 5
    return quant_hist