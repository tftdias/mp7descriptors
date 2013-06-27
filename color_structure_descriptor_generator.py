__author__ = 'Tiago Dias'


import sys
import getopt
import os.path
import numpy as np
import cv2
from cv2 import cv

import hmmd_quantifiers_generators as hmmd


def subsample_image(image):
    H, W = image.shape[0:2]
    p = max(0, np.floor(np.log2(np.sqrt(W*H)) - 7.5))
    K = np.power(2, p)

    new_W = np.int(W / K)
    new_H = np.int(H / K)

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
    hist = np.zeros(M, int)

    for i in range(image.shape[0]-8):
        for j in range(image.shape[1]-8):
            struct_elem = image[i:i+8, j:j+8]
            hist[np.unique(struct_elem)] += 1
    return hist


def normalize_histogram(hist):
    total = np.sum(hist)
    norm_hist = hist / np.float(total)
    return norm_hist


def quantify_histogram(hist):
    quant_hist = np.zeros(len(hist), dtype=np.uint8)
    # quant_hist[(hist >= 0) & (hist < 0.000000001)] = 0
    quant_hist[(hist >= 0.000000001) & (hist < 0.037)] = ((hist[(hist >= 0.000000001) & (hist < 0.037)] - 0.000000001) / ((0.037-0.000000001) / 25)).astype(int) + 1
    quant_hist[(hist >= 0.037) & (hist < 0.08)] = ((hist[(hist >= 0.037) & (hist < 0.08)] - 0.037) / ((0.08-0.037) / 20)).astype(int) + 26
    quant_hist[(hist >= 0.08) & (hist < 0.195)] = ((hist[(hist >= 0.08) & (hist < 0.195)] - 0.08) / ((0.195-0.08) / 35)).astype(int) + 46
    quant_hist[(hist >= 0.195) & (hist < 0.32)] = ((hist[(hist >= 0.195) & (hist < 0.32)] - 0.195) / ((0.32-0.195) / 35)).astype(int) + 81
    quant_hist[(hist >= 0.32) & (hist < 1.0)] = ((hist[(hist >= 0.32) & (hist < 1.0)] - 0.32) / ((1.0-0.32) / 140)).astype(int) + 116
    return quant_hist


def generate_color_structure_descriptor(image, M=256):
    # image subsampling
    image = subsample_image(image)
    # RGB to HMMD conversion
    image = rgb_to_hmmd(image)
    # quantify image
    quantifiers = {32: quantify_image_32,
                   64: quantify_image_64,
                   128: quantify_image_128,
                   256: quantify_image_256
    }
    image = quantifiers[M](image)
    # determine structured histogram
    hist = get_structured_histogram(image, M)
    # normalize histogram
    hist = normalize_histogram(hist)
    # quantify histogram bean values
    hist = quantify_histogram(hist)
    return hist


def write_file_descriptor(descriptor, filename):
    np.savetxt(filename, descriptor, fmt='%d', delimiter='\t')
    return True


if __name__ == '__main__':

    # obtain image filename
    if len(sys.argv) < 2:
        print 'Please, specify image filename.'
        sys.exit(0)
    image_filename = sys.argv[1]

    # obtain destination description filename
    if len(sys.argv) < 3:
        print 'Please, specify destination descriptor filename.'
        sys.exit(0)
    dest_filename = sys.argv[2]

    # obtain descriptor dimension
    M = 256
    opts, extraparams = getopt.getopt(sys.argv[1:], 'mM')
    for o, v in opts:
        if o in ['-M', '-m']:
            try:
                M = int(v)
            except:
                print 'Invalid dimension value.'
                sys.exit(0)

    # verify if file exists
    if not os.path.isfile(image_filename):
        print 'Invalid filename. Please, verify file name/location.'
        sys.exit(0)

    # read image file
    image = cv2.imread(image_filename)

    # generate descriptor
    descriptor = generate_color_structure_descriptor(image)

    # write descriptor to file
    if write_file_descriptor(descriptor, dest_filename):
        print 'Color structure descriptor from ' + image_filename + ' saved to ' + dest_filename + ' successfuly.'