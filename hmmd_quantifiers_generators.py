# __author__ = 'Tiago Dias'


from __future__ import division
import numpy as np
import pickle

import settings as sett


# max component values
HUE_VALUES = 256
DIFF_VALUES = 256
SUM_VALUES = 256

# subspaces indexes
SUBSPACE_0 = range(6)
SUBSPACE_1 = range(6, 20)
SUBSPACE_2 = range(20, 60)
SUBSPACE_3 = range(60, 110)
SUBSPACE_4 = range(110, 256)


def generate_hmmd_quantifier(hue_levels, sum_levels):
    subspaces_ind = [SUBSPACE_0, SUBSPACE_1, SUBSPACE_2, SUBSPACE_3, SUBSPACE_4]

    # creating new quantification matrix for 256 colors
    hmmd_quantified_values = np.zeros([HUE_VALUES, DIFF_VALUES, SUM_VALUES, 3], dtype=np.uint8)

    for subspace in range(5):
        # Hue levels
        ind_max = 0
        for level in range(hue_levels[subspace]):
            ind_min = ind_max
            ind_max += HUE_VALUES / hue_levels[subspace]
            hmmd_quantified_values[ind_min:ind_max, subspaces_ind[subspace], :, 0] = level

        # Diff levels
        hmmd_quantified_values[:, subspaces_ind[subspace], :, 1] += subspace

        # Sum levels
        ind_max = 0
        for level in range(sum_levels[subspace]):
            ind_min = ind_max
            ind_max += SUM_VALUES / sum_levels[subspace]
            hmmd_quantified_values[:, subspaces_ind[subspace], ind_min:ind_max, 2] = level

    hmmd_quantifier = np.zeros([HUE_VALUES, DIFF_VALUES, SUM_VALUES], dtype=np.uint8)

    for h_ind in range(HUE_VALUES):
        for d_ind in range(DIFF_VALUES):
            for s_ind in range(SUM_VALUES):
                hue = hmmd_quantified_values[h_ind, d_ind, s_ind, 0]
                diff = hmmd_quantified_values[h_ind, d_ind, s_ind, 1]
                sum = hmmd_quantified_values[h_ind, d_ind, s_ind, 2]
                hmmd_quantifier[h_ind, d_ind, s_ind] = np.sum(hue_levels[:diff] * sum_levels[:diff]) + \
                                                       hue * sum_levels[diff] + sum

    return hmmd_quantifier


def get_hmmd_quantifier_256():
    try:
        with open(sett.QUANTIFIER_256_FILENAME, 'rb') as f:
            return pickle.load(f)
    except IOError:
        hue_levels = np.array([1, 4, 16, 16, 16])
        sum_levels = np.array([32, 8, 4, 4, 4])
        return generate_hmmd_quantifier(hue_levels, sum_levels)


def get_hmmd_quantifier_128():
    try:
        with open(sett.QUANTIFIER_128_FILENAME, 'rb') as f:
            return pickle.load(f)
    except IOError:
        hue_levels = np.array([1, 4, 8, 8, 8])
        sum_levels = np.array([16, 4, 4, 4, 4])
        return generate_hmmd_quantifier(hue_levels, sum_levels)


def get_hmmd_quantifier_64():
    try:
        with open(sett.QUANTIFIER_64_FILENAME, 'rb') as f:
            return pickle.load(f)
    except IOError:
        hue_levels = np.array([1, 4, 4, 8, 8])
        sum_levels = np.array([8, 4, 4, 2, 1])
        return generate_hmmd_quantifier(hue_levels, sum_levels)


def get_hmmd_quantifier_32():
    try:
        with open(sett.QUANTIFIER_32_FILENAME, 'rb') as f:
            return pickle.load(f)
    except IOError:
        hue_levels = np.array([1, 4, 4, 4, 4])
        sum_levels = np.array([8, 4, 4, 1, 1])
        return generate_hmmd_quantifier(hue_levels, sum_levels)


def create_hmmd_quantifier_256_file():
    quantifier = get_hmmd_quantifier_256()
    return create_hmmd_quantifier_file(quantifier, sett.QUANTIFIER_256_FILENAME)


def create_hmmd_quantifier_128_file():
    quantifier = get_hmmd_quantifier_128()
    return create_hmmd_quantifier_file(quantifier, sett.QUANTIFIER_128_FILENAME)


def create_hmmd_quantifier_64_file():
    quantifier = get_hmmd_quantifier_64()
    return create_hmmd_quantifier_file(quantifier, sett.QUANTIFIER_64_FILENAME)


def create_hmmd_quantifier_32_file():
    quantifier = get_hmmd_quantifier_32()
    return create_hmmd_quantifier_file(quantifier, sett.QUANTIFIER_32_FILENAME)


def create_hmmd_quantifier_file(quantifier, QUANTIFIER_FILENAME):
    with open(QUANTIFIER_FILENAME, 'wb') as f:
        pickle.dump(quantifier, f)
    return True


def create_all_hmmd_quantifier_files():
    success = True

    print '%-*s' % (50, 'Generating HMMD quantifier for 256 colors...'),
    if create_hmmd_quantifier_256_file():
        print '[DONE]'
    else:
        success = False
        print '[FAIL]'

    print '%-*s' % (50, 'Generating HMMD quantifier for 128 colors...'),
    if create_hmmd_quantifier_128_file():
        print '[DONE]'
    else:
        success = False
        print '[FAIL]'

    print '%-*s' % (50, 'Generating HMMD quantifier for 64 colors...'),
    if create_hmmd_quantifier_64_file():
        print '[DONE]'
    else:
        success = False
        print '[FAIL]'

    print '%-*s' % (50, 'Generating HMMD quantifier for 32 colors...'),
    if create_hmmd_quantifier_32_file():
        print '[DONE]'
    else:
        success = False
        print '[FAIL]'

    return success


if __name__ == '__main__':
    create_all_hmmd_quantifier_files()