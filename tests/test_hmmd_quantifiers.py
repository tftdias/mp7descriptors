__author__ = 'Tiago'

import unittest
import pickle
import os

from hmmd_quantifiers_generators import *
import settings as sett


class TestHMMDQuantifiers(unittest.TestCase):

    def test_hmmd256_generation(self):
        quantifier = get_hmmd_quantifier_256()
        correct_size = (HUE_VALUES, DIFF_VALUES, SUM_VALUES, 3)
        self.assertEqual(quantifier.shape, correct_size)


    def test_hmmd128_generation(self):
        quantifier = get_hmmd_quantifier_128()
        correct_size = (HUE_VALUES, DIFF_VALUES, SUM_VALUES, 3)
        self.assertEqual(quantifier.shape, correct_size)


    def test_hmmd64_generation(self):
        quantifier = get_hmmd_quantifier_64()
        correct_size = (HUE_VALUES, DIFF_VALUES, SUM_VALUES, 3)
        self.assertEqual(quantifier.shape, correct_size)


    def test_hmmd32_generation(self):
        quantifier = get_hmmd_quantifier_32()
        correct_size = (HUE_VALUES, DIFF_VALUES, SUM_VALUES, 3)
        self.assertEqual(quantifier.shape, correct_size)


    def test_hmmd_quantifiers_files_creation(self):
        self.assertEqual(create_all_hmmd_quantifier_files(), True)


    def test_hmmd256_quantifier_file(self):
        with open(sett.QUANTIFIER_256_FILENAME, 'rb') as f:
            quantifier = pickle.load(f)
        correct_size = (HUE_VALUES, DIFF_VALUES, SUM_VALUES, 3)
        self.assertEqual(quantifier.shape, correct_size)


    def test_hmmd128_quantifier_file(self):
        with open(sett.QUANTIFIER_128_FILENAME, 'rb') as f:
            quantifier = pickle.load(f)
        correct_size = (HUE_VALUES, DIFF_VALUES, SUM_VALUES, 3)
        self.assertEqual(quantifier.shape, correct_size)


    def test_hmmd64_quantifier_file(self):
        with open(sett.QUANTIFIER_64_FILENAME, 'rb') as f:
            quantifier = pickle.load(f)
        correct_size = (HUE_VALUES, DIFF_VALUES, SUM_VALUES, 3)
        self.assertEqual(quantifier.shape, correct_size)


    def test_hmmd32_quantifier_file(self):
        with open(sett.QUANTIFIER_32_FILENAME, 'rb') as f:
            quantifier = pickle.load(f)
        correct_size = (HUE_VALUES, DIFF_VALUES, SUM_VALUES, 3)
        self.assertEqual(quantifier.shape, correct_size)


if __name__ == '__main__':
    unittest.main()
