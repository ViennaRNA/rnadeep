
import unittest
import numpy as np
from rnadeep.encoding_utils import encode_sequence_matrix, encode_structure_matrix

class TestUtils(unittest.TestCase):

    def test_encode_sequence_matrix(self):
        a = np.array(['ACGU', 'GCCC'])
        print(encode_sequence_matrix(a))

    def test_encode_matrices(self):
        seq = "GCCCUUGUCGAGAGGAACUCGAGACACCCA"
        dbr = ".....((((..(((...)))..))))...."

        m1 = encode_sequence_matrix([seq])
        m2 = encode_structure_matrix([dbr])
        print(m1.shape, m1.dtype)
        print(m2.shape, m2.dtype)

    def test_encode_matrices(self):
        seq1 = "GCCCUUGUCGAGAGGAACUCGAGACACCCA"
        dbr1 = ".....((((..(((...)))..))))...."
        seq2 = "CUCGUCGCCUUAAUCCAGUGCGGGCGCUAGACA"
        dbr2 = "((.(.(((((...........)))))).))..."

        m1 = encode_sequence_matrix([seq1, seq2])
        m2 = encode_structure_matrix([dbr1, dbr2])
        print(m1.shape, m1.dtype)
        print(m2.shape, m2.dtype)
        print(m1)




