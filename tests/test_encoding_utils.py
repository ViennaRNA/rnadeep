
import unittest
import numpy as np
from rnadeep.encoding_utils import (encode_sequence_matrix, 
                                    encode_structure_matrix,
                                    encode_padded_sequence_matrix, 
                                    encode_padded_structure_matrix)

class TestUtils(unittest.TestCase):

    def test_encode_matrices(self):
        seq = "GCCCUUGUCGAGAGGAACUCGAGACACCCA"
        dbr = ".....((((..(((...)))..))))...."

        m1 = encode_sequence_matrix([seq])
        m2 = encode_structure_matrix([dbr])
        assert m1.dtype == np.float32
        assert m2.dtype == np.float32
        assert m1.shape == (1, 30, 30, 8)
        assert m2.shape == (1, 30, 30, 1)

    def test_encode_padded_matrices(self):
        seq1 = "CCUUGUCGAAGAGACACC"
        dbr1 = "...((((.....)))).."
        seq2 = "CGUCGCCUUACGGGCGCUCA"
        dbr2 = ".(.(((((...))))))..."

        x1, x2 = encode_padded_sequence_matrix([seq1, seq2])
        assert x1.dtype == np.float32
        assert x2.dtype == np.float32
        assert x1.shape == (2, 20, 20, 8)
        assert x2.shape == (2, 20, 20, 1)
        # 1: 16*16 + 20*20 * 2
        assert dict(zip(*np.unique(x1, return_counts = True))) == {0.0: 4952, 1.0: 1448}
        # 1: 16*16 + 20*20
        assert dict(zip(*np.unique(x2, return_counts = True))) == {0.0: 76, 1.0: 724}

        y = encode_padded_structure_matrix([dbr1, dbr2])
        assert y.dtype == np.float32
        assert y.shape == (2, 20, 20, 1)
        # 1: 10 base pairs in total
        assert dict(zip(*np.unique(y, return_counts = True))) == {0.0: 780, 1.0: 20}


