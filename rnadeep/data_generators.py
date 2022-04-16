
import math
from tensorflow.keras.utils import Sequence

from .encoding_utils import (encode_sequence_matrix,
                             encode_structure_matrix,
                             encode_padded_sequence_matrix,
                             encode_padded_structure_matrix, 
                             encode_sequence,
                             encode_structure,
                             encode_sequence_windows)

class MatrixEncoding(Sequence):
    def __init__(self, batch_size, sequences, structures):
        """ Use this data generator if all sequences have same length, 
        or with batch_size = 1.  """
        self.x = sequences
        self.y = structures
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size] # Sequences
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size] # Structures
        x = encode_sequence_matrix(batch_x)
        y = encode_structure_matrix(batch_y)
        return x, y
        
class PaddedMatrixEncoding(Sequence):
    def __init__(self, batch_size, sequences, structures):
        self.x = sequences
        self.y = structures
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size] # Sequences
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size] # Structures

        maxlen = max(len(ss) for ss in batch_x)
        x1, x2 = encode_padded_sequence_matrix(batch_x, maxlen)
        y = encode_padded_structure_matrix(batch_y, maxlen)
        return [x1, x2], y

class BinaryEncoding(Sequence):
    def __init__(self, batch_size, sequences, structures):
        self.x = sequences
        self.y = structures
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size] # Sequences
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size] # Structures
        x = encode_sequence(batch_x)
        y = encode_structure(batch_y)
        return x, y

class BinaryWindowEncoding(Sequence):
    def __init__(self, batch_size, sequences, structures, window_size):
        self.x = sequences
        self.y = structures
        self.window_size = window_size
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size] # Sequences
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size] # Structures
        # Returns a +/- window-sized window for every nucleotide 
        x = encode_sequence_windows(batch_x, self.window_size)
        # Reshape to see per nucleotide if paired or unpaired
        y = encode_structure(batch_y).reshape(len(batch_y) * len(batch_y[0]), 1)
        return x, y
