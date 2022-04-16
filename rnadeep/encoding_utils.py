
import numpy as np
import RNA

def one_hot_encode(char):
    char = char.upper()
    if (char=='A'):
        return np.array([1,0,0,0])
    elif (char=='C'):
        return np.array([0,1,0,0])
    elif (char=='G'):
        return np.array([0,0,1,0])
    elif (char=='U'):
        return np.array([0,0,0,1])
    else:
        return np.array([0,0,0,0])

def one_hot_matrix(seq):
    matrix = np.zeros((len(seq), len(seq), 8), dtype = int)
    for i, char_i in enumerate(seq):
        hot_i = one_hot_encode(char_i)
        for j, char_j in enumerate(seq):
            hot_j = one_hot_encode(char_j)
            matrix[i][j] = np.concatenate((hot_i, hot_j))
            matrix[j][i] = np.concatenate((hot_j, hot_i))
    return matrix
 
def base_pair_matrix(ss):
    # ptable[i] = j if (i.j) pair or 0 if i is unpaired, 
    # ptable[0] contains the length of the structure.
    ptable = RNA.ptable(ss) 
    matrix = np.zeros((len(ss), len(ss), 1), dtype = int)
    for i in range(1, len(ptable)):
        if ptable[i] != 0:
            j = ptable[i]
            matrix[i-1][j-1] = 1
    return matrix
 
def encode_sequence_matrix(sequences):
    """
    Make a BP probability matrix with one-hot encoding of basepairs.
    NOTE: This only works if all sequences have the same length, otherwise
    you need to use: encode_padded_sequence_matrix
    """
    assert min(len(s) for s in sequences) == max(len(s) for s in sequences)
    return np.asarray([one_hot_matrix(seq) for seq in sequences], dtype=np.float32)
   
def encode_structure_matrix(structures):
    """
    Make a BP probability matrix with one-hot encoding of basepairs.
    NOTE: This only works if all sequences have the same length!
    """
    assert min(len(s) for s in structures) == max(len(s) for s in structures)
    return np.asarray([base_pair_matrix(ss) for ss in structures], dtype=np.float32)

def encode_padded_sequence_matrix(sequences, max_length):
    batch_size = len(sequences)

    # TODO: why are there two?
    x_array = np.zeros((batch_size, max_length, max_length, 8))
    matrix_array = np.zeros((batch_size, max_length, max_length))

    for i, seq in enumerate(sequences): 
            wl = max_length - len(seq)

            # The one hot encoding.
            x = one_hot_matrix(seq)
            x = np.pad(x, ((0, wl), (0, wl), (0, 0)), 'constant')
            x_array[i] = x

            # Sequence=1 vs padding=0
            matrix = np.ones((len(seq), len(seq)))
            matrix = np.pad(matrix, ((0, wl), (0, wl)), 'constant')
            matrix_array[i] = matrix

    return x_array, matrix_array

def encode_padded_structure_matrix(structures, max_length):
    batch_size = len(structures)

    y_array = np.zeros((batch_size, max_length, max_length, 1))
    for j in range(batch_size):
        ss = structures[j]
        wl = max_length - len(ss)
        y = base_pair_matrix(ss)
        y = np.pad(y, ((0, wl), (0, wl), (0, 0)), 'constant')
        y_array[j] = y
    return y_array

def binary_encode(structure):
    intab = "()."
    outtab = "110"
    trantab = str.maketrans(intab, outtab)
    structure = structure.translate(trantab)
    values = np.asarray(list(structure), dtype = int)
    return values.reshape(len(values), 1)

def encode_structure(structures):
    return np.asarray([binary_encode(ss) for ss in structures])

def encode_sequence(sequences):
    return np.asarray([[one_hot_encode(char) for char in seq] for seq in sequences])

def create_windows(sequences, window_size):
    windows = []
    for seq in sequences:
        for pos in range(len(seq)):
            window = seq[max(0,pos-window_size):(min(len(seq),pos+window_size)+1)]
            window = ("N" * -(pos-window_size)) + window + ("N" * (1+pos+window_size-len(seq)))
            windows.append(window)
    return np.asarray(windows)
    
def encode_sequence_windows(sequences, window_size):
    windows = create_windows(sequences, window_size)
    return np.asarray([[one_hot_encode(char) for char in sw] for sw in windows])

