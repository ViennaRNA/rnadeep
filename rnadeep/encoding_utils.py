
import numpy as np

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

def make_pair_table(ss, base=0, chars=['.']):
    """ Return a secondary struture in form of pair table.

    Args:
      ss (str): secondary structure in dot-bracket format
      base (int, optional): choose between a pair-table with base 0 or 1
      chars (list, optional): a list of characters to be are ignored, default:
        ['.']

    **Example:**
       base=0: ((..)). => [5,4,-1,-1,1,0,-1]
        i.e. start counting from 0, unpaired = -1
       base=1: ((..)). => [7,6,5,0,0,2,1,0]
        i.e. start counting from 1, unpaired = 0, pt[0]=len(ss)

    Returns:
      [list]: A pair-table
    """
    stack = []
    if base == 0:
        pt = [-1] * len(ss)
    elif base == 1:
        pt = [0] * (len(ss) + base)
        pt[0] = len(ss)
    else:
        raise Exception(f"unexpected value in make_pair_table: (base = {base})")

    for i, char in enumerate(ss, base):
        if (char == '('):
            stack.append(i)
        elif (char == ')'):
            try:
                j = stack.pop()
            except IndexError as e:
                raise Exception("Too many closing brackets in secondary structure")
            pt[i] = j
            pt[j] = i
        elif (char not in set(chars)):
            raise Exception(f"unexpected character in sequence: '{char}'")
    if stack != []:
        raise Exception("Too many opening brackets in secondary structure")
    return pt
  
def base_pair_matrix(ss):
    # ptable[i] = j if (i.j) pair or 0 if i is unpaired, 
    # ptable[0] contains the length of the structure.
    # ptable = RNA.ptable(ss)
    ptable = make_pair_table(ss, 1)
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

def encode_padded_sequence_matrix(sequences, max_length = None):
    if max_length is None:
        max_length = max(len(ss) for ss in sequences)
    batch_size = len(sequences)

    xs = np.zeros((batch_size, max_length, max_length, 8), dtype = np.float32)
    masks = np.zeros((batch_size, max_length, max_length), dtype = np.float32)

    for i, seq in enumerate(sequences): 
            wl = max_length - len(seq)

            # The one hot encoding.
            x = one_hot_matrix(seq)
            x = np.pad(x, ((0, wl), (0, wl), (0, 0)), 'constant')
            xs[i] = x

            # Sequence = 1, padding = 0
            mask = np.ones((len(seq), len(seq)))
            mask = np.pad(mask, ((0, wl), (0, wl)), 'constant')
            masks[i] = mask

    return xs, masks

def encode_padded_structure_matrix(structures, max_length = None):
    if max_length is None:
        max_length = max(len(ss) for ss in structures)
    batch_size = len(structures)

    ys = np.zeros((batch_size, max_length, max_length, 1), dtype = np.float32)
    for j in range(batch_size):
        ss = structures[j]
        wl = max_length - len(ss)
        y = base_pair_matrix(ss)
        y = np.pad(y, ((0, wl), (0, wl), (0, 0)), 'constant')
        ys[j] = y
    return ys

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

