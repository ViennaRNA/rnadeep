#!/usr/bin/env python3

import math
import os
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

def seq_one_hot_encode(seq):
    onehot_encoded = list()
    for char in seq:
    	onehot_encoded.append(one_hot_encode(char))
    return np.asarray(onehot_encoded)

def encode_seq_array(seq_array):
    encoded_array = np.asarray(list(map(seq_one_hot_encode, seq_array)))
    return encoded_array

def binary_encode(structure):
    intab = "()."
    outtab = "110"
    trantab = str.maketrans(intab, outtab)
    structure = structure.translate(trantab)
    values = np.asarray(list(structure),dtype = int)
    return values.reshape(len(values), 1)

def encode_structure_array(structure_array):
    encoded_array = np.asarray(list(map(binary_encode, structure_array)))
    return encoded_array

def create_windows(array,window_size=25):
    windows = []
    for seq in array:
        for pos in range(len(seq)):
            window = seq[max(0,pos-window_size):(min(len(seq),pos+window_size)+1)]
            window = ("N" * -(pos-window_size)) + window + ("N" * (1+pos+window_size-len(seq)))
            windows.append(window)
    return np.asarray(windows)

def encode_seq_array_windows(seq_array,window_size):
    seq_array_windows = create_windows(seq_array,window_size)
    encoded_array = np.asarray(list(map(seq_one_hot_encode, seq_array_windows)))
    return encoded_array
