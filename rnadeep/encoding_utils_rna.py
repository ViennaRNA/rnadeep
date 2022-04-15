#!/usr/bin/env python3

import argparse
import RNA
import math
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split

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

    
def encode_matrix_array(structure_array):
    encoded_array = np.asarray(list(map(compute_structure_matrix, structure_array)))
    return encoded_array

def encode_onehot_matrix_array(sequence_array):
    encoded_array = np.asarray(list(map(concat_one_hot, sequence_array)))
    return encoded_array

def concat_one_hot(seq):
    matrix = np.zeros((len(seq),len(seq),8), dtype=int)
    for i in range(len(seq)):
        for j in range(len(seq)):
            one_hot_i = one_hot_encode(seq[i])
            one_hot_j = one_hot_encode(seq[j])
            matrix[i][j] = np.concatenate((one_hot_i,one_hot_j))
            matrix[j][i] = np.concatenate((one_hot_j,one_hot_i))
    return matrix

def compute_structure_matrix(ss):
    matrix = np.zeros((len(ss),len(ss),1), dtype=int)
    #(ss, mfe) = RNA.fold(seq)
    pair_table = RNA.ptable(ss) #function for p table in Python?
    #adjacency matrix to adjacency list
    #table[i]=j if (i.j) pair or 0 if i is unpaired, table[0] contains the length of the structure
    #CHECK INDEXING
    for i in range(1,len(ss)+1):
        if pair_table[i] != 0:
            matrix[i-1][pair_table[i]-1]=1
    return matrix
    
def encode_padded_matrix_array(structure_array, max_length):
    batch_size = len(structure_array)
    y_array = np.zeros((batch_size,max_length,max_length,1))
    for j in range(batch_size):
            ss = structure_array[j]
            seq_length = len(ss)
            y = compute_structure_matrix(ss)
            y = np.pad(y, ((0, max_length-seq_length), (0, max_length-seq_length),(0,0)), 'constant')
            y_array[j] = y
    return y_array

def encode_padded_onehot_matrix_array(sequence_array,max_length):
    batch_size = len(sequence_array)
    x_array = np.zeros((batch_size,max_length,max_length,8))
    matrix_array = np.zeros((batch_size,max_length,max_length))
    for j in range(batch_size):
            seq = sequence_array[j]
            seq_length = len(seq)
            x = concat_one_hot(seq)
            x = np.pad(x, ((0, max_length-seq_length), (0, max_length-seq_length),(0,0)), 'constant')
            x_array[j] = x
            matrix = np.ones((seq_length,seq_length))
            matrix = np.pad(matrix, ((0, max_length-seq_length), (0, max_length-seq_length)), 'constant')
            matrix_array[j] = matrix
    return (x_array,matrix_array)