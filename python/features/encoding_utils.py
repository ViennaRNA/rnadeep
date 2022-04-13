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

def seq_one_hot_encode(seq):
    onehot_encoded = list()
    for char in seq:
    	onehot_encoded.append(one_hot_encode(char))
    return np.asarray(onehot_encoded)

def encode_seq_array(seq_array):
    encoded_array = map(seq_one_hot_encode, seq_array)
    return encoded_array

def binary_encode(structure):
    intab = "()."
    outtab = "110"
    trantab = str.maketrans(intab, outtab)
    structure = structure.translate(trantab)
    values = np.asarray(list(structure),dtype = int)
    return values.reshape(len(values), 1)

def encode_structure_array(structure_array):
    encoded_array = map(binary_encode, structure_array)
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

def save_padded(sequences,structures,max_length,directory):
    dataset_size = sequences.size
    x_array = np.zeros((dataset_size,max_length,max_length,8))
    y_array = np.zeros((dataset_size,max_length,max_length,1))
    matrix_array = np.zeros((dataset_size,max_length,max_length))
    for i in range(dataset_size):
        seq = sequences[i]
        ss = structures[i]
        seq_length = len(seq)
        x = concat_one_hot(seq)
        y = compute_structure_matrix(ss)
        x = np.pad(x, ((0, max_length-seq_length), (0, max_length-seq_length),(0,0)), 'constant')
        y = np.pad(y, ((0, max_length-seq_length), (0, max_length-seq_length),(0,0)), 'constant')
        x_array[i] = x
        y_array[i] = y
        matrix = np.ones((seq_length,seq_length))
        matrix = np.pad(matrix, ((0, max_length-seq_length), (0, max_length-seq_length)), 'constant')
        matrix_array[i] = matrix
    np.save(os.path.join(directory, "matrices"), matrix_array)
    np.save(os.path.join(directory, "padded_structures"), y_array)
    np.save(os.path.join(directory, "padded_sequences"),x_array)

def save_padded_andsplit(sequences,structures,directory,batch_size):
    dataset_size = sequences.size
    number_batches = math.ceil(dataset_size/batch_size)
    os.makedirs(os.path.join(directory, "matrices_%i" % (batch_size)),exist_ok=True)
    os.makedirs(os.path.join(directory, "padded_structures_%i" % (batch_size)),exist_ok=True)
    os.makedirs(os.path.join(directory, "padded_sequences_%i" % (batch_size)),exist_ok=True)
    i = 0
    for z in range(number_batches):
        start = i
        end = min(i+batch_size, dataset_size)
        max_length = len(sequences[end-1])
        x_array = np.zeros((end-start,max_length,max_length,8))
        y_array = np.zeros((end-start,max_length,max_length,1))
        matrix_array = np.zeros((end-start,max_length,max_length))
        u = 0
        for j in range(start,end):
            seq = sequences[j]
            ss = structures[j]
            seq_length = len(seq)
            x = concat_one_hot(seq)
            y = compute_structure_matrix(ss)
            x = np.pad(x, ((0, max_length-seq_length), (0, max_length-seq_length),(0,0)), 'constant')
            y = np.pad(y, ((0, max_length-seq_length), (0, max_length-seq_length),(0,0)), 'constant')
            x_array[u] = x
            y_array[u] = y
            matrix = np.ones((seq_length,seq_length))
            matrix = np.pad(matrix, ((0, max_length-seq_length), (0, max_length-seq_length)), 'constant')
            matrix_array[u] = matrix
            u += 1
        np.save(os.path.join(directory, "matrices_%i" % (batch_size), "id-%i" % (z)), matrix_array)
        np.save(os.path.join(directory, "padded_structures_%i" % (batch_size), "id-%i" % (z)), y_array)
        np.save(os.path.join(directory, "padded_sequences_%i" % (batch_size), "id-%i" % (z)),x_array)
        i += 32

def save_notpadded(sequences,structures,directory):
    dataset_size = sequences.size
    length = len(sequences[0])
    x_array = np.zeros((dataset_size,length,length,8))
    y_array = np.zeros((dataset_size,length,length,1))
    for i in range(dataset_size):
        seq = sequences[i]
        ss = structures[i]
        x = concat_one_hot(seq)
        y = compute_structure_matrix(ss)
        x_array[i] = x
        y_array[i] = y
    np.save(os.path.join(directory, "notpadded_structures"), y_array)
    np.save(os.path.join(directory, "notpadded_sequences"),x_array)

def save_from_numpy_padded_andsplit(storage_directory,processed_directory,batch_size = 32):
    sequences = np.load(os.path.join(storage_directory, "simulated_sequences.npy"))
    structures = np.load(os.path.join(storage_directory, "simulated_structures.npy"))
    save_padded_andsplit(sequences,structures,processed_directory, batch_size)

def save_from_numpy_notpadded(storage_directory,processed_directory):
    sequences = np.load(os.path.join(storage_directory, "simulated_sequences.npy"))
    structures = np.load(os.path.join(storage_directory, "simulated_structures.npy"))
    save_notpadded(sequences,structures,processed_directory)

def save_from_numpy_padded(storage_directory,processed_directory):
    sequences = np.load(os.path.join(storage_directory, "simulated_sequences.npy"))
    structures = np.load(os.path.join(storage_directory, "simulated_structures.npy"))
    max_length = len(sequences[len(sequences) - 1])
    save_padded(sequences,structures,max_length,processed_directory)

def save_from_numpy_validationsplit(storage_directory,processed_directory):
    sequences = np.load(os.path.join(storage_directory, "simulated_sequences.npy"))
    structures = np.load(os.path.join(storage_directory, "simulated_structures.npy"))
    xtrain, xval, ytrain, yval = train_test_split(sequences , structures, test_size = 0.2)
    os.makedirs(os.path.join(storage_directory, "train"),exist_ok=True)
    os.makedirs(os.path.join(storage_directory, "val"),exist_ok=True)
    os.makedirs(os.path.join(processed_directory, "train"),exist_ok=True)
    os.makedirs(os.path.join(processed_directory, "val"),exist_ok=True)
    np.save(os.path.join(storage_directory, "train","simulated_sequences.npy"),xtrain)
    np.save(os.path.join(storage_directory, "train","simulated_structures.npy"),ytrain)
    np.save(os.path.join(storage_directory, "val","simulated_sequences.npy"),xval)
    np.save(os.path.join(storage_directory, "val","simulated_structures.npy"),yval)
    """
    xtrain= np.load(os.path.join(storage_directory, "train","simulated_sequences.npy"))
    ytrain= np.load(os.path.join(storage_directory, "train","simulated_structures.npy"))
    xval= np.load(os.path.join(storage_directory, "val","simulated_sequences.npy"))
    yval= np.load(os.path.join(storage_directory, "val","simulated_structures.npy"))
    """
    save_notpadded(xtrain,ytrain,os.path.join(processed_directory, "train"))
    save_notpadded(xval,yval,os.path.join(processed_directory, "val"))

def main():
    storage_directory = "/scr/romulus/jwielach/rnadeepfold/data/interim/"
    processed_directory = "/scr/romulus/jwielach/rnadeepfold/data/processed/"
    sets = ["train","val","test"]

    storage = os.path.join(storage_directory,  "50000_length70")
    processed = os.path.join(processed_directory,  "50000_length70")
    print("Processing set  in \n %s \n and saving to \n %s \n" % (storage,processed))
    save_from_numpy_validationsplit(storage,processed)
    """
    for i in range(len(sets)):
        set = sets[i]
        #create datasets for length 70
        storage = os.path.join(storage_directory,  "length_70",set)
        processed = os.path.join(processed_directory,  "length_70",set)
        save_from_numpy_notpadded(storage,processed)
        #create datasets for length 100
        storage = os.path.join(storage_directory,  "length_100",set)
        processed = os.path.join(processed_directory,  "length_100",set)
        save_from_numpy_notpadded(storage,processed)
        #format uniform dataset from 25-100
        storage = os.path.join(storage_directory, "uniform","25_100",set)
        processed = os.path.join(processed_directory, "uniform","25_100",set)
        save_from_numpy_padded(storage,processed)
        batch_size = 32
        #format uniform dataset from 25-500
        storage = os.path.join(storage_directory, "uniform","25_500",set)
        processed = os.path.join(processed_directory, "uniform","25_500",set)
        save_from_numpy_padded_andsplit(storage,processed,batch_size)
    for i in range(len(sets)):
        set = sets[i]
        batch_size = 32
        #format bpRNA dataset
        storage = os.path.join(storage_directory, "bpRNA_simulated",set)
        processed = os.path.join(processed_directory, "bpRNA_simulated",set)
        print("Processing set %s in \n %s \n and saving to \n %s \n" % (set,storage,processed))
        save_from_numpy_padded_andsplit(storage,processed,batch_size)
    """

if __name__ == '__main__':
    main()
