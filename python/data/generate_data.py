#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import os
import RNA
from sklearn.model_selection import train_test_split
import scipy.stats as ss

def create_dataset_differentlengths(lengths):
    #assumes already sorted lengths
    simulated_structures = []
    simulated_sequences = []
    for length in lengths:
        seq = RNA.random_string(int(length),"ACGU")
        (ss, mfe) = RNA.fold(seq)
        simulated_structures.append(ss)
        simulated_sequences.append(seq)
    return (simulated_sequences,simulated_structures)

def create_dataset_onelength(sequencelength, datasetsize,save_directory):
    simulated_structures = []
    simulated_sequences = []
    for _ in range(datasetsize):
        seq = RNA.random_string(int(sequencelength),"ACGU")
        (ss, mfe) = RNA.fold(seq)
        simulated_structures.append(ss)
        simulated_sequences.append(seq)
    dataset = (simulated_sequences,simulated_structures)
    save_dataset_asarray(dataset,save_directory)

def create_from_lengthfile(input,save_directory):
    lengths = np.load(input)
    lengths = np.sort(lengths)
    dataset = create_dataset_differentlengths(lengths)
    save_dataset_asarray(dataset,save_directory)

def create_uniform(start, end, number, save_directory):
    np.random.seed(1)
    #rng = np.random.default_rng()
    #lengths = rng.integers(start,end,endpoint=True, size=number)
    lengths = np.random.randint(start,end+1, number)
    lengths = np.sort(lengths)
    np.save(os.path.join(save_directory, "lengths"), lengths)
    dataset = create_dataset_differentlengths(lengths)
    save_dataset_asarray(dataset,save_directory)

def create_uniform_plus(start,end,total_number,save_directory):
    np.random.seed(1)
    all_lengths = []
    end = 50
    for percentage in [0.1,0.3,0.6]:
        print(start)
        print(end)
        print(percentage)
        number = int(total_number*percentage)
        lengths = np.random.randint(start,end+1, number)
        all_lengths.append(lengths)
        start += 25
        end += 25
    np.concatenate(all_lengths, axis=0 )
    all_lengths = np.concatenate(all_lengths, axis=0 )
    all_lengths = np.sort(all_lengths)
    np.save(os.path.join(save_directory, "lengths"), all_lengths)
    dataset = create_dataset_differentlengths(all_lengths)
    save_dataset_asarray(dataset,save_directory)

def create_normal_plus(start, end, central,std, number, save_directory):
    np.random.seed(1)
    X1 = (np.random.normal(central ,std, int(number*0.75)).round().astype(np.int))
    #rng = np.random.default_rng()
    #X2 = rng.integers(start,end,endpoint=True, size=number-int(number*0.75))
    X2 = np.random.randint(start,end+1, number-int(number*0.75))
    lengths = np.concatenate((X1,X2),axis = 0)
    lengths = np.sort(lengths)
    np.save(os.path.join(save_directory, "lengths"), lengths)
    dataset = create_dataset_differentlengths(lengths)
    save_dataset_asarray(dataset,save_directory)

def save_dataset_asarray(dataset,save_directory):
    simulated_sequences,simulated_structures = dataset
    simulated_structures = np.asarray(simulated_structures, dtype=np.unicode_)
    simulated_sequences =  np.asarray(simulated_sequences, dtype=np.unicode_)
    np.save(os.path.join(save_directory, "simulated_structures"), simulated_structures)
    np.save(os.path.join(save_directory, "simulated_sequences"), simulated_sequences)

def save_from_numpy_validationsplit(storage_directory):
    sequences = np.load(os.path.join(storage_directory, "simulated_sequences.npy"))
    structures = np.load(os.path.join(storage_directory, "simulated_structures.npy"))
    xtrain, xval, ytrain, yval = train_test_split(sequences , structures, test_size = 0.2)
    os.makedirs(os.path.join(storage_directory, "train"),exist_ok=True)
    os.makedirs(os.path.join(storage_directory, "val"),exist_ok=True)
    np.save(os.path.join(storage_directory, "train","simulated_sequences.npy"),xtrain)
    np.save(os.path.join(storage_directory, "train","simulated_structures.npy"),ytrain)
    np.save(os.path.join(storage_directory, "val","simulated_sequences.npy"),xval)
    np.save(os.path.join(storage_directory, "val","simulated_structures.npy"),yval)

def main():
    """
    #create repeated datasets
    directory = "/scr/romulus/jwielach/rnadeepfold/data/interim/10000_length70_ids"
    size  = 10000
    for i in range(10,20):
        os.makedirs(os.path.join(directory, "id_%i" % (i)))
        save_directory = os.path.join(directory, "id_%i" % (i))
        create_dataset_onelength(70, size,save_directory)
        save_from_numpy_validationsplit(save_directory)
    """

    directory = "/scr/romulus/jwielach/rnadeepfold/data/interim/"
    sizes = [30000,10000,50000,5000]
    sets = ["train_30k","train_10k","train_50k","val_5k"]
    for i in range(len(sizes)):
        size = sizes[i]
        set = sets[i]
        print("Creating dataset for set %s, containing %i sequences" % (set,size))
        #create uniform dataset from 25-100
        save_directory = os.path.join(directory, "uniform_plus_25-100",set)
        os.makedirs(os.path.join(save_directory))
        create_uniform_plus(25, 100, size, save_directory)
        #create normal dataset from 25-100
        #save_directory = os.path.join(directory, "normal_lowspike_25-100",set)
        #os.makedirs(os.path.join(save_directory))
        #create_normal_plus(25, 100, 35,5, size, save_directory)
    
    """
    directory = "/scr/romulus/jwielach/rnadeepfold/data/interim/validation_lengths"
    size = 20000
    #base_length = 70
    #for length in range(base_length-30,base_length+80+1,10):
    for length in [20,25,30]:
        os.makedirs(os.path.join(directory, "len_%i_%i" % (length,size)))
        save_directory = os.path.join(directory, "len_%i_%i" % (length,size))
        create_dataset_onelength(length, size,save_directory)
   """

if __name__ == '__main__':
    main()
