#!/usr/bin/env python

import os
import numpy as np
import scipy.stats as ss
#from sklearn.model_selection import train_test_split
import RNA

def generate_data_files(lengths, datadir):
    """ Save sequence/structure pairs for the given lengths.
    """
    os.makedirs(os.path.join(datadir))
    sequences, structures, energies = list(zip(*generate_random_structures(lengths)))
    sequences = np.asarray(sequences, dtype = np.unicode_)
    structures = np.asarray(structures, dtype = np.unicode_)
    np.save(os.path.join(datadir, "sequences"), sequences)
    np.save(os.path.join(datadir, "structures"), structures)

def generate_random_structures(lengths):
    """ Using RNA.fold to generate sequence/structure pairs.
    """
    return ((seq, *RNA.fold(seq)) for seq in 
            (RNA.random_string(int(l), "ACGU") for l in lengths))

def sample_uniform(start, end, number):
    return np.random.randint(start, end+1, number)

def sample_uniform_plus(start, end, number):
    """
    This function generates 
        - 10% sequences with [start:end] length distribution
        - 30% sequences with [start+25:end+25] length distribution
        - 60% sequences with [start+50:end+50] length distribution

    Note (SB): I do not understand why.
    """
    lengths = np.array([], dtype = int)
    for percentage in [0.1, 0.3, 0.6]:
        n = int(number * percentage)
        print(f'Generating {n} sequences within [{start}:{end}].')
        lengths = np.append(lengths, np.random.randint(start, end+1, n), axis = 0)
        start += 25
        end += 25
    return lengths

def sample_normal_plus(start, end, central, std, number):
    X1 = (np.random.normal(central, std, int(number*0.75)).round().astype(np.int64))
    X2 = np.random.randint(start,end+1, number-int(number*0.75))
    return np.concatenate((X1,X2), axis = 0)

def save_from_numpy_validationsplit(datadir):
    sequences = np.load(os.path.join(datadir, "sequences.npy"))
    structures = np.load(os.path.join(datadir, "structures.npy"))
    xtrain, xval, ytrain, yval = train_test_split(sequences , structures, test_size = 0.2)
    os.makedirs(os.path.join(datadir, "train"), exist_ok = True)
    os.makedirs(os.path.join(datadir, "val"), exist_ok = True)
    np.save(os.path.join(datadir, "train", "sequences.npy"), xtrain)
    np.save(os.path.join(datadir, "train", "structures.npy"), ytrain)
    np.save(os.path.join(datadir, "val", "sequences.npy"), xval)
    np.save(os.path.join(datadir, "val", "structures.npy"), yval)

def main():
    #np.random.seed(1) # NOTE: now or never.
    datadir = "data/interim/"

    # Specific length distributions ...
    sizes = [500, 300, 100, 50]
    names = ["train_50k", "train_30k", "train_10k", "val_5k"]
    for size, name in zip(sizes, names):
        print(f"Creating dataset for set {name}, containing {size} sequences")

        # Length distribution method "uniform_25-100":
        lengths = np.sort(sample_uniform(25, 100, size))
        directory = os.path.join(datadir, "uniform_25-100", name)
        generate_data_files(lengths, directory)

        # Length distribution method "uniform_plus_25-100":
        lengths = np.sort(sample_uniform_plus(25, 50, size))
        directory = os.path.join(datadir, "uniform_plus_25-100", name)
        generate_data_files(lengths, directory)

        # Length distribution method "normal_lowspike_25-100":
        lengths = np.sort(sample_normal_plus(25, 100, 35, 5, size))
        directory = os.path.join(datadir, "normal_lowspike_25-100", name)
        generate_data_files(lengths, directory)

        #TODO: Length distribution method "normal_25-100"

    # Repeated data set.
    size = 100
    length = 70
    name = f"{size}_length{length}_ids"
    for i in range(1, 11):
        lengths = np.full(size, length)
        directory = os.path.join(datadir, name, f"id_{i:02d}")
        generate_data_files(lengths, directory)

    # Validation data set.
    size = 200
    name = "validation_lengths"
    for l in [20, 25, 30]:
        directory = os.path.join(datadir, name, f"len_{l}_{size}")
        lengths = np.full(size, l)
        generate_data_files(lengths, directory)

if __name__ == '__main__':
    main()
