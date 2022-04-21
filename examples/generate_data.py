#!/usr/bin/env python

import os
import random
import numpy as np
from rnadeep.sampling import (generate_random_structures, 
                              write_data_file,
                              draw_sets,
                              write_fixed_len_data_file,
                              write_uniform_len_data_file,
                              write_normal_len_data_file)

def main():
    """Generates random sequence data files.

    datadir/datatype.fasta
    '''
    >rseq_{i}_{energy}\n
    sequence\n
    structure\n
    >rseq_{i+1}_{energy}\n
    sequence\n
    structure\n
    '''
    """
    datadir = "data/"
    if not os.path.exists(datadir):
        os.mkdir(datadir)

    #
    # Random sequence/structure pairs with static length
    #
    fname = write_fixed_len_data_file(70, num = 1000, root = datadir)
    print(f'Wrote file: {fname}')

    fname = write_fixed_len_data_file(70, num = 100, root = datadir)
    print(f'Wrote file: {fname}')
    
    #
    # Examples to draw random sets (training, validation, test) from a file.
    #
    for s in draw_sets(fname, splits = [0.8, 0.1, 0.1]):
        tags, seqs, dbrs = list(zip(*s))

    for s in draw_sets(fname, splits = [1]):
        tags, seqs, dbrs = list(zip(*s))

    #
    # Testing variable length distributions ...
    #
    fname = write_uniform_len_data_file(25, 100, num = 100, root = datadir)
    print(f'Wrote file: {fname}')

    #
    # Reproduce previous "uniform-plus results:
    #
    name = f"uniform_plus_len25-100_n100"
    fname = os.path.join(datadir, name)
    l1 = np.random.randint(25,  50+1, int(100 * 0.1))
    l2 = np.random.randint(50,  75+1, int(100 * 0.3))
    l3 = np.random.randint(75, 100+1, int(100 * 0.6))
    rdata = generate_random_structures(np.concatenate((l1,l2,l3), axis = 0))
    write_data_file(rdata, fname)
    print(f'Wrote file: {fname}')

    #
    # Testing normal distributions ...
    #
    fname = write_normal_len_data_file(50, 5, num = 100, root = datadir)
    print(f'Wrote file: {fname}')

    #
    # Testing example spike distribution ...
    #
    num = 100
    name = f"uniform_len25-100_spike{35}_{5}_n{num}"
    fname = os.path.join(datadir, name)
    l1 = np.random.normal(35, 5, int(num * 0.75)).round().astype(np.int64)
    l2 = np.random.randint(25, 100 + 1, num - int(num * 0.75))
    rdata = generate_random_structures(np.concatenate((l1,l2), axis = 0))
    write_data_file(rdata, fname)
    print(f'Wrote file: {fname}')


if __name__ == '__main__':
    main()
