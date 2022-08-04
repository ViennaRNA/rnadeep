import os
import RNA
import numpy as np
from rnadeep.sampling import write_data_file

# This script converts an older deprecated input format 
# where sequences and structures were given as .npy files 
# into the current standard: fasta-like files with 
# >name
# SEQUENCE
# STRUCTURE
# ...
root = '' # TODO: set the root directory here.

seqs = 'simulated_sequences.npy'
dbrs = 'simulated_structures.npy'

seqs = np.load(os.path.join(root, seqs))
dbrs = np.load(os.path.join(root, dbrs))

print(seqs)
print(dbrs)
data = []
for seq, dbr in zip(seqs, dbrs):
    [ss, en] = RNA.fold(seq)
    assert ss == dbr
    data.append((seq, ss, en))

# alternatively:
# data = ((seq, *RNA.fold(seq)) for seq in seqs)

fname = 'combined.fa'
write_data_file(data, os.path.join(root, fname))

