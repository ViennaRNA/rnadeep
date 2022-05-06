
import os
import RNA
import numpy as np
from itertools import combinations

def julia_version(a):
    # make symmetric
    a = (a + a.T)/2

    # Take the maximum from each row and write it into a vector
    row_maxes = a.max(axis=1).reshape(-1, 1)
    col_maxes = a.max(axis=0).reshape(-1, 1)
    assert np.allclose(row_maxes, col_maxes)

    # Translate to only 0's and 1's dependent on th >= 0.5
    return np.where((a == row_maxes) & (a >= 0.5), 1, 0)

def remove_conflicts(a, seq = None):
    """ 
    """
    # extract make symmteric upper triangle
    a = np.triu((a + a.T)/2, 1)
    # remove unpaired elements
    a = np.where((a < 0.5), 0, a)

    nbp = np.count_nonzero(a)

    # Get indices of the largest element in a
    (i, j) = np.unravel_index(a.argmax(), a.shape)
    while (i, j) != (0, 0):
        if not seq or canon_bp(seq[i],seq[j]):
            a[i,], a[:,j] = 0, 0 # looks inefficient
            a[j,], a[:,i] = 0, 0 # looks inefficient
            a[i,j] = -1
        else:
            a[i,j] = 0
        (i, j) = np.unravel_index(a.argmax(), a.shape)
    return np.where((a == -1), 1, a), nbp
   
def canon_bp(i, j):
    can_pair = {'A': {'A': 0, 'C': 0, 'G': 0, 'U': 1},
                'C': {'A': 0, 'C': 0, 'G': 1, 'U': 0},
                'G': {'A': 0, 'C': 1, 'G': 0, 'U': 1},
                'U': {'A': 1, 'C': 0, 'G': 1, 'U': 0}}
    return can_pair[i][j]

def julia_prediction(seqs, data):
    """
    Wenn ich richtig verstehe, wird ab Zeile 36 eine 0,1 Matrix gebaut, die 1
    Einträge enthält wenn der NN output >0.5 war und der Eintrag der größte auf
    der Zeile ist. Kling gut, ausser dass auch pro Spalte höchstens eine 1
    stehen darf.  Wird die NN Matrix vielleicht vorher schon symmetrisch
    gemacht?

    Der code failed auch, wenn der Maximalwert in einer Zeile doppelt vorkommt,
    aber das ist hoffentlich selten genug.
    """
    nn_structs = []
    collisions = 0
    tot_nbp = 0
    for (seq, nnd) in zip(seqs, data):
        ## remove one nesting
        a = np.reshape(nnd, (len(seq), len(seq)))
        
        # version 1: allow all base-pairs
        a, nbp = remove_conflicts(a)
        #tot_nbp += nbp
        ## version 2: only canonical base-pairs
        #a = remove_conflicts(a, seq)
        # version 3: juila's version (with symmetry correction)
        #a = julia_version(a)
        #unique, counts = np.unique(a, return_counts=True)

        # Make a pair table by looping over the upper triangular matrix ...
        pt = np.zeros(len(seq)+1, dtype = int)
        pt[0] = len(seq)
        for i in range(len(seq)):
            for j in range(i+1, len(seq)):
                if a[i][j] == 1:
                    if pt[i+1] == pt[j+1] == 0:
                        pt[i+1], pt[j+1] = j+1, i+1
                    else:
                        collisions += 1

        #remove pseudoknots & convert to dot-bracket
        ptable = tuple(map(int,pt))
        processed = RNA.pt_pk_remove(ptable)
        nns = RNA.db_from_ptable(processed)
        nn_structs.append(nns)
    tot_nbp /= len(seqs)
    #print(tot_nbp)
    #print(f'{collisions = }')
    return nn_structs

def julia_looptypes(seqs, structs):
    stats = {'S': 0, # stack (actually: paired)
             'E': 0, # exterior
             'B': 0, # bulge
             'H': 0, # hairpin
             'I': 0, # interior
             'M': 0} # multi
    counts = {'#S': 0, # stack (actually: paired)
              '#E': 0, # exterior
              '#B': 0, # bulge
              '#H': 0, # hairpin
              '#I': 0, # interior
              '#M': 0} # multi
    for (seq, ss) in zip(seqs, structs):
        assert len(seq) == len(ss)
        # Counting paired vs unpaired is easy ...
        S = len([n for n in ss if n != '.'])
        L = len([n for n in ss if n == '.'])
        # ... but which loop are the unpaired ones in?
        tree = RNA.db_to_tree_string(ss, 5) # type 5
        print(f'\r', end = '') # Unfortunately, the C function above prints a string!!!
        tdata = [x for x in tree.replace(")","(").split("(") if x and x != 'R']
        scheck, lcheck = 0, 0
        for x in tdata:
            if x[0] == 'S':
                stats[x[0]] += 2*int(x[1:])/len(seq)
                counts[f'#{x[0]}'] += 1 # hmmm.... 1 or 2?
                scheck += 2*int(x[1:])
            else:
                stats[x[0]] += int(x[1:])/len(seq)
                counts[f'#{x[0]}'] += 1
                lcheck += int(x[1:])
        assert scheck == S and lcheck == L
    stats = {t: c/len(seqs) for t, c in stats.items()} 
    counts = {t: c/len(seqs) for t, c in counts.items()} 
    assert np.isclose(sum(stats.values()), 1.)
    return stats, counts

def get_bp_counts(seqs, structs):
    counts = {}
    for (seq, ss) in zip(seqs, structs):
        pt = RNA.ptable(ss)
        for i, j in enumerate(pt[1:], 1):
            if j == 0 or i > j:
                continue
            bp = (seq[i-1], seq[j-1])
            counts[bp] = counts.get(bp, 0) + 1
    return counts

def main():
    header = (f'Model Length paired exterior bulge hairpin interior multi '
           f'#helices #exterior #bulge #hairpin #interior #multi '
           f'base-pairs   %GC   %CG   %AU   %UA   %GU   %UG          %NC')

    root = 'predictions/'
    datadirs = ['spotrna_padded_m3_100000_length119-120-003_bpRNAinv120-008',
                'spotrna_padded_m3_100000_length119-120-003_bpRNAinv120-022']
    for dd in datadirs:
        print(f'\n# Analyzed model results from {dd}')
            
        seqs = np.load(os.path.join(root, dd, 'sequences.npy'))
        vrna = np.load(os.path.join(root, dd, 'structures.npy'))
        data = np.load(os.path.join(root, dd, 'matrices.npy'), allow_pickle = True)

        l = max(len(s) for s in seqs)
        if min(len(s) for s in seqs) != l:
            l = 0

        # Show loop types from the MFE structures
        lt_vrna, lt_counts = julia_looptypes(seqs, vrna)
        bp_vrna = get_bp_counts(seqs, vrna)
        bp_tot_vrna = sum(bp_vrna.values())
        bp_vrna = {bp: cnt/bp_tot_vrna for bp, cnt in bp_vrna.items()}
        
        nc_vrna = sum([val for (i, j), val in bp_vrna.items() if not canon_bp(i,j)])
        #assert nc_vrna == 0

        print(header)
        print((f"{'vrna':5s} {l:>6d}"
               f"{lt_vrna['S']:>7.3f} "
               f"{lt_vrna['E']:>8.3f} "
               f"{lt_vrna['B']:>5.3f} "
               f"{lt_vrna['H']:>7.3f} "
               f"{lt_vrna['I']:>8.3f} "
               f"{lt_vrna['M']:>5.3f} "
               f"{lt_counts['#S']:>8.3f} "
               f"{lt_counts['#E']:>9.3f} "
               f"{lt_counts['#B']:>6.3f} "
               f"{lt_counts['#H']:>8.3f} "
               f"{lt_counts['#I']:>9.3f} "
               f"{lt_counts['#M']:>6.3f} "
               f"{bp_tot_vrna:>10d} "
               f"{bp_vrna[('G','C')]:>5.3f} "
               f"{bp_vrna[('C','G')]:>5.3f} "
               f"{bp_vrna[('A','U')]:>5.3f} "
               f"{bp_vrna[('U','A')]:>5.3f} "
               f"{bp_vrna[('G','U')]:>5.3f} "
               f"{bp_vrna[('U','G')]:>5.3f} "
               f"{nc_vrna:>12.10f} "))

        nnss = julia_prediction(seqs, data)
        # Show loop types from the neural network structures
        lt_nnss, lt_counts = julia_looptypes(seqs, nnss)
        bp_nnss = get_bp_counts(seqs, nnss)
        bp_tot_nnss = sum(bp_nnss.values())
        bp_nnss = {bp: cnt/bp_tot_nnss for bp, cnt in bp_nnss.items()}
        nc_nnss = sum([val for (i, j), val in bp_nnss.items() if not canon_bp(i,j)])
        print((f"{'nnss':5s} {l:>6d}"
               f"{lt_nnss['S']:>7.3f} "
               f"{lt_nnss['E']:>8.3f} "
               f"{lt_nnss['B']:>5.3f} "
               f"{lt_nnss['H']:>7.3f} "
               f"{lt_nnss['I']:>8.3f} "
               f"{lt_nnss['M']:>5.3f} "
               f"{lt_counts['#S']:>8.3f} "
               f"{lt_counts['#E']:>9.3f} "
               f"{lt_counts['#B']:>6.3f} "
               f"{lt_counts['#H']:>8.3f} "
               f"{lt_counts['#I']:>9.3f} "
               f"{lt_counts['#M']:>6.3f} "
               f"{bp_tot_nnss:>10d} "
               f"{bp_nnss[('G','C')]:>5.3f} "
               f"{bp_nnss[('C','G')]:>5.3f} "
               f"{bp_nnss[('A','U')]:>5.3f} "
               f"{bp_nnss[('U','A')]:>5.3f} "
               f"{bp_nnss[('G','U')]:>5.3f} "
               f"{bp_nnss[('U','G')]:>5.3f} "
               f"{nc_nnss:>12.10f} "))
   
        # In case you care, look at the test structures vs predicted structures!
        for s, v, n in zip(seqs, vrna, nnss):
            print(f'# {s}')
            print(f'# {v} (test)')
            print(f'# {n} (pred)')

if __name__ == '__main__':
    main()

