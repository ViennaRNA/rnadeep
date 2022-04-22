
import os
import random
import numpy as np
import RNA

def rseq(l):
    return ''.join(random.choice('ACGU') for _ in range(l))

def generate_random_structures(lengths):
    """ Using RNA.fold to generate sequence/structure pairs.
    """
    return ((seq, *RNA.fold(seq)) for seq in 
            (rseq(l) for l in lengths))

def write_data_file(data, fname, mode = 'w'):
    """ Save sequence/structure pairs for the given lengths.
    """
    with open(fname, mode) as f:
        for id, (seq, dbr, en) in enumerate(data):
            en = int(round(en*100))
            f.write(f'>rseq_id={id}_en={en}\n{seq}\n{dbr}\n')

def draw_sets(fname, splits):
    assert sum(splits) <= 1

    with open(fname) as f:
        l = [l.strip() for l in f]
        assert len(l) % 3 == 0
        tags = l[0::3]
        seqs = l[1::3]
        dbrs = l[2::3]

    if not (len(tags) == len(seqs) == len(dbrs)):
        raise ValueError('Something about the input file is odd.')

    num = len(tags)
#    if not all(np.isclose(int(num*s), num*s) for s in splits):
#        nums = [num*s for s in splits]
#        raise ValueError(f'Provided splits do not yield integers. ({nums})')
    
    a = np.arange(num)
    for s in splits:
        ids = np.random.choice(a, int(num*s), replace = False)
        nsd = []
        for i in sorted(ids):
            nsd.append((tags[i], seqs[i], dbrs[i]))
        yield nsd
        a = [i for i in a if i not in ids]

def write_fixed_len_data_file(seqlen, num, root = ''):
    name = f"fixlen{seqlen}_n{num}.fa"
    fname = os.path.join(root, name)
    lengths = np.full(num, seqlen)
    rdata = generate_random_structures(lengths)
    write_data_file(rdata, fname)
    return fname

def write_uniform_len_data_file(minlen, maxlen, num, root = ''):
    name = f"uniform_len{minlen}-{maxlen}_n{num}.fa"
    fname = os.path.join(root, name)
    lengths = np.random.randint(minlen, maxlen+1, num)
    rdata = generate_random_structures(lengths)
    write_data_file(rdata, fname)
    return fname

def write_normal_len_data_file(central, std, num, root = ''):
    name = f"normal_dist{central}-{std}_n{num}.fa"
    fname = os.path.join(root, name)
    lengths = np.random.normal(central, std, num).round().astype(np.int64)
    rdata = generate_random_structures(lengths)
    write_data_file(rdata, fname)
    return fname


