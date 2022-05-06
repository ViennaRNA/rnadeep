import os
import numpy as np
from tensorflow import keras
from rnadeep.metrics import mcc, f1, sensitivity
from rnadeep.sampling import draw_sets
from rnadeep.data_generators import MatrixEncoding, PaddedMatrixEncoding

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

mymodel = 'spotrna_padded_m3_100000_length119-120-003_bpRNAinv120-022'
m = keras.models.load_model(mymodel,
                            custom_objects = {"mcc": mcc,
                                              "f1": f1, 
                                              "sensitivity": sensitivity})

# Choose a testset.
fname = 'tdata/test.fa'
[tests] = list(draw_sets(fname))
[tags, seqs, dbrs] = zip(*tests)

paddeddata = True
if paddeddata: 
    gen = PaddedMatrixEncoding(1, seqs, dbrs)
    print(m.evaluate(gen))
    mrxs = m.predict(gen)
    # Convert tf.RaggedTensor to numpy arrays.
    mrxs = mrxs.numpy()
    for i in range(len(mrxs)):
        mrxs[i] = np.asarray([x for x in mrxs[i]], dtype=np.float32)
else:
    gen = MatrixEncoding(1, seqs, dbrs)
    print(m.evaluate(gen))
    mrxs = m.predict(gen)

outdir = f"predictions/{mymodel}"
if not os.path.exists(outdir):
    os.makedirs(outdir)

np.save(os.path.join(outdir, 'sequences'), seqs)
np.save(os.path.join(outdir, 'structures'), dbrs)
np.save(os.path.join(outdir, 'matrices'), mrxs)

