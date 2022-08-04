#!/usr/bin/env python

import os
import numpy as np
from tensorflow.keras.models import load_model

from rnadeep.metrics import mcc, f1, sensitivity
from rnadeep.data_generators import PaddedMatrixEncoding
from rnadeep.sampling import draw_sets

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

def main():
    data = '' # choose an input file.
    model = '' # choose a trained model here
    outdir = f"predictions/{model}-{data}"

    evaluate = True
    predict = False
    batch_size = 1

    # Choose a testset.
    [datas] = list(draw_sets(data))
    [tags, seqs, dbrs] = zip(*datas)
    tgen = PaddedMatrixEncoding(batch_size, seqs, dbrs)

    model = load_model(basemodel,
                       custom_objects = {"mcc": mcc, "f1": f1, 
                                         "sensitivity": sensitivity})
    if evaluate: 
        print(m.evaluate(tgen))

    if predict:
        mrxs = m.predict(tgen)
        # Convert tf.RaggedTensor to numpy arrays.
        if not isinstance(mrxs, np.ndarray):
            mrxs = mrxs.numpy()
            for i in range(len(mrxs)):
                mrxs[i] = np.asarray([x for x in mrxs[i]], dtype=np.float32)
        
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        np.save(os.path.join(outdir, 'sequences'), seqs)
        np.save(os.path.join(outdir, 'structures'), dbrs)
        np.save(os.path.join(outdir, 'matrices'), mrxs)

if __name__ == '__main__':
    main()

