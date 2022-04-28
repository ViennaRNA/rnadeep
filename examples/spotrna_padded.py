#!/usr/bin/env python

#
# Training with padded data 
#

import os
import numpy as np
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint

#from rnadeep.myspotrna import spotrna
from rnadeep.spotrna import spotrna
from rnadeep.metrics import mcc, f1, sensitivity
from rnadeep.data_generators import PaddedMatrixEncoding
from rnadeep.sampling import draw_sets

# Suppress annoying warning messages
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

#
# Get the data for analysis
#
fname = "data/uniform_len70-120_n1000.fa"

[train, valid] = list(draw_sets(fname, splits = [0.8, 0.2]))
[train_tags, train_seqs, train_dbrs] = zip(*train)
[valid_tags, valid_seqs, valid_dbrs] = zip(*valid)

#
# Training/Validation data backup
#
with open(fname+'-train', 'w') as f:
    for t, s, d in zip(train_tags, train_seqs, train_dbrs):
        f.write(f'{t}\n{s}\n{d}\n')
with open(fname+'-valid', 'w') as f:
    for t, s, d in zip(valid_tags, valid_seqs, valid_dbrs):
        f.write(f'{t}\n{s}\n{d}\n')

#
# Model Settings
#
model = 1
use_mask = True
batch_size = 16
epochs = 10
data, _ = os.path.split(os.path.basename(fname))
name = f"spotrna_padded_m{model}_ep{epochs}_bs{batch_size}_{data}"

#
# Model Setup
#
train_generator = PaddedMatrixEncoding(batch_size, train_seqs, train_dbrs, use_mask = use_mask)
valid_generator = PaddedMatrixEncoding(batch_size, valid_seqs, valid_dbrs, use_mask = use_mask)

m = spotrna(model, use_mask)
m.compile(optimizer = "adam",
          loss = "binary_crossentropy", 
          metrics = ["acc", mcc, f1, sensitivity],
          run_eagerly = True)

# Callback functions for fitting.
csv_logger = CSVLogger(f"{name}.csv", separator = ';', append = True)
model_checkpoint = ModelCheckpoint(filepath = name + '-{epoch:03d}', 
                                   save_weights_only = False, 
                                   monitor = 'val_mcc', 
                                   mode = 'max', 
                                   save_best_only = True)

m.fit(x = train_generator, 
      validation_data = valid_generator,
      epochs = epochs,
      shuffle = True,
      verbose = 1,
      callbacks = [csv_logger, model_checkpoint])

# Save model after last epochs 
m.save(f"{name}.rnadeep")


