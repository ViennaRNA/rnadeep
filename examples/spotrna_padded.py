#!/usr/bin/env python

#
# Training with padded data 
#

import os
import numpy as np
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint

from rnadeep.spotrna import spotrna
from rnadeep.metrics import mcc, f1, sensitivity
from rnadeep.data_generators import PaddedMatrixEncoding
from rnadeep.sampling import draw_sets

#
# Get the data for analysis
#
fname = "data/uniform_len25-100_n100.fa"
train, valid, tests = list(draw_sets(fname, splits = [0.8, 0.1, 0.1]))
[train_tags, train_seqs, train_dbrs] = zip(*train)
[valid_tags, valid_seqs, valid_dbrs] = zip(*valid)
[tests_tags, tests_seqs, tests_dbrs] = zip(*tests)

#
# Model Settings
#
model = 1
batch_size = 8
epochs = 3
data = os.path.basename(fname)
name = f"spotrna_m{model}_bs{batch_size}_{data}"

#
# Model Setup
#
train_generator = PaddedMatrixEncoding(batch_size, train_seqs, train_dbrs)
valid_generator = PaddedMatrixEncoding(batch_size, valid_seqs, valid_dbrs)

m = spotrna(model, True)
m.compile(optimizer = "adam",
          loss = "binary_crossentropy", 
          metrics = ["acc", mcc, f1, sensitivity],
          run_eagerly = True)

# Callback functions for fitting.
csv_logger = CSVLogger(f"{name}.csv", separator = ';', append = True)
model_checkpoint = ModelCheckpoint(filepath = name, 
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

#save model after last epochs 
m.save(f"{name}_ep{epochs}.rnadeep")

