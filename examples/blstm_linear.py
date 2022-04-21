#!/usr/bin/env python

import os
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger

from rnadeep.lstm_models import blstm
from rnadeep.metrics import mcc, f1, sensitivity
from rnadeep.data_generators import BinaryEncoding
from rnadeep.sampling import draw_sets

#
# Get the data for analysis
#
fname = "data/fixlen70_n100.fa"
train, valid, tests = list(draw_sets(fname, splits = [0.8, 0.1, 0.1]))
[train_tags, train_seqs, train_dbrs] = zip(*train)
[valid_tags, valid_seqs, valid_dbrs] = zip(*valid)
[tests_tags, tests_seqs, tests_dbrs] = zip(*tests)

#
# Model Settings
#
batch_size = 5
epochs = 3
data = os.path.basename(fname)
name = f"blstm_bs{batch_size}_{data}"

#
# Model Setup
#
train_generator = BinaryEncoding(batch_size, train_seqs, train_dbrs)
valid_generator = BinaryEncoding(batch_size, valid_seqs, valid_dbrs)

m = blstm(5, 80)
m.compile(optimizer = Adam(learning_rate = 0.01),
          loss = "binary_crossentropy",
          metrics = ["acc", mcc, f1])

csv_logger = CSVLogger(f"{name}.csv", separator = ';', append = True)
m.fit(x = train_generator, 
      validation_data = valid_generator,
      epochs = epochs,
      shuffle = True,
      verbose = 1,
      callbacks = [csv_logger]) 

# save model after training
m.save(f"{name}_ep{epochs}.rnadeep")

