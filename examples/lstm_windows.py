#!/usr/bin/env python

import os
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger

from rnadeep.sliding_window import basic_window, conv_window
from rnadeep.metrics import mcc, f1, sensitivity
from rnadeep.data_generators import BinaryWindowEncoding
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
window_size = 15 # yields windows of size: 15 + 1 + 15
epochs = 3
data = os.path.basename(fname)
name = f"window_bs{batch_size}_ws{window_size}_{data}"

#
# Model Setup
#
train_generator = BinaryWindowEncoding(batch_size, train_seqs, train_dbrs, window_size)
valid_generator = BinaryWindowEncoding(batch_size, valid_seqs, valid_dbrs, window_size)

m = conv_window(window_size) # use basic_window or conv_window here 
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

