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

#Model Settings
model = 1
batch_size = 8
epochs = 3
rootd = "data/"
data = "uniform_25-100"
train = "train_30k"
valid = "val_5k"

name = f"spotrna_m{model}_bs{batch_size}_{data}_{train}"

sequences = np.load(os.path.join(rootd, data, train, "sequences.npy"), mmap_mode = "r")
structures = np.load(os.path.join(rootd, data, train, "structures.npy"), mmap_mode = "r")
train_generator = PaddedMatrixEncoding(batch_size, sequences, structures)

sequences = np.load(os.path.join(rootd, data, valid, "sequences.npy"), mmap_mode = "r")
structures = np.load(os.path.join(rootd, data, valid, "structures.npy"), mmap_mode = "r")
val_generator = PaddedMatrixEncoding(batch_size, sequences, structures)


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
      validation_data = val_generator,
      epochs = epochs,
      shuffle = True,
      verbose = 1,
      callbacks = [csv_logger, model_checkpoint])

#save model after last epochs 
m.save(f"{name}_ep{epochs}.rnadeep")

