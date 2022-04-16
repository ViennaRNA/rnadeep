#!/usr/bin/env python

#
# Training with nonpadded data
#

import os
import numpy as np
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint

from rnadeep.spotrna import spotrna
from rnadeep.metrics import mcc, f1, sensitivity
from rnadeep.data_generators import MatrixEncoding

#Model Settings
model = 1
batch_size = 5
epochs = 3

rootd = "data/"
train = "l70_n800"
valid = "l70_n200"

name = f"spotrna_m{model}_bs{batch_size}_{train}"

sequences = np.load(os.path.join(rootd, train, "sequences.npy"), mmap_mode = "r")
structures = np.load(os.path.join(rootd, train, "structures.npy"), mmap_mode = "r")
train_generator = MatrixEncoding(batch_size, sequences, structures)

sequences = np.load(os.path.join(rootd, valid, "sequences.npy"), mmap_mode = "r")
structures = np.load(os.path.join(rootd, valid, "structures.npy"), mmap_mode = "r")
valid_generator = MatrixEncoding(batch_size, sequences, structures)

m = spotrna(model, False)
m.compile(optimizer = "adam",
          loss = "binary_crossentropy", 
          metrics = ["acc", mcc, f1, sensitivity],
          run_eagerly = True)

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

