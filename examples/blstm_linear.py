#!/usr/bin/env python

import os
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger

from rnadeep.lstm_models import blstm
from rnadeep.metrics import mcc, f1, sensitivity
from rnadeep.data_generators import BinaryEncoding

batch_size = 5
epochs = 3

rootd = "data/"
train = "l70_n800"
valid = "l70_n200"

name = f"blstm_bs{batch_size}_{train}"

sequences = np.load(os.path.join(rootd, train, "sequences.npy"), mmap_mode = "r")
structures = np.load(os.path.join(rootd, train, "structures.npy"), mmap_mode = "r")
train_generator = BinaryEncoding(batch_size, sequences, structures)

sequences = np.load(os.path.join(rootd, valid, "sequences.npy"), mmap_mode = "r")
structures = np.load(os.path.join(rootd, valid, "structures.npy"), mmap_mode = "r")
valid_generator = BinaryEncoding(batch_size, sequences, structures)

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

