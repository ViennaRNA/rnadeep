
import os
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint

from rnadeep.lstm_models import blstm
from rnadeep.metrics import mcc, f1, sensitivity
from rnadeep.data_generators import BinaryEncoding


train_sizes = [10, 100, 200, 800]
layers = 5
neurons = 80
batch_size = 100
epochs = 50

rootd = "data/"
train = "l70_n800"
valid = "l70_n200"

for tsize in train_sizes:
    # Not like in the notebook
    vsize = int(round(tsize * 0.25))

    sequences = np.load(os.path.join(rootd, train, "sequences.npy"), mmap_mode = "r")[0:tsize]
    structures = np.load(os.path.join(rootd, train, "structures.npy"), mmap_mode = "r")[0:tsize]
    train_generator = BinaryEncoding(batch_size, sequences, structures)

    sequences = np.load(os.path.join(rootd, valid, "sequences.npy"), mmap_mode = "r")[0:vsize]
    structures = np.load(os.path.join(rootd, valid, "structures.npy"), mmap_mode = "r")[0:vsize]
    valid_generator = BinaryEncoding(batch_size, sequences, structures)

    name = f"blstm_s{tsize}_bs{batch_size}_l{layers}_n{neurons}"

    m = blstm(layers, neurons)
    m.compile(optimizer = Adam(learning_rate=0.01),
              loss = "binary_crossentropy", 
              metrics = ["acc", mcc, f1])

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

