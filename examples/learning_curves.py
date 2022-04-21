
import os
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint

from rnadeep.lstm_models import blstm
from rnadeep.metrics import mcc, f1, sensitivity
from rnadeep.data_generators import BinaryEncoding
from rnadeep.sampling import draw_sets

#
# Model Settings
#
fname = "data/fixlen70_n1000.fa"

train_sizes = [0.2, 0.4, 0.8]

layers = 5
neurons = 80
batch_size = 100
epochs = 50
data = os.path.basename(fname)

for train_size in train_sizes:
    valid_size = train_size / 8
    tests_size = 1 - (train_size + valid_size)
    print(train_size, valid_size, tests_size)
    #
    # Get the data for analysis
    #
    train, valid, tests = list(draw_sets(fname, splits = [train_size, valid_size, tests_size]))
    [train_tags, train_seqs, train_dbrs] = zip(*train)
    [valid_tags, valid_seqs, valid_dbrs] = zip(*valid)
    [tests_tags, tests_seqs, tests_dbrs] = zip(*tests)

    train_generator = BinaryEncoding(batch_size, train_seqs, train_dbrs)
    valid_generator = BinaryEncoding(batch_size, valid_seqs, valid_dbrs)

    name = f"blstm_s{train_size}_bs{batch_size}_l{layers}_n{neurons}_{data}"

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

