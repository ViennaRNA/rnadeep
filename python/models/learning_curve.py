#!/usr/bin/env python3
from sliding_window import basic_window
from metrics import mcc,f1,sensitivity
from data_generators import DataSliceFromInterimWindows
import numpy as np
from tensorflow.keras.callbacks import CSVLogger,ModelCheckpoint,Callback
import os
import statistics
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from lstm_models import blstm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
tf.get_logger().setLevel('FATAL')

directory = "/scr/romulus/jwielach/rnadeepfold/data/interim/100000_length70"


batch_size = 128
window_size = 71

train_sizes = [10,1000,10000,25000,50000,80000]

for size in train_sizes:
    m = basic_window(window_size)
    opt = Adam(learning_rate=0.01)
    csvname = "window_%i_%i_%i.csv" % (batch_size,window_size,size)
    csv_logger = CSVLogger(csvname, append=True, separator=';')
    m.compile(optimizer=opt,loss="binary_crossentropy", metrics=["acc",mcc,f1])
    train_generator = DataSliceFromInterimWindows(batch_size,os.path.join(directory,"train"),size,window_size)
    val_generator = DataSliceFromInterimWindows(batch_size,os.path.join(directory,"val"),size,window_size)
    m.fit(x=train_generator, validation_data = val_generator,epochs=100,shuffle = True,verbose=0,callbacks=[csv_logger])
