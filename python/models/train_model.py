#!/usr/bin/env python3
from sliding_window import basic_window,conv_window
from metrics import mcc,f1,sensitivity
from data_generators import DataFromInterimWindows,DataFromInterimFile
import numpy as np
from tensorflow.keras.callbacks import CSVLogger,ModelCheckpoint,Callback
import os
import statistics
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from lstm_models import blstm,complex_blstm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
tf.get_logger().setLevel('FATAL')

"""
base_directory = "/scr/romulus/jwielach/rnadeepfold/data/interim/10000_length70_ids"
batch_size = 128
window_size = 71
for i in range(0,10):
    m = conv_window(window_size)
    opt = Adam(learning_rate=0.01)
    csvname = "conv_window_10000_%i_%i_dataset%i.csv" % (batch_size,window_size,i)
    csv_logger = CSVLogger(csvname, append=True, separator=';')
    m.compile(optimizer=opt,loss="binary_crossentropy", metrics=["acc",mcc,f1])
    directory = os.path.join(base_directory,"id_%i" % (i))
    train_generator = DataFromInterimWindows(batch_size,os.path.join(directory,"train"),window_size)
    val_generator = DataFromInterimWindows(batch_size,os.path.join(directory,"val"),window_size)
    m.fit(x=train_generator, validation_data = val_generator,epochs=100,shuffle = True,verbose=0,callbacks=[csv_logger])
"""

"""
base_directory = "/scr/romulus/jwielach/rnadeepfold/data/interim/10000_length70_ids"
layers = 1
neurons = 40
batch_size = 128
for i in range(0,20):
  csvname = "lstm_%i_%i_%i_dataset%i.csv" % (batch_size,layers,neurons,i)
  csv_logger = CSVLogger(csvname, append=True, separator=';')
  m = blstm(layers,neurons)
  opt = Adam(learning_rate=0.01)
  m.compile(optimizer=opt,loss="binary_crossentropy", metrics=["acc",mcc,f1])
  directory = os.path.join(base_directory,"id_%i" % (i))
  train_generator = DataFromInterimFile(batch_size,os.path.join(directory,"train"))
  val_generator = DataFromInterimFile(batch_size,os.path.join(directory,"val"))
  m.fit(x=train_generator, validation_data = val_generator,epochs=150,shuffle = True,verbose=0,callbacks=[csv_logger])
"""

directory = "/scr/romulus/jwielach/rnadeepfold/data/interim/100000_length70"

"""
batch_size = 128
window_size = 71
for i in range(10,20):
    m = basic_window(window_size)
    opt = Adam(learning_rate=0.01)
    csvname = "window_10000_%i_%i_id%i.csv" % (batch_size,window_size,i)
    csv_logger = CSVLogger(csvname, append=True, separator=';')
    m.compile(optimizer=opt,loss="binary_crossentropy", metrics=["acc",mcc,f1])
    train_generator = DataFromInterimWindows(batch_size,os.path.join(directory,"train"),window_size)
    val_generator = DataFromInterimWindows(batch_size,os.path.join(directory,"val"),window_size)
    m.fit(x=train_generator, validation_data = val_generator,epochs=100,shuffle = True,verbose=0,callbacks=[csv_logger])
"""


layers = 1
neurons = 80
batch_size = 128
csvname = "lstm_100k_%i_%i_%i.csv" % (batch_size,layers,neurons)
csv_logger = CSVLogger(csvname, append=True, separator=';')
m = blstm(layers,neurons)
opt = Adam(learning_rate=0.01)
m.compile(optimizer=opt,loss="binary_crossentropy", metrics=["acc",mcc,f1])
train_generator = DataFromInterimFile(batch_size,os.path.join(directory,"train"))
val_generator = DataFromInterimFile(batch_size,os.path.join(directory,"val"))
m.fit(x=train_generator, validation_data = val_generator,epochs=150,shuffle = True,verbose=0,callbacks=[csv_logger])
"""
layers = 2
neurons = 80
batch_size = 128
csvname = "complexlstm_10k_%i_%i_%i.csv" % (batch_size,layers,neurons)
csv_logger = CSVLogger(csvname, append=True, separator=';')
m = complex_blstm(layers,neurons)
opt = Adam(learning_rate=0.01)
m.compile(optimizer=opt,loss="binary_crossentropy", metrics=["acc",mcc,f1])
train_generator = DataFromInterimFile(batch_size,os.path.join(directory,"train"))
val_generator = DataFromInterimFile(batch_size,os.path.join(directory,"val"))
m.fit(x=train_generator, validation_data = val_generator,epochs=150,shuffle = True,verbose=0,callbacks=[csv_logger])
"""
