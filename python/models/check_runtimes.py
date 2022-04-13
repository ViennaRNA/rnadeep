#!/usr/bin/env python3

from spotrna import spotrna
from metrics import mcc,f1,sensitivity
from data_generators import DataFromFile
import numpy as np
from tensorflow.keras.callbacks import CSVLogger,ModelCheckpoint,Callback
import os
import statistics
import time

class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

#Model Settings
model = 0
use_generator = True
multiprocessing = False

time_callback = TimeHistory()

directory = "/scr/romulus/jwielach/rnadeepfold/data/processed/10000_length70"

batch_size = 32
size = 8000
m = spotrna(model,False)
m.compile(optimizer="adam",loss="binary_crossentropy", metrics=["acc",mcc,f1])

if use_generator:
    train_generator = DataFromFile(batch_size,os.path.join(directory,"train"))
    val_generator = DataFromFile(batch_size,os.path.join(directory,"val"))
    m.fit(x=train_generator, validation_data = val_generator,epochs=5,shuffle = True,verbose=1,use_multiprocessing=multiprocessing,callbacks=[time_callback])
else:
    ytrain = np.load(os.path.join(directory,"train" ,"notpadded_structures.npy"),mmap_mode="r+")
    xtrain = np.load(os.path.join(directory,"train" ,"notpadded_sequences.npy"),mmap_mode="r+")
    yval= np.load(os.path.join(directory,"val" ,"notpadded_structures.npy"),mmap_mode="r+")
    xval = np.load(os.path.join(directory,"val" ,"notpadded_sequences.npy"),mmap_mode="r+")
    m.fit(xtrain, ytrain, batch_size=batch_size,epochs=5,validation_data=(xval,yval),shuffle = True,verbose=1,callbacks=[time_callback])

times = time_callback.times
print(times)
times = [float(i) for i in times]
filename = "/scr/romulus/jwielach/rnadeepfold/slurm_scripts/runtimes/runtimes_32_0_nm_g"
np.save(filename, np.asarray(times))

times_array = np.load("%s.npy" % (filename))
print(sum(times_array) / len(times_array))
print(statistics.pstdev(times_array))
