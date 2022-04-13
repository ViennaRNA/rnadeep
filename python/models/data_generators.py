#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from encoding_utils import encode_structure_array,encode_seq_array,encode_seq_array_windows
import os

class DataFromDirectory(Sequence):
    def __init__(self,directory,batch_size):
        self.batch_size = batch_size
        self.directory = directory
        self.files = os.listdir(os.path.join(self.directory, "input"))
        self.on_epoch_end()

    def __len__(self):
        return int(len(self.files))

    def __getitem__(self,batch_id):
        file = self.files[batch_id]
        matrix_directory = os.path.join(self.directory, "matrices_%i" % (self.batch_size))
        target_directory = os.path.join(self.directory, "padded_structures_%i" % (self.batch_size))
        input_directory= os.path.join(self.directory, "padded_sequences_%i" % (self.batch_size))
        x1 = np.load(os.path.join(input_directory, file))
        x2 = np.load(os.path.join(matrix_directory, file))
        y = np.load(os.path.join(target_directory, file))
        X = [x1,x2]
        return X,y

class DataFromFile(Sequence):
    def __init__(self,batch_size,directory):
        self.y = np.load(os.path.join(directory,"notpadded_structures.npy"),mmap_mode="r")
        self.x = np.load(os.path.join(directory,"notpadded_sequences.npy"),mmap_mode="r")
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
        l = int(self.x.shape[0] / self.batch_size)
        if l*self.batch_size < self.x.shape[0]:
            l += 1
        return l

    def __getitem__(self,batch_id):
        if ((batch_id+1)*self.batch_size > self.x.shape[0]):
            X = np.take(self.x,range(batch_id*self.batch_size,self.x.shape[0]-1), axis=0)
            y = np.take(self.y,range(batch_id*self.batch_size,self.x.shape[0]-1), axis=0)
        else:
            X = np.take(self.x,range(batch_id*self.batch_size,(batch_id+1)*self.batch_size), axis=0)
            y = np.take(self.y,range(batch_id*self.batch_size,(batch_id+1)*self.batch_size), axis=0)
        return X,y

class DataFromInterimFile(Sequence):
    def __init__(self,batch_size,directory):
        self.y = np.load(os.path.join(directory,"simulated_structures.npy"),mmap_mode="r")
        self.x = np.load(os.path.join(directory,"simulated_sequences.npy"),mmap_mode="r")
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
        l = int(self.x.shape[0] / self.batch_size)
        if l*self.batch_size < self.x.shape[0]:
            l += 1
        return l

    def __getitem__(self,batch_id):
        if ((batch_id+1)*self.batch_size > self.x.shape[0]):
            X = np.take(self.x,range(batch_id*self.batch_size,self.x.shape[0]-1), axis=0)
            y = np.take(self.y,range(batch_id*self.batch_size,self.x.shape[0]-1), axis=0)
        else:
            X = np.take(self.x,range(batch_id*self.batch_size,(batch_id+1)*self.batch_size), axis=0)
            y = np.take(self.y,range(batch_id*self.batch_size,(batch_id+1)*self.batch_size), axis=0)
        seq_length = len(X[0])
        y = encode_structure_array(y).reshape(y.shape[0],seq_length,1)
        X = encode_seq_array(X).reshape(X.shape[0],seq_length,4)
        return X,y

class DataSliceFromInterimFile(Sequence):
    def __init__(self,batch_size,directory,dataset_size):
        self.y = np.load(os.path.join(directory,"simulated_structures.npy"),mmap_mode="r")[0:dataset_size]
        self.x = np.load(os.path.join(directory,"simulated_sequences.npy"),mmap_mode="r")[0:dataset_size]
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
        l = int(self.x.shape[0] / self.batch_size)
        if l*self.batch_size < self.x.shape[0]:
            l += 1
        return l

    def __getitem__(self,batch_id):
        if ((batch_id+1)*self.batch_size > self.x.shape[0]):
            X = np.take(self.x,range(batch_id*self.batch_size,self.x.shape[0]-1), axis=0)
            y = np.take(self.y,range(batch_id*self.batch_size,self.x.shape[0]-1), axis=0)
        else:
            X = np.take(self.x,range(batch_id*self.batch_size,(batch_id+1)*self.batch_size), axis=0)
            y = np.take(self.y,range(batch_id*self.batch_size,(batch_id+1)*self.batch_size), axis=0)
        seq_length = len(X[0])
        y = encode_structure_array(y).reshape(y.shape[0],seq_length,1)
        X = encode_seq_array(X).reshape(X.shape[0],seq_length,4)
        return X,y


class DataFromInterimWindows(Sequence):
    def __init__(self,batch_size,directory,window_size):
        self.y = np.load(os.path.join(directory,"simulated_structures.npy"),mmap_mode="r")
        self.x = np.load(os.path.join(directory,"simulated_sequences.npy"),mmap_mode="r")
        self.window_size = window_size
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
        l = int(self.x.shape[0] / self.batch_size)
        if l*self.batch_size < self.x.shape[0]:
            l += 1
        return l

    def __getitem__(self,batch_id):
        if ((batch_id+1)*self.batch_size > self.x.shape[0]):
            X = np.take(self.x,range(batch_id*self.batch_size,self.x.shape[0]-1), axis=0)
            y = np.take(self.y,range(batch_id*self.batch_size,self.x.shape[0]-1), axis=0)
        else:
            X = np.take(self.x,range(batch_id*self.batch_size,(batch_id+1)*self.batch_size), axis=0)
            y = np.take(self.y,range(batch_id*self.batch_size,(batch_id+1)*self.batch_size), axis=0)
        seq_length = len(X[0])
        y = encode_structure_array(y).reshape(y.shape[0]*seq_length,1)
        X = encode_seq_array_windows(X,self.window_size)

        return X,y

class DataSliceFromInterimWindows(Sequence):
    def __init__(self,batch_size,directory,dataset_size,window_size):
        self.y = np.load(os.path.join(directory,"simulated_structures.npy"),mmap_mode="r")[0:dataset_size]
        self.x = np.load(os.path.join(directory,"simulated_sequences.npy"),mmap_mode="r")[0:dataset_size]
        self.window_size = window_size
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
        l = int(self.x.shape[0] / self.batch_size)
        if l*self.batch_size < self.x.shape[0]:
            l += 1
        return l

    def __getitem__(self,batch_id):
        if ((batch_id+1)*self.batch_size > self.x.shape[0]):
            X = np.take(self.x,range(batch_id*self.batch_size,self.x.shape[0]-1), axis=0)
            y = np.take(self.y,range(batch_id*self.batch_size,self.x.shape[0]-1), axis=0)
        else:
            X = np.take(self.x,range(batch_id*self.batch_size,(batch_id+1)*self.batch_size), axis=0)
            y = np.take(self.y,range(batch_id*self.batch_size,(batch_id+1)*self.batch_size), axis=0)
        seq_length = len(X[0])
        y = encode_structure_array(y).reshape(y.shape[0]*seq_length,1)
        X = encode_seq_array_windows(X,self.window_size)

        return X,y
