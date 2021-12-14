#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from encoding_utils_rna import encode_matrix_array,encode_onehot_matrix_array,encode_padded_matrix_array,encode_padded_onehot_matrix_array
import os

class DataMatrixFromInterimFile(Sequence):
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
        y = encode_matrix_array(y).reshape(y.shape[0],seq_length,seq_length,1)
        X = encode_onehot_matrix_array(X).reshape(X.shape[0],seq_length,seq_length,8)
        return X,y
        
class PaddedDataMatrixFromInterimFile(Sequence):
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
        seq_length = len(max(X, key=len))
        y = encode_padded_matrix_array(y, seq_length)
        x1,x2 = encode_padded_onehot_matrix_array(X, seq_length)
        X = [x1,x2]
        return X,y