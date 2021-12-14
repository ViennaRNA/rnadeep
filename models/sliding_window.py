#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals

from tensorflow.keras import backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense,Dropout, Conv1D,BatchNormalization,Input,LeakyReLU
import tensorflow as tf
import numpy as np

def basic_window(window_size):
    m = Sequential()
    m.add(Input(shape=(window_size*2+1,4)))
    m.add(Flatten())
    m.add(Dense(128, activation='relu'))
    m.add(Dropout(0.5))
    m.add(Dense(32, activation='relu'))
    m.add(Dense(1, activation = 'sigmoid'))
    return m
    
def basic_window_leakyrelu(window_size):
    m = Sequential()
    m.add(Input(shape=(window_size*2+1,4)))
    m.add(Flatten())
    m.add(Dense(128))
    m.add(LeakyReLU(alpha=0.05))
    m.add(Dropout(0.5))
    m.add(Dense(32))
    m.add(LeakyReLU(alpha=0.05))
    m.add(Dense(1, activation = 'sigmoid'))
    return m

def conv_window(window_size):
    m = Sequential()
    m.add(Input(shape=(window_size*2+1,4)))
    m.add(Conv1D(128, 5, padding='same', activation='relu'))
    m.add(BatchNormalization())
    m.add(Dropout(0.5))
    m.add(Conv1D(128, 3, padding='same', activation='relu'))
    m.add(BatchNormalization())
    m.add(Dropout(0.5))
    m.add(Conv1D(64, 3, padding='same', activation='relu'))
    m.add(BatchNormalization())
    m.add(Dropout(0.5))
    m.add(Flatten())
    m.add(Dense(128, activation='relu'))
    m.add(Dropout(0.5))
    m.add(Dense(32, activation='relu'))
    m.add(Dense(1, activation = 'sigmoid'))
    return m 
    
    