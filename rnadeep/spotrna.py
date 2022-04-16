#!/usr/bin/env python


import os
import numpy as np

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras.layers import Input, Lambda, Conv2D, LSTM, Dense, Flatten,TimeDistributed, Bidirectional,ELU, BatchNormalization, Dropout, multiply
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras import backend as K
from tensorflow.keras.layers import concatenate,Add

from .metrics import mcc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
tf.get_logger().setLevel('FATAL')

def flatten_diagonally(a):
  result = []
  diagonals = range(-a.shape[0] + 1, a.shape[1])
  for i in diagonals:
    x = diagonals[i]
    result.append(np.diagonal(a, offset=x, axis1=0, axis2=1).copy())
  return np.concatenate(result,axis=1)

def flatten_array(a):
  x = map(flatten_diagonally, a)
  return np.array(list(x), dtype=np.float32)

def my_lambda_func(x):
    return tf.numpy_function(flatten_array,[x],tf.float32)

def relu_bn(inputs: Tensor) -> Tensor:
    elu = ELU()(inputs)
    bn = BatchNormalization()(elu)
    return bn

def residual_block(x: Tensor, filters: int, dilation: float) -> Tensor:
    y = Dropout(0.25)(x)
    y = Conv2D(kernel_size=5, strides=1, filters=filters, padding="same",dilation_rate=dilation)(y)
    y = relu_bn(y)
    y = Dropout(0.25)(y)
    y = Conv2D(kernel_size=3, strides=1, filters=filters, padding="same",dilation_rate=dilation)(y)
    out = Add()([x, y])
    out = relu_bn(out)
    return out

def lstm_block(x: Tensor, neurons: int, third_dim: int) -> Tensor:
    horizontal = K.reshape(x,(K.shape(x)[0],K.shape(x)[1]**2,third_dim))
    hori_lstm = Bidirectional(LSTM(neurons, return_sequences=True))(horizontal)
    hori_lstm = K.reshape(hori_lstm,(K.shape(x)[0],K.shape(x)[1],K.shape(x)[2],neurons*2))

    vertical = K.reshape(tf.transpose(x),(K.shape(x)[0],K.shape(x)[1]**2,third_dim))
    verti_lstm = Bidirectional(LSTM(neurons,return_sequences=True))(vertical)
    verti_lstm = K.reshape(verti_lstm,(K.shape(x)[0],K.shape(x)[1],K.shape(x)[2],neurons*2))
    merge = concatenate([hori_lstm, verti_lstm])
    return merge
  
def diagonal_lstm_block(x: Tensor, neurons: int, third_dim: int) -> Tensor:
    horizontal = K.reshape(x,(K.shape(x)[0],K.shape(x)[1]**2,third_dim))
    hori_lstm = Bidirectional(LSTM(neurons, return_sequences=True))(horizontal)
    hori_lstm = K.reshape(hori_lstm,(K.shape(x)[0],K.shape(x)[1],K.shape(x)[2],neurons*2))

    vertical = K.reshape(K.transpose(x),(K.shape(x)[0],K.shape(x)[1]**2,third_dim))
    verti_lstm = Bidirectional(LSTM(neurons,return_sequences=True))(vertical)
    verti_lstm = K.reshape(verti_lstm,(K.shape(x)[0],K.shape(x)[1],K.shape(x)[2],neurons*2))
    
   
    diagonal = Lambda(my_lambda_func)(x)
    diagonal = K.reshape(diagonal,(K.shape(x)[0],K.shape(x)[1]**2,third_dim))
    diagonal_lstm = Bidirectional(LSTM(neurons,return_sequences=True))(diagonal)
    diagonal_lstm = K.reshape(diagonal_lstm,(K.shape(x)[0],K.shape(x)[1],K.shape(x)[2],neurons*2))
    
    merge = concatenate([hori_lstm, verti_lstm])
    merge = concatenate([merge,diagonal_lstm])
    return merge

def spotrna(model=0,use_mask = False):
    #Parameters
    if (model == 0):
        initial_num_filters = block_num_filters = 48
        num_resnet_blocks = 16
        dilation = 1 #no dilation
        num_lstm_blocks = 0
        lstm_neurons = 0
        num_fc_layers = 2
        fc_neurons = 512
    elif (model == 1):
        initial_num_filters = block_num_filters = 64
        num_resnet_blocks = 20
        dilation = 1 #no dilation
        num_lstm_blocks = 0
        lstm_neurons = 0
        num_fc_layers = 2
        fc_neurons = 512
    elif (model == 2):
        initial_num_filters = block_num_filters = 64
        num_resnet_blocks = 30
        dilation = 1 #no dilation
        num_lstm_blocks = 0
        lstm_neurons = 0
        num_fc_layers = 2
        fc_neurons = 512
    elif (model == 3):
        initial_num_filters = block_num_filters = 64
        num_resnet_blocks = 30
        dilation = 1 #no dilation
        num_lstm_blocks = 1
        lstm_neurons = 200
        num_fc_layers = 0
        fc_neurons = 0
    elif (model == 4):
        initial_num_filters = block_num_filters = 64
        num_resnet_blocks = 30
        dilation = 2**(block_num_filters%5)
        num_lstm_blocks = 0
        lstm_neurons = 0
        num_fc_layers = 2
        fc_neurons = 512
    elif (model == 5):
        initial_num_filters = block_num_filters = 64
        num_resnet_blocks = 15
        dilation = 1
        num_lstm_blocks = 1
        lstm_neurons = 100
        num_fc_layers = 0
        fc_neurons = 0
    elif (model == 6):
        initial_num_filters = block_num_filters = 10
        num_resnet_blocks = 3
        dilation = 1
        num_lstm_blocks = 1
        lstm_neurons = 20
        num_fc_layers = 1
        fc_neurons = 10

    #Input is matrix of one hot encodings
    input = Input(shape=(None,None,8))

    #ResNet
    #Initial 3x3 convolution
    t = Conv2D(kernel_size=3, strides=1, filters=initial_num_filters, padding="same",dilation_rate=dilation)(input)
    #Activation and normalisation
    t = relu_bn(t)
    #Resnet block
    for j in range(num_resnet_blocks):
        t = residual_block(t, filters=block_num_filters,dilation=dilation)

    #activation and normalisation
    t = relu_bn(t)
    if (num_lstm_blocks>0):
        #no bottleneck used in SPOT RNA
        #t = Conv2D(kernel_size=1, strides=1, filters= bottleneck_num_filters, padding="same")(t)
        #activation and normalisation
        #t = relu_bn(t)

        #LSTM blocks
        t = lstm_block(t, neurons=lstm_neurons, third_dim=block_num_filters)
        for j in range(num_lstm_blocks-1):
            t = lstm_block(t, neurons=lstm_neurons,third_dim= lstm_neurons*4)

    #FC layers
    for j in range(num_fc_layers):
        t = Dense(fc_neurons, activation=ELU())(t)

    #Dropout
    t = Dropout(0.5)(t)

    if use_mask:
        input_mask = Input(shape=(None,None))
        mask = K.reshape(input_mask,(K.shape(input_mask)[0],K.shape(input_mask)[1],K.shape(input_mask)[2],1))
        merge = multiply([t, mask])
        output = Dense(1, activation='sigmoid',use_bias=True)(merge)
        model = Model(inputs=[input,input_mask], outputs=output)
    else:
        output = Dense(1, activation='sigmoid',use_bias=True)(t)
        model = Model(inputs=input, outputs=output)
    return model

