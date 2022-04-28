#!/usr/bin/env python

import os
import numpy as np

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras.layers import Input, Lambda, Conv2D, Dense, Flatten,TimeDistributed, Bidirectional,ELU, BatchNormalization, Dropout, multiply
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras import backend as K
from tensorflow.keras.layers import concatenate,Add

from .metrics import mcc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
tf.get_logger().setLevel('FATAL')

def relu_bn(inputs: Tensor) -> Tensor:
    elu = ELU()(inputs)
    bn = BatchNormalization()(elu)
    return bn

def spotrna():

    # Parameters
    initial_num_filters = 48
    block_num_filters = 48
    num_resnet_blocks = 16
    num_fc_layers = 2
    fc_neurons = 512

    # Input is matrix of one hot encodings
    inputs = Input(shape = (None, None, 8), dtype = "float32") 

    # Reduce arbitrary input to fixed size K
    t = Conv2D(filters = initial_num_filters, 
               kernel_size = 3, 
               padding = "same")(inputs)

    print(t.shape())

    ## Activation and normalisation
    #t = relu_bn(t)
    ## Resnet block
    #for j in range(num_resnet_blocks):
    #    t = residual_block(t, filters=block_num_filters)

    ## Activation and normalisation
    #t = relu_bn(t)

    ## FC layers
    #for j in range(num_fc_layers):
    #    t = Dense(fc_neurons, activation=ELU())(t)

    ## Dropout
    #t = Dropout(0.5)(t)

    #if use_mask:
    #    input_mask = Input(shape=(None,None))
    #    mask = K.reshape(input_mask,(K.shape(input_mask)[0],K.shape(input_mask)[1],K.shape(input_mask)[2],1))
    #    merge = multiply([t, mask])
    #    output = Dense(1, activation='sigmoid',use_bias=True)(merge)
    #    model = Model(inputs=[input,input_mask], outputs=output)
    #else:
    #    output = Dense(1, activation='sigmoid',use_bias=True)(t)
    #    model = Model(inputs=input, outputs=output)
    #return model

