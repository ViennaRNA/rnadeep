import os
import numpy as np

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras.layers import Input,LSTM, Dense, Bidirectional, Dropout,TimeDistributed,LeakyReLU 
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import concatenate,Add

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
tf.get_logger().setLevel('FATAL')

def blstm(lstm_layers = 1,lstm_neurons = 20):
    input = Input(shape=(None,4))
    #input_mask = Input(shape=(None,1))
    #mask = K.reshape(input_mask,(K.shape(input_mask)[0],K.shape(input_mask)[1],1))

    #LSTM blocks
    t= Bidirectional(LSTM(lstm_neurons,return_sequences=True))(input)
    for j in range(lstm_layers-1):
        t = Bidirectional(LSTM(lstm_neurons,return_sequences=True))(t)
    output = TimeDistributed(Dense(1, activation='sigmoid',use_bias=True))(t)
    model = Model(inputs=input, outputs=output)
    return model

def complex_blstm(lstm_layers = 1,lstm_neurons = 40):
    input = Input(shape=(None,4))
    t= Bidirectional(LSTM(lstm_neurons,return_sequences=True))(input)
    for j in range(lstm_layers-1):
        t = Bidirectional(LSTM(lstm_neurons,return_sequences=True))(t)
    t = LSTM(int(lstm_neurons/2),return_sequences=True)(t)
    dense_1 = TimeDistributed(Dense(int(lstm_neurons/4)))(t)
    LR = LeakyReLU(alpha=0.1)(dense_1)
    output = TimeDistributed(Dense(1, activation='sigmoid',use_bias=True))(LR)
    model = Model(inputs=input, outputs=output)
    return model
