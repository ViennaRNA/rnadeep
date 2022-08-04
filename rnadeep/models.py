
import tensorflow as tf
from tensorflow.keras import Input, Model, layers

def spotrna_models(model = 1, use_mask = True):
    """ Some modifications to Julia's SPOT-RNA implementations.

    Supposed to be a reimplementation of the models in the 
    SPOT-RNA paper. If you find mistakes, please let us know!

    Overview:
        - Initial 3x3 convolution layer
        - ResNet blocks
        - Act./Norm.
        - 2D-BLSTM
        - Fully Connected blocks
        - Output masking layer (optional)
        - Output layer

    Args:
        model: select the model (0-4)
        use_mask: for padded input/output (defaults to True!)
    
    """
    if model == 0:
        resnet_blocks = 16
        rn_filters = 48
        rn_dilation = 1 # no dilation
        lstm_blocks = 0
        lstm_neurons = 0
        fc_blocks = 2
        fc_neurons = 512
    elif model == 1:
        resnet_blocks = 20
        rn_filters = 64
        rn_dilation = 1 # no dilation
        lstm_blocks = 0
        lstm_neurons = 0
        fc_blocks = 1
        fc_neurons = 512
    elif (model == 2):
        resnet_blocks = 30
        rn_filters = 64
        rn_dilation = 1 # no dilation
        lstm_blocks = 0
        lstm_neurons = 0
        fc_blocks = 1
        fc_neurons = 512
    elif (model == 3):
        resnet_blocks = 30
        rn_filters = 64
        rn_dilation = 1 # no dilation
        lstm_blocks = 1
        lstm_neurons = 200
        fc_blocks = 0
        fc_neurons = 0
    elif (model == 4):
        resnet_blocks = 30
        rn_filters = 64
        rn_dilation = 2**(rn_filters%5)
        lstm_blocks = 0
        lstm_neurons = 0
        fc_blocks = 1
        fc_neurons = 512

    inputs = Input(shape = (None, None, 8), dtype = "float32")

    # ResNet -- Initial 3x3 convolution
    x = layers.Conv2D(rn_filters, 3, padding = "same", dilation_rate = rn_dilation)(inputs)

    # ResNet blocks (Block A)
    for _ in range(resnet_blocks):
        # Activation and normalisation
        y = layers.ELU()(x)
        y = layers.BatchNormalization()(y)
        y = layers.Dropout(0.25)(y)
        y = layers.Conv2D(rn_filters, 3, padding="same", dilation_rate = rn_dilation)(y)
        y = layers.ELU()(y)
        y = layers.BatchNormalization()(y)
        y = layers.Dropout(0.25)(y)
        y = layers.Conv2D(rn_filters, 5, padding="same", dilation_rate = rn_dilation)(y)
        # Now add the layers
        x = layers.Add()([x, y])

    x = layers.ELU()(x)
    x = layers.BatchNormalization()(x)

    # 2D-BLSTM
    if lstm_blocks:
        # LSTM input must be 3D: (batch, timesteps, feature)
        (d1, d2, d3, d4) = tf.shape(x)
        lstm = layers.LSTM(lstm_neurons, return_sequences = True)
        for _ in range(lstm_blocks):
            # Horizontal pass
            xh = tf.reshape(x, (d1, d2*d3, d4))
            yh = layers.Bidirectional(lstm)(xh)
            yh = tf.reshape(yh, (d1, d2, d3, lstm_neurons*2))
            # Vertical pass
            xv = tf.reshape(tf.transpose(x, perm=[0, 2, 1, 3]), (d1, d2*d3, d4))
            yv = layers.Bidirectional(lstm)(xv)
            yv = tf.reshape(yv, (d1, d2, d3, lstm_neurons*2))
            # Combine passes
            x = layers.concatenate([yh, yv]) # Would the mean make more sense?
            (d1, d2, d3, d4) = tf.shape(x)

    # Fully connected blocks (Block B)
    for _ in range(fc_blocks):
        x = layers.Dense(fc_neurons, activation = layers.ELU())(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)

    if use_mask:
        mask = Input(shape = (None, None, 1))
        inputs = [inputs, mask]
        x = layers.multiply([x, mask])

    outputs = layers.Dense(1, activation='sigmoid', use_bias = True)(x)
    return Model(inputs = inputs, outputs = outputs)

