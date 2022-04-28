
from tensorflow import keras
from tensorflow.keras import layers

def spotrna(model = 1, use_mask = False):
    """ A copy from Julia's spotrna, 
    but only model 1
    """
    assert model == 1
    filters = 48
    resnet_blocks = 16
    lstm_blocks = 0
    # lstm_neurons = 0
    fc_layers = 2
    fc_neurons = 512
 
    inputs = keras.Input(shape = (None, None, 8), dtype = "float32")

    # ResNet -- Initial 3x3 convolution
    x = layers.Conv2D(filters, 3, padding = "same")(inputs)
    # Activation and normalisation
    x = layers.ELU()(x)
    x = layers.BatchNormalization()(x)

    # ResNet blocks
    for _ in range(resnet_blocks):
        y = layers.Dropout(0.25)(x)
        y = layers.Conv2D(filters, 5, padding="same")(y)
        y = layers.ELU()(y)
        y = layers.BatchNormalization()(y)

        y = layers.Dropout(0.25)(y)
        y = layers.Conv2D(filters, 3, padding="same")(y)
        x = layers.Add()([x, y])
        x = layers.ELU()(x)
        x = layers.BatchNormalization()(x)

    assert lstm_blocks == 0
    #x = layers.Bidirectional(layers.LSTM(lstm_neurons, return_sequences=True))(x)
    #for _ in range(lstm_blocks):

    # Fully connected layers
    for _ in range(fc_layers):
        x = layers.Dense(fc_neurons, activation = layers.ELU())(x)

    x = layers.Dropout(0.5)(x)

    if use_mask:
        mask = keras.Input(shape = (None, None, 1))
        inputs = [inputs, mask]
        x = layers.multiply([x, mask])

    outputs = layers.Dense(1, activation='sigmoid', use_bias = True)(x)
    model = keras.Model(inputs = inputs, outputs = outputs)
    print(model.summary())
    return model

