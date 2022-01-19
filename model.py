from tensorflow.keras.layers import Activation, Dropout
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D
from tensorflow.keras.layers import Input, concatenate, BatchNormalization
from tensorflow.keras.models import Model


def conv_block(inputs, filters, activation, batch_norm):
    hidden = Conv2D(filters, (3, 3), padding='same')(inputs)
    hidden = BatchNormalization()(hidden) if batch_norm else hidden
    hidden = Activation(activation)(hidden)
    hidden = Conv2D(filters, (3, 3), padding='same')(hidden)
    hidden = BatchNormalization()(hidden) if batch_norm else hidden
    hidden = Activation(activation)(hidden)
    hidden = Conv2D(filters, (3, 3), padding='same')(hidden)
    hidden = BatchNormalization()(hidden) if batch_norm else hidden
    outputs = Activation(activation)(hidden)
    return outputs


def pool_block(inputs, filters, activation, batch_norm, drop):
    hidden = MaxPooling2D(pool_size=(2, 2))(inputs)
    hidden = Dropout(drop)(hidden) if drop else hidden
    outputs = conv_block(hidden, filters, activation, batch_norm)
    return outputs


def conv_trans_block(inputs, layer, filters, activation, batch_norm, drop):
    hidden = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(inputs)
    hidden = concatenate([hidden, layer], axis=3)
    hidden = Dropout(drop)(hidden) if drop else hidden
    outputs = conv_block(hidden, filters, activation, batch_norm)
    return outputs


def get_model(image_shape, filters, depth, inc_rate, activation, drop, batch_norm):
    inputs = Input(shape=image_shape)
    layers = []
    hidden = conv_block(inputs, filters, activation, batch_norm)
    layers.append(hidden)
    for index in range(depth):
        filters = filters*inc_rate
        hidden = pool_block(hidden, filters, activation, batch_norm, drop)
        if index != depth-1:
            layers.append(hidden)
    for index in range(depth):
        filters = filters//inc_rate
        hidden = conv_trans_block(hidden, layers[-1-index], filters, activation, batch_norm, drop)
    outputs = Conv2D(2, (1, 1), activation='softmax')(hidden)
    model = Model(inputs=inputs, outputs=outputs)
    return model
