from keras.layers import GlobalAveragePooling2D, Dense, Multiply
import keras.backend as K

def se_block(tensor, ratio=16):

    channel_axis = -1  # Since we are using Tensorflow
    filters = K.int_shape(tensor)[channel_axis]

    x = GlobalAveragePooling2D()(tensor)
    x = Dense(filters // ratio, activation='relu', kernel_initializer = "he_normal", use_bias = False)(x)
    x = Dense(filters, activation='sigmoid', kernel_initializer = "he_normal", use_bias = False)(x)

    x = Multiply()([tensor, x])
    
    return x
