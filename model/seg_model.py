from model.resnet18 import resnet18
from model.squeezeexcitation import se_block
from keras.layers import *
from keras.models import *

def proposedNetwork(height=256, width = 256, num_class = 1):

    img_input = Input(shape=(height, width, 3), name='data')
    attentionMask = Input(shape=(height, width, 1))
    att = MaxPooling2D()(attentionMask)

    f1, f2, f3, f4, f5 = resnet18(img_input)

    x = MaxPooling2D(padding='same')(f5)

    x = UpSampling2D(size=2)(x)
    x = Conv2D(512, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(512, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = se_block(x)
    skip1 = x
    x = Concatenate()([x, f5])

    x = UpSampling2D()(x)
    x = Conv2D(256, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = se_block(x)
    skip2 = x
    x = Concatenate()([x, f4])

    x = UpSampling2D()(x)
    x = Conv2D(128, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = se_block(x)
    skip3 = x
    x = Concatenate()([x, f3])

    x = UpSampling2D()(x)
    x = Conv2D(64, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = se_block(x)
    skip4 = x
    x = Concatenate()([x, f2])

    x = UpSampling2D()(x)
    x = Conv2D(64, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    skip5 = x
    x = Multiply()([x, att])

    x = UpSampling2D()(x)
    x = Conv2D(32, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    skip6 = x

    skip4 = Conv2D(32, 1)(skip4)
    skip4 = UpSampling2D(4)(skip4)

    skip5 = Conv2D(32, 1)(skip5)
    skip5 = UpSampling2D(2)(skip5)

    x = Concatenate()([skip4, skip5, skip6])

    output = Conv2D(num_class, (1, 1),activation='sigmoid', padding='same')(x)

    model = Model(inputs=[img_input,attentionMask], outputs=[output])

    return model
