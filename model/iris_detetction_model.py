from model.resnet18 import resnet18
from keras.layers import *
from keras.models import *

def irisAttention(height=256, width = 256):
    img_input = Input(shape=(height, width, 3), name='data')
    f1, f2, f3, f4, f5 = resnet18(img_input)

    boundingbox = f5
    boundingbox = MaxPooling2D(strides=(8,8))(boundingbox)
    boundingbox = Conv2D(512, (3, 3), padding='same')(boundingbox)
    boundingbox = Activation('relu')(boundingbox)
    boundingbox = Conv2D(256, (3, 3), padding='same')(boundingbox)
    boundingbox = Activation('relu')(boundingbox)
    boundingbox = Conv2D(4, (1,1), padding='same')(boundingbox)
    boundingbox = Activation('sigmoid')(boundingbox)

    output = boundingbox

    model = Model(inputs=img_input, outputs=output)

    return model
