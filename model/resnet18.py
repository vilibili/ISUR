from keras.layers import *
from model.squeezeexcitation import se_block

def identity_block(input_tensor, kernel_size, filters, stage, block):
    conv_name_base = 'res' + str(stage) + block + '_branch'

    x = Conv2D(filters, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = Conv2D(filters, (kernel_size, kernel_size), padding='same',name=conv_name_base + '2b')(x)
    x = BatchNormalization()(x)

    x = se_block(x)

    x = Add(name='res' + str(stage) + block)([x, input_tensor])
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):

    conv_name_base = 'res' + str(stage) + block + '_branch'

    x = Conv2D(filters, (kernel_size, kernel_size), padding='same', strides=strides,
                      name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = Conv2D(filters, (kernel_size, kernel_size), padding='same',
                      name=conv_name_base + '2b')(x)
    x = BatchNormalization()(x)
    x = se_block(x)

    shortcut = Conv2D(filters, (1, 1), padding='same', strides=strides,
                             name=conv_name_base + '1')(input_tensor)
    #shortcut = BatchNormalization()(shortcut)

    x = Add(name='res' + str(stage) + block)([x, shortcut])
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)
    return x

def resnet18(img_input):

    base_filter = 64

    x = Conv2D(base_filter, (7, 7), strides=(2, 2), name='conv1', padding='same')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu', name='conv1_relu')(x)
    f1 = x
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1', padding='same')(x)

    x = conv_block(x, 3, base_filter, stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, base_filter, stage=2, block='b')
    f2 = x

    x = conv_block(x, 3, base_filter * 2, stage=3, block='a')
    x = identity_block(x, 3, base_filter * 2, stage=3, block='b')
    f3 = x

    x = conv_block(x, 3, base_filter * 4, stage=4, block='a')
    x = identity_block(x, 3, base_filter * 4, stage=4, block='b')
    f4 = x

    x = conv_block(x, 3, base_filter * 8, stage=5, block='a')
    x = identity_block(x, 3, base_filter * 8, stage=5, block='b')
    f5 = x

    return f1, f2, f3, f4, f5
