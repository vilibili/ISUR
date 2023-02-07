from keras import backend as K
from keras.losses import categorical_crossentropy, binary_crossentropy

def DiceLoss(y_true, y_pred, smooth=1e-15):

    # flatten label and prediction tensors
    y_pred = K.flatten(y_pred)
    y_true = K.flatten(y_true)

    intersection = K.sum(y_true * y_pred)
    dice_loss = (2 * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)

    return 1 - dice_loss

def DiceBCELoss(y_true, y_pred):
    loss = 0.7 * binary_crossentropy(y_true, y_pred)

    dice = DiceLoss(y_true, y_pred)

    loss += (0.3 * dice)
    return loss
