import os
import numpy as np
import imageio
import keras
import cv2
import matplotlib.pyplot as plt
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint,EarlyStopping, ReduceLROnPlateau
from utilities.datareader import datareader
from model.iris_detetction_model import irisAttention

dtreader = datareader(train_state=True)
valdtreader = datareader(train_state=False)

model = irisAttention()

model.summary()

ckpt_path = r'ckpt/irisAttention_CASIA-iris-distance.h5'

class loss_history(keras.callbacks.Callback):
    def __init__(self, x=0):
        self.x = x
    def on_epoch_begin(self, epoch, logs={}):

        bbox = self.model.predict(np.expand_dims(valdtreader.images[self.x], axis=0))
        bbox = np.squeeze(bbox)
        bbox = np.squeeze(bbox)
        bbox = np.squeeze(bbox)

        bbox[0] = bbox[0] * 256
        bbox[1] = bbox[1] * 256
        bbox[2] = bbox[2] * 256
        bbox[3] = bbox[3] * 256
        image = imageio.imread(os.path.join(r'dataset/images/'+valdtreader.imagelist[self.x].split(" ")[0]+'.jpeg'),as_gray=False, pilmode="RGB")
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
        cv2.rectangle(image,
                      (int(valdtreader.bboxes[self.x, 0, 0, 0] * 256), int(valdtreader.bboxes[self.x, 0, 0, 1] * 256)),
                      (int(valdtreader.bboxes[self.x, 0, 0, 2] * 256), int(valdtreader.bboxes[self.x, 0, 0, 3] * 256)),
                      (0, 255, 0),
                      2)
        plt.imshow(image)
        plt.show()

model.compile(optimizer=Adam(lr=0.0001), loss=['mse'], metrics=['accuracy'])

if os.path.exists(ckpt_path):
    model.load_weights(ckpt_path)
    print('the checkpoint is loaded successfully.')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1 ,patience=3 ,verbose=1, min_lr=0.00001)
earlystopper = EarlyStopping(patience=7, verbose=1)
checkpointer = ModelCheckpoint(ckpt_path,verbose=1,save_best_only=True)

r = model.fit(x=dtreader.images,
              y=[dtreader.bboxes],
              validation_data=(valdtreader.images,[ valdtreader.bboxes]),
              callbacks=[loss_history(), checkpointer, earlystopper,reduce_lr],
              epochs=40,
              verbose=1,
              batch_size=2,
              shuffle=True)

plt.plot(r.history['loss'])
plt.plot(r.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('epochs')
plt.legend(['Training','Validation'],loc='upper left')
plt.savefig(r'statistic\ISUR\irisAttention\CASIA-iris-distance\loss_irisAttention_CASIA-iris-distance.jpeg')
plt.show()

plt.plot(r.history['acc'])
plt.plot(r.history['val_acc'])
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epochs')
plt.legend(['Training','Validation'],loc='upper left')
plt.savefig(r'statistic\ISUR\irisAttention\CASIA-iris-distance\Accuracy_irisAttention_CASIA-iris-distance.jpeg')
plt.show()

np.save(r'statistic\ISUR\irisAttention\CASIA-iris-distance\irisAttention_CASIA-iris-distance.npy',r.history)
