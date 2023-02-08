import os
import numpy as np
import tqdm
import keras
import matplotlib.pyplot as plt
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint,EarlyStopping, ReduceLROnPlateau
from utilities.datareader import datareader
from model.seg_model import proposedNetwork
from model.lossesfunc import DiceBCELoss

dtreader = datareader(train_state=True)
valdtreader = datareader(train_state=False)

model = proposedNetwork()

model.summary()

ckpt_path = r'ckpt/PN_CASIA_distance_batch2.h5'

class loss_history(keras.callbacks.Callback):
    def __init__(self, x=4):
        self.x = x
    def on_epoch_begin(self, epoch, logs={}):

        pred = self.model.predict([np.expand_dims(valdtreader.images[self.x], axis=0)],verbose=1)
                                   np.expand_dims(valdtreader.AttMasks[self.x], axis=0)],verbose=1)
        pred = np.squeeze(pred)
        pred = (pred>0.5).astype(np.uint8)
        plt.imshow(pred)
        plt.show()

        Ei = 0
        n = 256 * 256
        for i in tqdm.trange(valdtreader.num):
            image, mask = np.expand_dims(valdtreader.images[i], axis=0), valdtreader.masks[i]
            mask = np.squeeze(mask)
            attMask = np.expand_dims(valdtreader.AttMasks[i], axis=0)
            pred = model.predict([image, attMask])
            pred = np.squeeze(pred)
            pred = (pred > 0.5).astype(np.int32)

            r = int(np.sum(pred ^ mask)) / n

            Ei += r

        E = Ei / valdtreader.num
        print('----Error rate : ', np.round(E, 6), '----')

model.compile(optimizer=Adam(lr=0.0001), loss=[DiceBCELoss], metrics=['accuracy'])

if os.path.exists(ckpt_path):
    model.load_weights(ckpt_path)
    print('the checkpoint is loaded successfully.')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1 ,patience=3 ,verbose=1, min_lr=0.00001)
earlystopper = EarlyStopping(patience=7, verbose=1)
checkpointer = ModelCheckpoint(ckpt_path,verbose=1,save_best_only=True)

r = model.fit(x=[dtreader.images, dtreader.AttMasks],
              y=[dtreader.masks],
              validation_data=([valdtreader.images, valdtreader.AttMasks], [valdtreader.masks]),
              callbacks=[loss_history(), earlystopper, reduce_lr],
              epochs=100,
              verbose=1,
              batch_size=2,
              shuffle=True)

plt.plot(r.history['loss'])
plt.plot(r.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('epochs')
plt.legend(['Training','Validation'],loc='upper left')
plt.savefig(r'statistic\ISUR\seg_model\CASIA_distance\loss_seg_model_CASIA_distance_batch2.jpeg')
plt.show()

plt.plot(r.history['acc'])
plt.plot(r.history['val_acc'])
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epochs')
plt.legend(['Training','Validation'],loc='upper left')
plt.savefig(r'statistic\ISUR\seg_model\CASIA_distance\Accuracy_seg_model_CASIA_distance_batch2.jpeg')
plt.show()

np.save(r'statistic\ISUR\seg_model\CASIA_distance\seg_model_CASIA_distance_batch2.npy',r.history)
