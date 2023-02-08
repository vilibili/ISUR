import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import tqdm
from scipy import io
import random

class datareader:
    def __init__(self,
                 image_path = r'dataset/images',
                 mask_path = r'dataset/masks',
                 train_file = r'dataset/CASIA-iris-distance_detection_train.txt',
                 val_file = r'dataset/CASIA-iris-distance_detection_val.txt',
                 train_state = True
                 ):
        self.train_state = train_state

        if train_state :
            path = train_file
            self.attPath = r'dataset/irisAtten'
        else:
            path = val_file
            self.attPath = r'dataset/irisAtten'
        self.image_path = image_path
        self.mask_path = mask_path
        self.imagelist = self.get_image_list(path)
        self.num = len(self.imagelist)
        print('number of images: ', self.num)
        self.cur_index = 0
        self.images, self.masks, self.bboxes, self.AttMasks = self.get_Images()

    def get_image_list(self, path):
            f = open(path,'r')
            text = f.readlines()
            f.close()
            imagelist = []
            for name in text:
                name = name.replace('\n', '')
                imagelist.append(name)
                if os.path.exists(os.path.join(self.image_path, name + '.jpeg')):
                    imagelist.append(name)
                else:
                    print('image is not found.', os.path.join(self.image_path, name + '.jpeg'))
                if os.path.exists(os.path.join(self.mask_path, name + '.png')) is False:
                    print('mask is not found.', os.path.join(self.mask_path, name + '.png'))
            return imagelist

    def read_data(self):

        name = self.imagelist[self.cur_index].split(" ")[0]
        bbox = self.imagelist[self.cur_index].split(" ")[1:]
        bbox = np.expand_dims(bbox, axis=0)
        bbox = np.expand_dims(bbox, axis=0)
        image = imageio.imread(os.path.join(self.image_path, name + '.jpeg'),as_gray=False, pilmode="RGB")
        #image = np.expand_dims(image, axis=2)

        mask = imageio.imread(os.path.join(self.mask_path, name + '.png'))

        AttMask = io.loadmat(os.path.join(self.attPath, name + '.mat'))['mydata']

        if self.train_state:
            mask[mask == 255] = 1
            mask = np.expand_dims(mask,axis=2)
            return image, mask, bbox, AttMask
        else:
            mask[mask == 255] = 1
            mask = np.expand_dims(mask, axis=2)
            return image, mask, bbox, AttMask

    def get_Images(self):
        images = []
        masks = []
        bboxes = []
        AttMasks = []
        for _ in tqdm.trange(self.num):
            image, mask, bbox, AttMask = self.read_data()
            images.append(image)
            masks.append(mask)
            bboxes.append(bbox)
            AttMasks.append(AttMask)
            self.cur_index += 1
        images = np.array(images)#dtype=unit8
        masks = np.array(masks)
        bboxes = np.array(bboxes).astype(float)/256
        AttMasks = np.array(AttMasks)
        return images, masks, bboxes, AttMasks
