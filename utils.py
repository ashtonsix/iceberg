import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from img_preprocessing import random_rotation, random_translation, random_zoom, random_channel_shift, flip_axis

import keras.backend as K
from keras.models import Sequential, load_model
from keras.layers import Conv2D, GlobalMaxPooling2D, MaxPooling2D, AveragePooling2D, Dense, Dropout, Input, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import Adam, SGD

def show_imgs(imgs, labels=None, figsize=(32, 6)):
    f = plt.figure(figsize=figsize)
    rows = 2
    cols = len(imgs)

    for i in range(len(imgs)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if labels is not None:
            sp.set_title(labels[i], fontsize=16)
        plt.imshow(imgs[i,:,:,0])
    
    for i in range(len(imgs)):
        sp = f.add_subplot(rows, cols, cols+i+1)
        sp.axis('Off')
        if labels is not None:
            sp.set_title(labels[i], fontsize=16)
        plt.imshow(imgs[i,:,:,1])

def Inception(input, filters, activation='elu'):
    tower_1 = Conv2D(filters, (1, 1), padding='same', activation=activation)(input)
    tower_1 = Conv2D(filters, (3, 3), padding='same', activation=activation)(tower_1)

    tower_2 = Conv2D(filters, (1, 1), padding='same', activation=activation)(input)
    tower_2 = Conv2D(filters, (5, 5), padding='same', activation=activation)(tower_2)

    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input)
    tower_3 = Conv2D(filters, (1, 1), padding='same', activation=activation)(tower_3)
    
    tower_4 = Conv2D(filters, (3, 3), padding='same', activation=activation)(input)
    tower_4 = MaxPooling2D((2, 2), strides=(1, 1), padding='same')(tower_4)

    output = Concatenate(axis=3)([tower_1, tower_2, tower_3, tower_4])

    return output

def count_params(model):
    return int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))

def aug_img(img, rotate=None, translate=None, zoom=None, channel_shift=None, flip=None):
    augs = []
    if not rotate is None:
        img, theta = random_rotation(img, rotate)
        augs.append(theta)
    else:
        augs.append(0)
    if not translate is None:
        img = random_translation(img, translate, translate)
    if not zoom is None:
        img = random_zoom(img, zoom)
    if not channel_shift is None:
        img = random_channel_shift(img, channel_shift)
    if not flip is None:
        flip_h = np.random.random() > 0.5
        flip_v = np.random.random() > 0.5
        if flip[0] and flip_h: img = flip_axis(img, 1)
        if flip[1] and flip_v: img = flip_axis(img, 0)
        augs.append(1 if flip_h else 0)
        augs.append(1 if flip_v else 0)
    else:
        augs.append(0)
        augs.append(0)
    
    return (img, augs)