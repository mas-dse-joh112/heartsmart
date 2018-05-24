#!/usr/bin/env python

import cv2 
import re, sys
import fnmatch, shutil, subprocess
from IPython.utils import io
import glob
import random
import json
import os 

import numpy as np
from keras.models import *
from keras.layers import Input, concatenate, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.losses import binary_crossentropy
import keras.backend as K
from keras import backend as keras
#from helper_functions  import *
from helper_utilities  import *

from loss_functions  import *


#from helpers_dicom import DicomWrapper as dicomwrapper

#Fix the random seeds for numpy (this is for Keras) and for tensorflow backend to reduce the run-to-run variance
from numpy.random import seed
seed(12345)
from tensorflow import set_random_seed
set_random_seed(12345)

#import matplotlib.pyplot as plt
#%matplotlib inline

import tensorflow as tf

#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
#os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
#os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6"
#os.environ["CUDA_VISIBLE_DEVICES"] = "5,6, 7"

from modelmgpu import ModelMGPU
#GPUs = 2

# import time

# start = time.time()
# print("START:", start)

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.90
# session = tf.Session(config=config)

print("\nSuccessfully imported packages!!!\n")


class myUnet(object):
    def __init__(self, model_name = "unet", nGPU=0, image_size = 256, dropout = True, optimizer = 'Adam', lr=.00001, loss_fn="dice_loss"):
        self.img_rows = image_size
        self.img_cols = image_size
        self.parallel_model = None
        self.patient = None
        self.data_source = None
        self.file_source = None
        self.image_size = None
        self.source_type = None
        self.test_source_path = None
        self.image_4d_file = None
        self.image_source_file = None
        self.image_one_file = None
        self.sourcedict = dict()
        self.model_name = model_name
        self.train_size = 0
        ### Model training parameters
        self.nGPUs = nGPU
        self.lossfn_str = loss_fn
        self.dropout = dropout
        self.learningrate_str = str(lr)
        if loss_fn == 'dice_loss':
            self.loss_fn = dice_loss
        elif loss_fn == 'log_dice_loss':
            self.loss_fn = log_dice_loss
        elif loss_fn == 'bce_dice_loss':
            self.loss_fn = bce_dice_loss
        elif loss_fn == 'binary_crossentropy':
            self.loss_fn = 'binary_crossentropy'
        else :
            self.loss_fn = 'binary_crossentropy'
            
        self.optimizer_str = optimizer    
        if optimizer == 'Adam':
            self.optimizer = Adam(lr = lr)
        elif optimizer == 'RMSprop':
            self.optimizer = RMSprop(lr = lr)
        else :
            print ("unknown optimizer: default to Adam", optimizer)
            self.optimizer = Adam(lr = lr)      
        self.learningrate = lr
        self.metrics = [dice_coeff, 'binary_accuracy']
        self.epoch = 50 # gets updated later
        self.batch_size = 16 # gets updated later
        
        self.build_unet3()

    
    def load_data(self, train_data, test_data):
        print('-'*30)
        print("loading data")
        self.train_images, self.train_labels = load_images_and_labels(train_data, normalize= True)
        self.test_images, self.test_labels = load_images_and_labels(test_data, normalize= True)       
        print("loading data done")
        print('-'*30)
        

    def get_crop_shape(self, src, dest):
        # width, the 3rd dimension
        cw = (src.get_shape()[2] - dest.get_shape()[2]).value
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw/2), int(cw/2) + 1
        else:
            cw1, cw2 = int(cw/2), int(cw/2)
        # height, the 2nd dimension
        ch = (src.get_shape()[1] - dest.get_shape()[1]).value
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch/2), int(ch/2) + 1
        else:
            ch1, ch2 = int(ch/2), int(ch/2)

        return (ch1, ch2), (cw1, cw2)
    
    def build_unet3(self):
        
        '''
        Input shape
        4D tensor with shape: (samples, channels, rows, cols) if data_format='channels_first' 
        or 4D tensor with shape: (samples, rows, cols, channels) if data_format='channels_last' (default format).
        
        Output shape
        4D tensor with shape: (samples, filters, new_rows, new_cols) if data_format='channels_first' or 
        4D tensor with shape: (samples, new_rows, new_cols, filters) if data_format='channels_last'. 
        rows and cols values might have changed due to padding.
        '''
        print('-'*30)
        print ("Building U-net model3")
        print('-'*30)
        
        inputs = Input((self.img_rows, self.img_cols,1))
        
        conv0 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)     
        conv0 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv0)
        print ("conv0 shape:",conv0.shape)
        pool0 = MaxPooling2D(pool_size=(2, 2))(conv0)
        print ("pool0 shape:",pool0.shape)

        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool0)     
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        print ("conv1 shape:",conv1.shape)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        print ("pool1 shape:",pool1.shape)

        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        print ("conv2 shape:",conv2.shape)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        print ("pool2 shape:",pool2.shape)

        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        print ("conv3 shape:",conv3.shape)
        print ("Adding .5 dropout layer")
        if self.dropout == True:
            print ("Adding dropout layer")
            conv3 = Dropout(0.5)(conv3)

        up4 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv3))
        merge4 = concatenate([conv2,up4], axis = 3)
        conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge4)
        conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        print ("conv4 shape:",conv4.shape)
        if self.dropout == True:
            print ("Adding dropout layer")
            conv4 = Dropout(0.5)(conv4)

        up5 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv4))
        merge5 = concatenate([conv1,up5], axis = 3)
        conv5 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge5)
        conv5 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        print ("conv5 shape:",conv5.shape)
        if self.dropout == True:
            print ("Adding dropout layer")
            conv5 = Dropout(0.5)(conv5)

        up6 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
        merge6 = concatenate([conv0,up6], axis = 3)
        conv6 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
        print ("conv6 shape:",conv6.shape)

        conv7 = Conv2D(1, 1, activation = 'sigmoid')(conv6)

        self.model = Model(input = inputs, output = conv7)
        self.parallel_model = ModelMGPU(self.model, self.nGPUs)

        #self.model.compile(optimizer=RMSprop(lr=0.0001), loss=penalized_bce_loss(weight=0.08), metrics=['binary_accuracy'])
        #self.model.compile(optimizer=RMSprop(lr=0.0001), loss=dice_loss, metrics=[dice_coeff])

        #metrics=['accuracy'] calculates accuracy automatically from cost function. So using binary_crossentropy shows binary 
        #accuracy, not categorical accuracy.Using categorical_crossentropy automatically switches to categorical accuracy
        #One can get both categorical and binary accuracy by using metrics=['binary_accuracy', 'categorical_accuracy']
        
        #self.parallel_model.compile(optimizer = Adam(lr = 1e-4), loss = dice_loss, metrics = [dice_coeff])
        #self.model.compile(optimizer = Adam(lr = 1e-4), loss = dice_loss, metrics = [dice_coeff])
        
        print ("compiling the model")
        #self.model.compile(optimizer = self.optimizer, loss = self.loss_fn, metrics = self.metrics)
        try:
            self.parallel_model.compile(optimizer = self.optimizer, loss = self.loss_fn, metrics = self.metrics)

            #self.parallel_model.compile(optimizer = Adam(lr = 1e-4), loss = bce_dice_loss, metrics = [dice_coeff])
        except ValueError:
            print ("Error invalid parameters to model compilation")
        
    def build_unet2(self):
        
        '''
        Input shape
        4D tensor with shape: (samples, channels, rows, cols) if data_format='channels_first' 
        or 4D tensor with shape: (samples, rows, cols, channels) if data_format='channels_last' (default format).
        
        Output shape
        4D tensor with shape: (samples, filters, new_rows, new_cols) if data_format='channels_first' or 
        4D tensor with shape: (samples, new_rows, new_cols, filters) if data_format='channels_last'. 
        rows and cols values might have changed due to padding.
        '''
        print('-'*30)
        print ("Building U-net model")
        print('-'*30)
        
        inputs = Input((self.img_rows, self.img_cols,1))
        
        conv0 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        print ("conv0 shape:",conv0.shape)
     
        conv0 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv0)
        print ("conv0 shape:",conv0.shape)
        pool0 = MaxPooling2D(pool_size=(2, 2))(conv0)
        print ("pool0 shape:",pool0.shape)

        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool0)
        print ("conv1 shape:",conv1.shape)
     
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        print ("conv1 shape:",conv1.shape)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        print ("pool1 shape:",pool1.shape)

        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        print ("conv2 shape:",conv2.shape)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        print ("conv2 shape:",conv2.shape)
        pool2 = MaxPooling2D(pool_size=(2, 2), border_mode="same")(conv2)
        print ("pool2 shape:",pool2.shape)

        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        print ("conv3 shape:",conv3.shape)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        print ("conv3 shape:",conv3.shape)
        pool3 = MaxPooling2D(pool_size=(2, 2), border_mode="same")(conv3)
        print ("pool3 shape:",pool3.shape)

        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        print ("conv4 shape:",conv4.shape)
        pool4 = MaxPooling2D(pool_size=(2, 2), border_mode="same")(conv4)

        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        print ("conv5 shape:",conv5.shape)
        print ("Adding .5 dropout layer")
        if self.dropout == True:
            print ("Adding .5 dropout layer")
            conv5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
        print ("up6 shape:",up6.shape)
#         ch, cw = self.get_crop_shape(conv4, up6)
#         crop_conv4 = Cropping2D(cropping=(ch,cw))(conv4)
#         print ("crop_conv4 shape:",crop_conv4.shape)
#         merge6 = concatenate([crop_conv4,up6], axis = 3)
        ch, cw = self.get_crop_shape(up6, conv4)
        crop_up6 = Cropping2D(cropping=(ch,cw))(up6)
        print ("crop_conv4 shape:",crop_up6.shape)
        merge6 = concatenate([conv4,crop_up6], axis = 3)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
        if self.dropout == True:
            print ("Adding .5 dropout layer")
            conv6 = Dropout(0.5)(conv6)

        up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        print ("up7 shape:",up7.shape)
        ch, cw = self.get_crop_shape(conv3, up7)
        crop_conv3 = Cropping2D(cropping=(ch,cw))(conv3)
        print ("crop_conv3 shape:",crop_conv3.shape)
        merge7 = concatenate([crop_conv3,up7], axis = 3)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
        if self.dropout == True:
            print ("Adding .5 dropout layer")
            conv7 = Dropout(0.5)(conv7)

        up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        print ("up8 shape:",up8.shape)
        ch, cw = self.get_crop_shape(conv2, up8)
        crop_conv2 = Cropping2D(cropping=(ch,cw))(conv2)
        print ("crop_conv2 shape:",crop_conv2.shape)
        merge8 = concatenate([crop_conv2,up8], axis = 3)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
        if self.dropout == True:
            print ("Adding .5 dropout layer")
            conv8 = Dropout(0.5)(conv8)
        
        up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        print ("up9 shape:",up9.shape)
        ch, cw = self.get_crop_shape(conv1, up9)
        crop_conv1 = Cropping2D(cropping=(ch,cw))(conv1)
        print ("crop_conv1 shape:",crop_conv1.shape)
        merge9 = concatenate([crop_conv1,up9], axis = 3)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        if self.dropout == True:
            print ("Adding .5 dropout layer")
            conv9 = Dropout(0.5)(conv9)
            
        up10 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv9))
        print ("up10 shape:",up10.shape)
        ch, cw = self.get_crop_shape(conv0, up10)
        crop_conv0 = Cropping2D(cropping=(ch,cw))(conv0)
        print ("crop_conv0 shape:",crop_conv0.shape)
        merge10 = concatenate([crop_conv0,up10], axis = 3)
        conv10 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge10)
        conv10 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv10)       
        
        #conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv11 = Conv2D(1, 1, activation = 'sigmoid')(conv10)

        print ("compiling the model")
        self.model = Model(input = inputs, output = conv11)
        self.parallel_model = ModelMGPU(self.model, self.nGPUs)

        #self.model.compile(optimizer=RMSprop(lr=0.0001), loss=penalized_bce_loss(weight=0.08), metrics=['binary_accuracy'])
        #self.model.compile(optimizer=RMSprop(lr=0.0001), loss=dice_loss, metrics=[dice_coeff])

        #metrics=['accuracy'] calculates accuracy automatically from cost function. So using binary_crossentropy shows binary 
        #accuracy, not categorical accuracy.Using categorical_crossentropy automatically switches to categorical accuracy
        #One can get both categorical and binary accuracy by using metrics=['binary_accuracy', 'categorical_accuracy']
        
        #self.parallel_model.compile(optimizer = Adam(lr = 1e-4), loss = dice_loss, metrics = [dice_coeff])
        #self.model.compile(optimizer = Adam(lr = 1e-4), loss = dice_loss, metrics = [dice_coeff])
        
        print ("compiling the model")
        #self.model.compile(optimizer = self.optimizer, loss = self.loss_fn, metrics = self.metrics)
        try:
            self.parallel_model.compile(optimizer = self.optimizer, loss = self.loss_fn, metrics = self.metrics)

            #self.parallel_model.compile(optimizer = Adam(lr = 1e-4), loss = bce_dice_loss, metrics = [dice_coeff])
        except ValueError:
            print ("Error invalid parameters to model compilation")

    
    
    def build_unet(self):
        
        '''
        Input shape
        4D tensor with shape: (samples, channels, rows, cols) if data_format='channels_first' 
        or 4D tensor with shape: (samples, rows, cols, channels) if data_format='channels_last' (default format).
        
        Output shape
        4D tensor with shape: (samples, filters, new_rows, new_cols) if data_format='channels_first' or 
        4D tensor with shape: (samples, new_rows, new_cols, filters) if data_format='channels_last'. 
        rows and cols values might have changed due to padding.
        '''
        print('-'*30)
        print ("Building U-net model")
        print('-'*30)
        
        inputs = Input((self.img_rows, self.img_cols,1))

        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        print ("conv1 shape:",conv1.shape)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        print ("conv1 shape:",conv1.shape)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        print ("pool1 shape:",pool1.shape)

        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        print ("conv2 shape:",conv2.shape)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        print ("conv2 shape:",conv2.shape)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        print ("pool2 shape:",pool2.shape)

        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        print ("conv3 shape:",conv3.shape)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        print ("conv3 shape:",conv3.shape)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        print ("pool3 shape:",pool3.shape)

        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        #drop4 = Dropout(0.5)(conv4)
        drop4 = conv4
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        drop5 = conv5
        if self.dropout == True:
            print ("Adding dropout layer")
            drop5 = Dropout(0.5)(drop5)

        up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        merge6 = concatenate([drop4,up6], axis = 3)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
        if self.dropout == True:
            print ("Adding dropout layer")
            conv6 = Dropout(0.5)(conv6)

        up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = concatenate([conv3,up7], axis = 3)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
        if self.dropout == True:
            print ("Adding dropout layer")
            conv7 = Dropout(0.5)(conv7)

        up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = concatenate([conv2,up8], axis = 3)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
        if self.dropout == True:
            print ("Adding dropout layer")
            conv8 = Dropout(0.5)(conv8)
        
        up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = concatenate([conv1,up9], axis = 3)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        #conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

        self.model = Model(input = inputs, output = conv10)
        self.parallel_model = ModelMGPU(self.model, self.nGPUs)

        #self.model.compile(optimizer=RMSprop(lr=0.0001), loss=penalized_bce_loss(weight=0.08), metrics=['binary_accuracy'])
        #self.model.compile(optimizer=RMSprop(lr=0.0001), loss=dice_loss, metrics=[dice_coeff])

        #metrics=['accuracy'] calculates accuracy automatically from cost function. So using binary_crossentropy shows binary 
        #accuracy, not categorical accuracy.Using categorical_crossentropy automatically switches to categorical accuracy
        #One can get both categorical and binary accuracy by using metrics=['binary_accuracy', 'categorical_accuracy']
        
        #self.parallel_model.compile(optimizer = Adam(lr = 1e-4), loss = dice_loss, metrics = [dice_coeff])
        #self.model.compile(optimizer = Adam(lr = 1e-4), loss = dice_loss, metrics = [dice_coeff])
        
        print ("compiling the model")
        #self.model.compile(optimizer = self.optimizer, loss = self.loss_fn, metrics = self.metrics)
        try:
            self.parallel_model.compile(optimizer = self.optimizer, loss = self.loss_fn, metrics = self.metrics)

            #self.parallel_model.compile(optimizer = Adam(lr = 1e-4), loss = bce_dice_loss, metrics = [dice_coeff])
        except ValueError:
            print ("Error invalid parameters to model compilation")
            


        #self.model.compile(optimizer = Adam(lr = 1e-4), loss = bce_dice_loss, metrics = [dice_coeff])

        #self.model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = [dice_coeff])
        #self.model.compile(optimizer = Adam(lr = 1e-4), loss = my_bce_loss, metrics = ['binary_accuracy'])
    
    def load_pretrained_weights(self, model_file):
        self.model_file = model_file
        print('-'*30)
        print('Loading pre-trained weights...')
        self.parallel_model.load_weights(self.model_file)
        #self.model.load_weights(self.model_file)
        print('-'*30)   

    def predict(self, test_image_array, test_label_array ="none"):
        self.test_images = test_image_array
        self.test_labels = test_label_array
        print('-'*30)
        print('predict test data....')
        self.predictions = self.parallel_model.predict(self.test_images, batch_size=1, verbose=1)
        #self.predictions = self.model.predict(self.test_images, batch_size=1, verbose=1)
        print('-'*30)
        print('-'*30)
        

        if self.test_labels != "none" :
            scores = self.parallel_model.evaluate (self.predictions, self.test_labels, batch_size=4)
            #scores = self.model.evaluate (self.predictions, self.test_labels, batch_size=4)
            print ("Prediction Scores before rounding", scores)

            pred2 = np.round(self.predictions)
            scores = self.parallel_model.evaluate (pred2,  self.test_labels, batch_size=4)
            #scores = self.model.evaluate (pred2,  self.test_labels, batch_size=4)
            print ("Prediction Scores after rounding", scores)

    def train_and_predict(self, model_path, batch_size = 4, nb_epoch = 10, augmentation = False): 
 
        model_file = model_path + self.model_name + '.hdf5'
        self.model_file = model_file #path to save the weights with best model
        self.batch_size = batch_size
        self.epoch = nb_epoch
        model_checkpoint = ModelCheckpoint(self.model_file, monitor='loss',verbose=0, save_best_only=True)
        
        if augmentation == True :
            print ("perform augmentation")
            sample_size, x_val, y_val, ax = self.train_images.shape
            #save original train images
            self.original_train_images = self.train_images
            self.original_train_labels = self.train_labels
            # we create two instances with the same arguments
#             data_gen_args = dict(
#                                  rotation_range=90.,
#                                  width_shift_range=0.05,
#                                  height_shift_range=0.05,
#                                  zoom_range=0.1)
            data_gen_args = dict(
                     rotation_range=90.,
                     width_shift_range=0.025,
                     height_shift_range=0.025,
                     zoom_range=0.05)

            image_datagen = ImageDataGenerator(**data_gen_args)
            mask_datagen = ImageDataGenerator(**data_gen_args)
            
            # Provide the same seed and keyword arguments to the fit and flow methods
            seed = 1
            image_generator = image_datagen.flow(self.train_images, y=None, seed = seed, batch_size=sample_size)
            mask_generator = mask_datagen.flow(self.train_labels,  y=None, seed = seed, batch_size=sample_size)
            train_generator = zip(image_generator, mask_generator)

            MAX_AUG=3
            print('-'*30)
            print('Augmenting training data...')
            augmentation_round = 0
            for img_tr, mask_tr in train_generator:
                    self.train_images = np.concatenate((self.train_images, img_tr), axis=0)
                    self.train_labels = np.concatenate((self.train_labels, mask_tr), axis=0)
                    print ("Augmentation round: ", augmentation_round+1, img_tr.shape, self.train_images.shape, self.train_labels.shape)
                    augmentation_round += 1
                    if (augmentation_round == MAX_AUG):
                          break
                            
        samples, x, y, z = self.train_images.shape
        print ("samples, x, y", samples, x, y)
        self.train_size = samples
        print('-'*30)
        print('Fitting model...')
        print('-'*30)
        self.history = self.parallel_model.fit(self.train_images, self.train_labels, batch_size, nb_epoch, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])
        #self.history = self.model.fit(self.train_images, self.train_labels, batch_size, nb_epoch, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])

        print('-'*30)
        print('predict test data....')
        #first load the pre-trained weights that were saved from best run
        self.load_pretrained_weights(self.model_file)
        
        self.predictions = self.parallel_model.predict(self.test_images, batch_size=1, verbose=1)
        #self.predictions = self.model.predict(self.test_images, batch_size=1, verbose=1)
        self.scores = self.parallel_model.evaluate (self.predictions, self.test_labels, batch_size=4)
        #self.scores = self.model.evaluate (self.predictions, self.test_labels, batch_size=4)
        print ("Prediction Scores", self.parallel_model.metrics_names, self.scores)
        #print("%s: %.2f%%" % (self.model.metrics_names[1], self.scores[1]*100))
        print('-'*30)
        

            
    def save_model_info(self, mypath = "./"):
        learn_file =  self.model_name + "_learning_history.json"
        learn_file = mypath + learn_file
        hist = self.history.history
        #Append model name and training parameters to the dictionary
        hist['tr_model_name'] = self.model_name 
        hist['tr_size'] = self.train_size
        ### Model training parameters
        hist['tr_nGPUs'] = self.nGPUs
        hist['tr_dropout'] = str(self.dropout) 
        hist['tr_loss_fn'] = self.lossfn_str 
        hist['tr_optimizer'] = self.optimizer_str
        hist['tr_lrrate'] = self.learningrate_str 
        hist['tr_epoch'] = self.epoch 
        hist['tr_batchsize'] = self.batch_size 
        print('-'*30)
        print ("Saving Evaluation Scores on test set")
        for i in range (len(self.scores)):
            hist['eval_'+ self.parallel_model.metrics_names[i]] = self.scores[i]
        print('-'*30)            
        print ("Saving learning history", learn_file)
        with open(learn_file, 'w') as file:
            json.dump(self.history.history, file, indent=2)
        
        pred_file =  self.model_name + "_predictions.npy"
        pred_file = mypath + pred_file
        print ("Saving predictions", pred_file)
        np.save(pred_file, self.predictions)
        
        
#         pred_file =  self.model_name + "_predictions_rounded.npy"
#         pred_file = mypath + pred_file
#         np.save(pred_file, np.round(self.predictions))
        print('-'*30)
        
        print ("Saving Performance Statistics")
        perf = get_performance_statistics (self.test_labels, self.predictions)
        #Append model name and training parameters to the dictionary
        perf['tr_model_name'] = self.model_name 
        perf['tr_size'] = self.train_size
        ### Model training parameters
        perf['tr_nGPUs'] = self.nGPUs
        perf['tr_dropout'] = str(self.dropout)
        perf['tr_loss_fn'] = self.lossfn_str 
        perf['tr_optimizer'] = self.optimizer_str
        perf['tr_lrrate'] = self.learningrate_str 
        perf['tr_epoch'] = self.epoch 
        perf['tr_batchsize'] = self.batch_size 
        
        print ("Saving Evaluation Scores on test set")
        for i in range (len(self.scores)):
            perf['eval_'+ self.parallel_model.metrics_names[i]] = self.scores[i]
        
        self.perf = perf
        print ("Perf Statistics: ", self.perf)
        
        perf_file =  self.model_name + "_performance.json"
        perf_file = mypath + perf_file
        
        print ("Saving Performance values", perf_file)
        with open(perf_file, 'w') as file:
            json.dump(self.perf, file, indent=2)
        print('-'*30)


            



