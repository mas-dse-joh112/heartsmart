#!/usr/bin/env python

import cv2 
import re, sys
import fnmatch, shutil, subprocess
from IPython.utils import io
import glob
import random
import json
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import numpy as np
from keras.models import *
from keras.layers import Input, concatenate, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.losses import binary_crossentropy
import keras.backend as K
from helpers_dicom import DicomWrapper as dicomwrapper

#Fix the random seeds for numpy (this is for Keras) and for tensorflow backend to reduce the run-to-run variance
from numpy.random import seed
seed(100)
from tensorflow import set_random_seed
set_random_seed(200)

#import matplotlib.pyplot as plt
#%matplotlib inline

import tensorflow as tf

print("\nSuccessfully imported packages!!!\n")


class myUnet(object):
    def __init__(self, patient, source_type, image_size = 256, model_type = "small"):
        self.img_rows = image_size
        self.img_cols = image_size
        self.patient = patient
        self.source_type = source_type
        self.model_type = model_type
        self.image_4d_file = None
        self.image_source_file = None
        self.image_one_file = None
        self.sourcedict = dict()


        if model_type == "small":
            self.build_unet_small()
        elif model_type == "large":
            self.build_unet()
        elif model_type == "large2":
            self.build_unet()
        else :
            print ("Specify valid model_type (small, large, large2)")
            return
    
    def dice_coeff(self, y_true, y_pred):
        smooth = 1.
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return score

    def load_images(self, imgfile):
        print('-'*30)
        print('load np arrays of images ...')
        print('-'*30)
        print ("Loading files : ", imgfile)
    
        im = np.load(imgfile)
        images = im.astype('float32')
    
        ##Normalize the pixel values, (between 0..1)
        x_min = images.min(axis=(1, 2), keepdims=True)
        x_max = images.max(axis=(1, 2), keepdims=True)
        images2 = (images - x_min)/(x_max-x_min)

        print("shape, max, min, mean of original image set:", images.shape, images.max(), images.min(), images.mean())
        print("shape, max, min, mean after normalization  :", images2.shape, images2.max(), images2.min(), images2.mean())
        return images2

    def load_data(self, train_data, test_data):
        print('-'*30)
        print("loading data")
        self.train_images, self.train_labels = load_images_and_labels(train_data)
        self.test_images, self.test_labels = load_images_and_labels(test_data)       
        print("loading data done")
        print('-'*30)

    def build_unet_small(self):
        '''
        Input shape
        4D tensor with shape: (samples, channels, rows, cols) if data_format='channels_first' 
        or 4D tensor with shape: (samples, rows, cols, channels) if data_format='channels_last' (default format).
        
        Output shape
        4D tensor with shape: (samples, filters, new_rows, new_cols) if data_format='channels_first' or 
        4D tensor with shape: (samples, new_rows, new_cols, filters) if data_format='channels_last'. 
        rows and cols values might have changed due to padding.
        
        He_normal initialization: It draws samples from a truncated normal distribution centered on 0 with stddev = sqrt(2 / fan_in) 
        where  fan_in is the number of input units in the weight tensor.
        '''

        print('-'*30)
        print ("Building smaller version of U-net model")
        print('-'*30)
        
        inputs = Input((self.img_rows, self.img_cols,1))

        conv1 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        print ("conv1 shape:",conv1.shape)
        conv1 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        print ("conv1 shape:",conv1.shape)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        print ("pool1 shape:",pool1.shape)

        conv2 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        print ("conv2 shape:",conv2.shape)
        conv2 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        print ("conv2 shape:",conv2.shape)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        print ("pool2 shape:",pool2.shape)

        conv3 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        print ("conv3 shape:",conv3.shape)
        conv3 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        print ("conv3 shape:",conv3.shape)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        print ("pool3 shape:",pool3.shape)

        conv4 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        drop4 = Dropout(0.2)(conv4)
        
        up5 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop4))
        merge5 = concatenate([conv3,up5], axis = 3)
        conv5 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge5)
        conv5 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        
        up6 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
        merge6 = concatenate([conv2,up6], axis = 3)
        conv6 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

        up7 = Conv2D(8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = concatenate([conv1,up7], axis = 3)
        conv7 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

        conv8 = Conv2D(1, 1, activation = 'sigmoid')(conv7)

        self.model = Model(input = inputs, output = conv8)

        self.model.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=[self.dice_coeff])
        #model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])


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
        drop5 = Dropout(0.2)(conv5)

        up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        merge6 = concatenate([drop4,up6], axis = 3)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

        up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = concatenate([conv3,up7], axis = 3)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

        up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = concatenate([conv2,up8], axis = 3)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

        up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = concatenate([conv1,up9], axis = 3)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        #conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

        self.model = Model(input = inputs, output = conv10)

        #self.model.compile(optimizer=RMSprop(lr=0.0001), loss=penalized_bce_loss(weight=0.08), metrics=['binary_accuracy'])
        #self.model.compile(optimizer=RMSprop(lr=0.0001), loss=dice_loss, metrics=[dice_coeff])

        #metrics=['accuracy'] calculates accuracy automatically from cost function. So using binary_crossentropy shows binary 
        #accuracy, not categorical accuracy.Using categorical_crossentropy automatically switches to categorical accuracy
        #One can get both categorical and binary accuracy by using metrics=['binary_accuracy', 'categorical_accuracy']
        
        self.model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = [self.dice_coeff])
        #self.model.compile(optimizer = Adam(lr = 1e-4), loss = my_bce_loss, metrics = ['binary_accuracy'])
    
    def load_pretrained_weights(self, model_file):
        self.model_file = model_file
        print('-'*30)
        print('Loading pre-trained weights...')
        self.model.load_weights(self.model_file)
        print('-'*30)   

    def predict(self, test_image_array, test_label_array ="none"):
        self.test_images = test_image_array
        self.test_labels = test_label_array
        print('-'*30)
        print('predict test data....')
        self.predictions = self.model.predict(self.test_images, batch_size=1, verbose=1)
        print('-'*30)
        print('-'*30)
        
        if self.test_labels != "none" :
            scores = self.model.evaluate (self.predictions, self.test_labels, batch_size=4)
            print ("Prediction Scores before rounding", scores)

            pred2 = np.round(self.predictions)
            scores = self.model.evaluate (pred2,  self.test_labels, batch_size=4)
            print ("Prediction Scores after rounding", scores)

    def train_and_predict(self, model_file, batch_size = 4, nb_epoch = 10, augmentation = False): 
        self.model_file = model_file #path to save the weights with best model
        model_checkpoint = ModelCheckpoint(self.model_file, monitor='loss',verbose=1, save_best_only=True)
        
        if augmentation == True :
            sample_size, x_val, y_val, ax = self.train_images.shape
            #save original train images
            self.original_train_images = self.train_images
            self.original_train_labels = self.train_labels
            # we create two instances with the same arguments
            data_gen_args = dict(
                                 rotation_range=90.,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 zoom_range=0.2)

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
                            
        print('-'*30)
        print('Fitting model...')
        print('-'*30)
        self.history = self.model.fit(self.train_images, self.train_labels, batch_size, nb_epoch, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])

        print('-'*30)
        print('predict test data....')
        self.predictions = self.model.predict(self.test_images, batch_size=1, verbose=1)
        scores = self.model.evaluate (self.predictions, self.test_labels, batch_size=4)
        print ("Prediction Scores", scores)
        pred_file = "predictions_sk2.npy"
        pred_file = UNET_TRAIN_DIR + pred_file
        np.save(pred_file, self.predictions)
        print('-'*30)
        
        
    def train_with_augmentation(self, model_file, batch_size = 4, nb_epoch = 10 ):

        #np.concatenate((acdc_train_img, sb_train_img), axis=0)
        sample_size, x_val, y_val, ax = self.train_images.shape

        self.model_file = model_file #path to save the weights with best model
        model_checkpoint = ModelCheckpoint(self.model_file, monitor='loss',verbose=1, save_best_only=True)
        
        if augmentation == True :
            #save original train images
            self.original_train_images = self.train_images
            self.original_train_labels = self.train_labels
            # we create two instances with the same arguments
            data_gen_args = dict(
                                 rotation_range=90.,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 zoom_range=0.2)

            image_datagen = ImageDataGenerator(**data_gen_args)
            mask_datagen = ImageDataGenerator(**data_gen_args)
            
            # Provide the same seed and keyword arguments to the fit and flow methods
            seed = 1
            image_generator = image_datagen.flow(self.train_images, y=None, seed = seed, batch_size=sample_size)
            mask_generator = mask_datagen.flow(self.train_labels,  y=None, seed = seed, batch_size=sample_size)
            train_generator = zip(image_generator, mask_generator)

            MAX_AUG=2
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
        
        print('-'*30)
        print('Fitting model...')
        
        self.history = self.model.fit(self.train_images, self.train_labels, batch_size, nb_epoch, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])
        
        print('-'*30)
        print('Run Predictions on test data')
        self.predictions = self.model.predict(self.test_images, batch_size=1, verbose=1)
        scores = self.model.evaluate (self.predictions, self.test_labels, batch_size=4)
        print ("Prediction Scores", scores)
        pred_file = "predictions_aug_sk1.npy"
        pred_file = UNET_TRAIN_DIR + pred_file
        np.save(pred_file, self.predictions)
        print('-'*30)        
    


    def train_with_augmentation2(self, model_file, batch_size = 4, nb_epoch = 10 ):

        sample_size, x_val, y_val, ax = self.train_images.shape

        model_file = UNET_MODEL_DIR+'unet_aug2.hdf5' #path to save the weights with best model
        model_checkpoint = ModelCheckpoint(model_file, monitor='loss',verbose=1, save_best_only=True)
        
        # we create two instances with the same arguments
        data_gen_args = dict(featurewise_center=True,
                             featurewise_std_normalization=True,
                             rotation_range=90.,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2)
        
        image_datagen = ImageDataGenerator(**data_gen_args)
        mask_datagen = ImageDataGenerator(**data_gen_args)

        # Provide the same seed and keyword arguments to the fit and flow methods
        seed = 1
        image_datagen.fit(self.train_images, augment=True, seed=seed)
        mask_datagen.fit(self.train_labels, augment=True, seed=seed)

        # Provide the same seed and keyword arguments to the fit and flow methods
        seed = 1
        image_generator = image_datagen.flow(self.train_images, y=None, seed = seed, batch_size=sample_size)
        mask_generator = mask_datagen.flow(self.train_labels,  y=None, seed = seed, batch_size=sample_size)
        train_generator = zip(image_generator, mask_generator)
        
        print('-'*30)
        print('Fitting model...')
        
        self.model.fit(self.train_images, self.train_labels, batch_size, nb_epoch, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])
        
        MAX_AUG=2
        augmentation_round = 0
        for img_tr, mask_tr in train_generator:
                print ("Augmentation round: ", augmentation_round+1, img_tr.shape)
                s, x1, y1, p = img_tr.shape
                self.model.fit(img_tr, mask_tr, batch_size, nb_epoch, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])
                augmentation_round += 1
                if (augmentation_round == MAX_AUG):
                      break
            
        
        print('-'*30)
        print('Run Predictions on test data')
        self.predictions = self.model.predict(self.test_images, batch_size=1, verbose=1)
        
        pred_file = "predictions_aug_sk1.npy"
        pred_file = UNET_TRAIN_DIR + pred_file
        np.save(pred_file, self.predictions)
        print('-'*30)
        
    def save_img(self):
        pred_file = "predictions.npy"
        pred_file = UNET_TRAIN_DIR + pred_file
        print("array to image")
        imgs = np.load(pred_file)
        for i in range(imgs.shape[0]):
            img = imgs[i]
            img = array_to_img(img)
            img.save("./%d.jpg"%(i))

    def display_ytrue_ypred (self, num_images = 4, random_images = False, evaluate= False):
        # if there are no test labels then just display the predictions
        if self.test_labels == "none":
            self.display_ypred (num_images = num_images, random_images = False)
            return
        
        ts , tl= self.test_images, self.test_labels
        pred = self.predictions
        samples, x, y, z = pred.shape
        print ("samples, max, min ", samples, pred.max(), pred.min())
        pred2 = np.round(pred)
        if (evaluate == True) :
            scores = self.model.evaluate (pred, tl, batch_size=1)
            print ("Prediction Scores", scores)

            scores = self.model.evaluate (pred2, tl, batch_size=1)
            print ("Prediction Scores after rounding", scores)
        ##Print few images wih actual labels and predictions
        display_list = []

        if random_images == True:
            display_list = random.sample(range(0, samples), num_images)
        else :
            display_list = [i for i in range (num_images)]
            
        """
        for i in display_list:
            f, axs = plt.subplots(1,4,figsize=(15,15))
            plt.subplot(141),plt.imshow(ts[i].reshape(x, y))
            plt.title('test image '+str(i)), plt.xticks([]), plt.yticks([])
            plt.subplot(142),plt.imshow(tl[i].reshape(x, y))
            plt.title('test label'), plt.xticks([]), plt.yticks([])
            plt.subplot(143),plt.imshow(pred2[i].reshape(x, y))
            plt.title('pred label'), plt.xticks([]), plt.yticks([])
            plt.subplot(144),plt.imshow(tl[i].reshape(x, y)), plt.imshow(pred2[i].reshape(x, y), 'binary', interpolation='none', alpha=0.5)
            plt.title('overlay'), plt.xticks([]), plt.yticks([])
            plt.show()
        """
            
#             f, axs = plt.subplots(1,3,figsize=(15,15))
#             plt.subplot(131),plt.imshow(ts[i].reshape(x, y))
#             plt.title('test Image'+ str(i)), plt.xticks([]), plt.yticks([])
#             plt.subplot(132),plt.imshow(tl[i].reshape(x, y))
#             plt.title('test label'), plt.xticks([]), plt.yticks([])
#             plt.subplot(133),plt.imshow(pred[i].reshape(x, y))
#             plt.title('Predicted mask'), plt.xticks([]), plt.yticks([])
#             plt.show()
    
    
    def display_ypred (self, num_images = 4, random_images = False):
        ts = self.test_images
        pred = self.predictions
        samples, x, y, z = pred.shape
        print ("samples, max, min ", samples, pred.max(), pred.min())
        pred2 = np.round(pred)

        display_list = []

        if random_images == True:
            display_list = random.sample(range(0, samples), num_images)
        else :
            display_list = [i for i in range (num_images)]

        """
        for i in display_list:
            f, axs = plt.subplots(1,3,figsize=(15,15))
            plt.subplot(131),plt.imshow(ts[i].reshape(x, y))
            plt.title('test image '+str(i)), plt.xticks([]), plt.yticks([])
            plt.subplot(132),plt.imshow(pred2[i].reshape(x, y))
            plt.title('prediction'), plt.xticks([]), plt.yticks([])
            plt.subplot(133),plt.imshow(ts[i].reshape(x, y)), plt.imshow(pred2[i].reshape(x, y), 'binary', interpolation='none', alpha=0.3)
            plt.title('overlay'), plt.xticks([]), plt.yticks([])
            plt.show()
        """
    
    def dump_and_sort(self):
        count = 0
        slicedict = {}
        origpath = '/masvol/data/dsb/'+self.source_type+'/'+str(self.patient)+'/study/'
        new_path = '/opt/output/dsb/volume/1/3/'+self.source_type+'_'+str(self.patient)+'.json'

        with open(self.image_one_file, 'r') as inputs:
            jin = json.load(inputs)

            for i in sorted(jin):
                count += 1
                rootnode = i.split("/")
                tmp=rootnode[-1].split('_')
                sax = tmp[0] +'_'+ tmp[1]
                frame = tmp[-1]
                #print (sax, frame)

                if sax in slicedict:
                    slicedict[sax].update({frame: jin[i]})
                else:
                    slicedict.update({sax: {frame: jin[i]}})

                #print (sax, frame)
            min_max = self.get_min_max_slice(slicedict, origpath)
            print (min_max)
            
            with open(new_path, 'w') as output:
                output.write("{0}\n".format(json.dumps(min_max)))

    def get_min_max_slice(self, slicedict, origpath):
        identified = {}

        for i in slicedict: #i is sax
            zmin = 9999999
            zmax = 0
            zminframe = ''
            zmaxframe = ''

            for j in slicedict[i]: #j is frame
                zcount = slicedict[i][j]['ones']
                
                if zcount < zmin:
                    zmin = zcount
                    zminframe = j

                if zcount > zmax:
                    zmax = zcount
                    zmaxframe = j

            maxpath = i+'/'+zmaxframe.strip('.npy')
            minpath = i+'/'+zminframe.strip('.npy')
            maxsl = None
            minsl = None

            try:
                maxdw = dicomwrapper(origpath, maxpath )
                maxsl = maxdw.slice_location()
            except:
                print (origpath, maxpath)
                maxsl = None
 
            try:
                mindw = dicomwrapper(origpath, minpath)
                minsl = mindw.slice_location()
            except:
                print (origpath, minpath)
                minsl = None

            identified[i] = {'zmin':zmin,
                             'zminframe': zminframe,
                             'minSL': minsl,
                             'zmax': zmax,
                             'zmaxframe': zmaxframe,
                             'maxSL': maxsl}
        #print (identified)
        return identified

    def get_ones(self):
        print ('l', len(self.predictions))
        sourcefiles = []

        with open(self.image_source_file, 'r') as sourceinput:
            for i in sourceinput:
                sourcefiles = i.strip().split(',')

        print ('SF',len(sourcefiles))

        for i in sourcefiles:
            self.sourcedict[i] = {'ones':0} # init, may not have prediction

        pred2  = np.round(self.predictions)
        print (pred2.shape)

        for i in range(len(pred2)):
            zcount = np.count_nonzero(pred2[i])
            self.sourcedict.update({sourcefiles[i]: {'ones':zcount}}) # save ones count for now

        with open(self.image_one_file, 'w') as output:
            output.write("{0}\n".format(json.dumps(self.sourcedict)))

    def plot_accuracy_and_loss(self):
        # list all data in history
        print(self.history.history.keys())
        history = self.history
        print ("First and final values of learning curve")
        for key in history.history:
            print (key, history.history[key][0], history.history[key][-1])

        """
        # summarize history for accuracy
        if 'dice_coeff' in self.history.history.keys():
            plt.plot(history.history['dice_coeff'])
            plt.plot(history.history['val_dice_coeff'])
            plt.title('model accuracy(dice_coeff)')
        elif 'val_acc' in self.history.history.keys():
            plt.plot(history.history['acc'])
            plt.plot(history.history['val_acc'])
            plt.title('model accuracy')
        elif 'categorical_accuracy' in self.history.history.keys():
            plt.plot(history.history['categorical_accuracy'])
            plt.plot(history.history['val_categorical_accuracy'])
            plt.title('categorical_accuracy')
        elif 'binary_accuracy' in self.history.history.keys():
            plt.plot(history.history['binary_accuracy'])
            plt.plot(history.history['val_binary_accuracy'])
            plt.title('binary_accuracy')
        else : 
            print ("new loss function, not in the list")
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        """

    def do_predict(self):
        for image_file in [mymodel.image_4d_file]:
            ts = mymodel.load_images(image_file)
            print (ts.shape)
            tl = "none"
            print('Run predictions...')
            mymodel.predict(test_image_array = ts, test_label_array = tl)
            print('-'*30)
            mymodel.display_ytrue_ypred(num_images = 4, random_images = False, evaluate = False)
            #mymodel.predictions (2, 256, 256, 1)
            #save the predictions in the form of numpy array
            #pred_file = "/opt/output/dsb/norm/1/3/unet_model_test/models/dsb_884_256_predictions.npy"
            #np.save(pred_file, mymodel.predictions)

if __name__ == "__main__":
    arg = sys.argv[1:]
    #patient = 884
    #image_size = 256
    #source_type = "test"
    if len(arg) != 3:
        print ('provide patient number, 256 (or 176), and test (train or validate)')
        sys.exit()

    patient, image_size, source_type = arg
    patient = int(patient)
    image_size = int(image_size)
    dsb_source = "unet_model_{0}".format(source_type)
    test_source_path = "/opt/output/dsb/norm/1/3"

    #test_image_list = [#"/opt/output/dsb/norm/1/3/unet_model_test/data/dsb_884_256_train.npy", \
                        #"/opt/output/dsb/norm/1/3/unet_model_test/data/dsb_885_256_train.npy", \
                        #"/opt/output/dsb/norm/1/3/unet_model_test/data/dsb_886_256_train.npy", \
                        #"/opt/output/dsb/norm/1/3/unet_model_test/data/dsb_887_256_train.npy", \
    #                  ]

    model_file = "/masvol/heartsmart/unet_model/models_baseline/combined_{0}.hdf5".format(image_size)

    print('-'*30)
    print ("Creating U-net model...")
    mymodel = myUnet(patient, source_type, image_size = image_size, model_type = "large")
    print('-'*30)
    print ("Loading the pre-trained weights...")
    mymodel.load_pretrained_weights(model_file)
    print('-'*30)
    
    mymodel.image_4d_file = "{0}/{1}/data/dsb_{2}_{3}_train.npy".format(test_source_path, dsb_source, patient, image_size)
    mymodel.image_source_file = "{0}/{1}/data/dsb_{2}_image_path.txt".format(test_source_path, dsb_source, patient)
    mymodel.image_one_file = "{0}/{1}/models/dsb_{2}_{3}_one_count.json".format(test_source_path, dsb_source, patient, image_size)

    mymodel.do_predict()
    mymodel.get_ones()
    #mymodel.sort_by_slice()
    mymodel.dump_and_sort()
