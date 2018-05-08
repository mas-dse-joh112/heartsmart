#!/usr/bin/env python

import cv2 
import re, sys
import fnmatch, shutil, subprocess
from IPython.utils import io
import glob
import random
import json
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
from keras.models import *
from keras.layers import Input, concatenate, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.losses import binary_crossentropy
import keras.backend as K
from keras import backend as keras
from helper_functions  import *
from loss_functions  import *

from keras.utils import multi_gpu_model

#from helpers_dicom import DicomWrapper as dicomwrapper

#Fix the random seeds for numpy (this is for Keras) and for tensorflow backend to reduce the run-to-run variance
from numpy.random import seed
seed(1234)
from tensorflow import set_random_seed
set_random_seed(1234)

#import matplotlib.pyplot as plt
#%matplotlib inline

import tensorflow as tf


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.90
session = tf.Session(config=config)

print("\nSuccessfully imported packages!!!\n")


class ModelMGPU(Model):
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)

class myUnet(object):
    def __init__(self, model_name = "unet", image_size = 256, model_type = "small"):
        self.img_rows = image_size
        self.img_cols = image_size
        self.parallel_model = None
        self.history = None
        self.patient = None
        self.data_source = None
        self.file_source = None
        self.image_size = None
        self.source_type = None
        self.test_source_path = None
        self.model_type = model_type
        self.image_4d_file = None
        self.image_source_file = None
        self.image_one_file = None
        self.sourcedict = dict()
        self.model_name = model_name


        if model_type == "small":
            self.build_unet_small()
        elif model_type == "large":
            self.build_unet()
        elif model_type == "large2":
            self.build_unet()
        else :
            print ("Specify valid model_type (small, large, large2)")
            return
    

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
        
        #self.model.compile(optimizer = Adam(lr = 1e-4), loss = dice_loss, metrics = [dice_coeff])

        self.parallel_model = ModelMGPU(self.model, 3)
        self.parallel_model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = [dice_coeff])
        #jh self.model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = [dice_coeff])
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

    def train_and_predict(self, model_path, batch_size = 4, nb_epoch = 10, augmentation = False): 
        model_file = model_path + model_name + '.hdf5'
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
        self.history = self.parallel_model.fit(self.train_images, self.train_labels, batch_size, nb_epoch, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])
        #jh self.history = self.model.fit(self.train_images, self.train_labels, batch_size, nb_epoch, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])

        print('-'*30)
        print('predict test data....')
        self.predictions = self.parallel_model.predict(self.test_images, batch_size=1, verbose=1)
        #jh self.predictions = self.model.predict(self.test_images, batch_size=1, verbose=1)
        self.scores = self.parallel_model.evaluate (self.predictions, self.test_labels, batch_size=4)
        #jh self.scores = self.model.evaluate (self.predictions, self.test_labels, batch_size=4)
        print ("Prediction Scores", self.scores)
        print('-'*30)
        
        

    def train_with_augmentation(self, model_file, batch_size = 4, nb_epoch = 10 ):

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
        
        self.parallel_model.fit(self.train_images, self.train_labels, batch_size, nb_epoch, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])
        #jh self.model.fit(self.train_images, self.train_labels, batch_size, nb_epoch, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])
        
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
        
            
    def save_model_info(self, mypath = "./"):
        learn_file =  self.model_name + "_learning_history.json"
        learn_file = mypath + learn_file
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
        self.perf = get_performance_statistics (self.test_labels, self.predictions)
        print ("Perf Statistics: ", self.perf)
        perf_file =  self.model_name + "_performance.json"
        perf_file = mypath + perf_file
        
        print ("Saving Performance values", perf_file)
        with open(perf_file, 'w') as file:
            json.dump(self.perf, file, indent=2)
        print('-'*30)


            
#################
# Method to create a U-net model and train it
# Create a U-Net model, train the model and run the predictions and save the trained weights and predictions
#
##########################
# /masvol/heartsmart/unet_model/data/sunnybrook_176_train_images.npy 
# /masvol/heartsmart/unet_model/data/sunnybrook_176_train_images.npy 
# /masvol/heartsmart/unet_model/data/sunnybrook_176_train_labels.npy

def train_unet_model (model_name, image_size, training_images, training_labels, test_images, test_labels, model_path,  batch_size, epochs, augmentation = False, model_summary = False):

    #samples, x, y, z = pred.shape
    
    train_data = {}
    test_data = {}

    train_data["images"] = training_images
    train_data["labels"] = training_labels
    test_data["images"] = test_images
    test_data["labels"] = test_labels

    
    if not os.path.exists(model_path):
        print ("creating dir ", model_path)
        os.makedirs(model_path)
            
    # get the u-net model and load train and test data
    myunet = myUnet(model_name = model_name, image_size = image_size, model_type = "large")
    myunet.load_data (train_data, test_data)

    if (model_summary == True):
        myunet.model.summary()
        
    res = myunet.train_and_predict(model_path, batch_size = batch_size, nb_epoch = epochs, augmentation = augmentation)
    
#     if (augmentation == True) :
#         res = myunet.train_with_augmentation(model_file, batch_size = batch_size, nb_epoch = epochs)
#     else :
#         res = myunet.train_and_predict(model_file, batch_size = batch_size, nb_epoch = epochs)
        
    return myunet


if __name__ == "__main__":
    img_size_list = [176, 256]
    args = sys.argv[1:]
    print ("total arguments", len(args), args)
    if len(args) != 11:
        print ("insufficient arguments ")
        print (" enter model_name, image_size, training_images, training_labels, test_images, test_labels, model_path,  batch_size, epochs, augmentation, model_summary")
        sys.exit() 
    
    model_name = sys.argv[1]
    image_size = int(sys.argv[2])
    training_images = sys.argv[3]
    training_labels = sys.argv[4]
    test_images = sys.argv[5]
    test_labels = sys.argv[6]
    model_path = sys.argv[7]
    batch_size = int(sys.argv[8])
    epochs = int(sys.argv[9])
    augmentation = sys.argv[10]
    model_summary = sys.argv[11]
    

    if image_size not in img_size_list:
        print ("image size %d is not supported"%image_size)
        sys.exit()
    
    mymodel = train_unet_model (model_name, image_size, training_images, training_labels, test_images, test_labels, model_path,  batch_size, epochs, augmentation, model_summary)
    
    mymodel.save_model_info(model_path)           


