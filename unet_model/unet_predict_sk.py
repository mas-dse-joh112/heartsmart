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

from helper_utilities import *
from helper_plotutilities import *
from loss_functions  import *

#from unet_model_batchnormal  import *
from unet_model  import *



#from helpers_dicom import DicomWrapper as dicomwrapper

#Fix the random seeds for numpy (this is for Keras) and for tensorflow backend to reduce the run-to-run variance
from numpy.random import seed
seed(12345)
from tensorflow import set_random_seed
set_random_seed(12345)

#import matplotlib.pyplot as plt
#%matplotlib inline

import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
#os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

from modelmgpu import ModelMGPU
#GPUs = 1

import time

start = time.time()
print("START:", start)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.90
session = tf.Session(config=config)
print("\nSuccessfully imported packages!!!\n")


            
#################
# Method to create a U-net model and train it
# Create a U-Net model, train the model and run the predictions and save the trained weights and predictions
#
##########################
# /masvol/heartsmart/unet_model/data/sunnybrook_176_train_images.npy 
# /masvol/heartsmart/unet_model/data/sunnybrook_176_train_images.npy 
# /masvol/heartsmart/unet_model/data/sunnybrook_176_train_labels.npy

def train_unet_model (model_name, nGPU, image_size, training_images, training_labels, test_images, test_labels, model_path, dropout, optimizer, learningrate, lossfun, batch_size, epochs, augmentation = False, model_summary = False):
    
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
    myunet = myUnet(model_name = model_name, nGPU = nGPU, image_size = image_size, dropout = dropout, optimizer = optimizer, lr=learningrate, loss_fn = lossfn)
    myunet.load_data (train_data, test_data)

    if (model_summary == True):
        print ("Printing model summary ")
        #myunet.model.summary()
        myunet.parallel_model.summary()
        
    res = myunet.train_and_predict(model_path, batch_size = batch_size, nb_epoch = epochs, augmentation = augmentation)
    
#     if (augmentation == True) :
#         res = myunet.train_with_augmentation(model_file, batch_size = batch_size, nb_epoch = epochs)
#     else :
#         res = myunet.train_and_predict(model_file, batch_size = batch_size, nb_epoch = epochs)
        
    return myunet

def predict_with_pretrained_weights(model_name, nGPU, model_file, image_size, test_image_file, test_label_file = "none"):
    img_size_list = [176, 256]
    if image_size not in img_size_list:
        print ("image size %d is not supported"%image_size)
        return 
    
    size_str = str(image_size)
    
    test_data = {}
    test_data["images"] = test_image_file
    test_data["labels"] = test_label_file

    model_file = model_file

    print('-'*30)
    print ("Get Test images and labels...")
    if test_label_file == "none": 
        ts = load_images(test_image_file, normalize= True)
        tl = "none"
    else :
        ts , tl= load_images_and_labels(test_data, normalize= True)

    print('-'*30)
    print ("Creating U-net model...")
    myunet = myUnet(model_name = model_name, nGPU = nGPU, image_size = image_size)
    print('-'*30)
    print ("Loading the pre-trained weights...")
    myunet.load_pretrained_weights(model_file)
    print('-'*30)

    print('Run predictions...')
    myunet.predict(test_image_array = ts, test_label_array = tl)
    print('-'*30)

    return myunet



if __name__ == "__main__":
    img_size_list = [176, 256]
    args = sys.argv[1:]
    print ("total arguments", len(args), args)
    if len(args) != 5:
        print ("insufficient arguments ")
        print (" enter model_name, model_file, image_size, test_images, test_labels")
        sys.exit() 
    
    model_name = sys.argv[1]
    model_path = sys.argv[2]
    image_size = int(sys.argv[3])
    test_images = sys.argv[4]
    test_labels = sys.argv[5]

    if image_size not in img_size_list:
        print ("image size %d is not supported"%image_size)
        sys.exit()
    
    mymodel = predict_with_pretrained_weights (model_name, model_path, image_size, test_images, test_labels)
    
    mymodel.save_prediction_info(model_path) 
