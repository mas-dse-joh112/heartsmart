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

GPU_CLUSTER = "0,1"
#GPU_CLUSTER = "2,3"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_CLUSTER
GPUs = len(GPU_CLUSTER.split(','))

#os.environ["CUDA_VISIBLE_DEVICES"] = "5"
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

def train_unet_model (model_name, image_size, training_images, training_labels, test_images, test_labels, model_path, dropout, optimizer, learningrate, lossfun, batch_size, epochs, augmentation = False, model_summary = False):
    
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
    myunet = myUnet(model_name = model_name, nGPU= GPUs, image_size = image_size, dropout = dropout, optimizer = optimizer, lr=learningrate, loss_fn = lossfn)
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


if __name__ == "__main__":
    img_size_list = [176, 256]
    args = sys.argv[1:]
    print ("total arguments", len(args), args)
    if len(args) != 15:
        print ("insufficient arguments ")
        print (" enter model_name, image_size, training_images, training_labels, test_images, test_labels, model_path, dropout (True or False), optimizer, learningrate, loss_function, batch_size, epochs, augmentation (True or False), model_summary (True or False)")
        sys.exit() 
    
    model_name = sys.argv[1]
    image_size = int(sys.argv[2])
    training_images = sys.argv[3]
    training_labels = sys.argv[4]
    test_images = sys.argv[5]
    test_labels = sys.argv[6]
    model_path = sys.argv[7]
    if (sys.argv[8] == "True"):
        dropout = True
    else:
        dropout = False
        
    optimizer = sys.argv[9]
    lr = float(sys.argv[10])
    lossfn = sys.argv[11]
    batch_size = int(sys.argv[12])
    epochs = int(sys.argv[13])
    if (sys.argv[14] == "True"):
        augmentation = True
    else:
        augmentation = False
        
    if (sys.argv[15] == "True"):
        model_summary = True
    else:
        model_summary = False

    
    if image_size not in img_size_list:
        print ("image size %d is not supported"%image_size)
        sys.exit()
    
    mymodel = train_unet_model (model_name, image_size, training_images, training_labels, test_images, test_labels, model_path, dropout, optimizer, lr, lossfn, batch_size, epochs, augmentation, model_summary)
    
    mymodel.save_model_info(model_path)           


    end = time.time()
    print ("END:", end - start)
