#!/usr/bin/env python

""" Module for predicting images files with the supplied of the model file
    Parameters are set in a config file passed in
"""

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
#from helper_plotutilities import *
from loss_functions  import *

#from unet_model_batchnormal  import *
from unet_model import *

#Fix the random seeds for numpy (this is for Keras) and for tensorflow backend to reduce the run-to-run variance
from numpy.random import seed
seed(12345)
from tensorflow import set_random_seed
set_random_seed(12345)

import tensorflow as tf

GPU_CLUSTER = "0,1,2,3,4,5,6,7" # set in config file, override later
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_CLUSTER
GPUs = None

from modelmgpu import ModelMGPU

import time

start = time.time()
print("START:", start)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

print("\nSuccessfully imported packages!!!\n")

#################
# Method load pre-trained weights to unet-model and run predictions 
#
##########################


def predict_with_pretrained_weights(model_name, nGPU, model_file, image_size, test_image_file, test_label_file = "none", augmentation = False, contrast_normalize = False):
    """
    

    Args:
      model_name:  unique model name for identificatio
      nGPU: number of GPU devices using
      model_file: model file for predicting
      image_size: uniform image size
      test_image_file: image files in 4d numpy array for testing
      test_label_file:  (Default value = "none"), label files or none
      augmentation:  (Default value = False)
      contrast_normalize:  (Default value = False)

    Returns: U-net model instance

    """
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
        #ts = load_images(test_image_file, normalize= True, contrast_normalize = True)
        ts = load_images(test_image_file, normalize= True, contrast_normalize = contrast_normalize)
        tl = "none"
    else :
        ts , tl= load_images_and_labels(test_data, normalize= True, contrast_normalize = contrast_normalize)

    print('-'*30)
    print ("Creating U-net model...")
    myunet = myUnet(model_name = model_name, nGPU = nGPU, image_size = image_size)
    print('-'*30)
    print ("Loading the pre-trained weights...")
    myunet.load_pretrained_weights(model_file)
    print('-'*30)

    print('Run predictions...')
    myunet.predict(test_image_array = ts, test_label_array = tl, augmentation=augmentation)
    print('-'*30)

    return myunet


if __name__ == "__main__":
    if len(sys.argv) < 1:
        print ('Provide a config file')

    myconfig = sys.argv[1]

    args = __import__(myconfig)

    img_size_list = [176, 256]

    arg_list = ['model_name','image_size','weights_file','image_file','label_file','save_folder','augmentation','contrast_normalization']
    dir_args = dir(args)

    for x in arg_list:
        if x not in dir_args:
            print ("insufficient arguments ")
            print (" enter model_name, image_size, model_file (weights file), image_file, label_file(enter none if no labels), save_folder(to save predictions), augmentation (True or False), contrast_normalization (True or False) in the config file")
            sys.exit() 

    image_size = args.image_size

    if image_size not in img_size_list:
        print ("image size %d is not supported"%image_size)
        sys.exit()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_CLUSTER
    GPUs = len(args.GPU_CLUSTER.split(','))
    config.gpu_options.per_process_gpu_memory_fraction = args.per_process_gpu_memory_fraction
    session = tf.Session(config=config)

    model_name = args.model_name
    image_size = args.image_size
    weights_file = args.weights_file
    image_file = args.image_file
    label_file = args.label_file
    save_folder = args.save_folder
    augment = args.augmentation
    contrast_norm = args.contrast_normalization

    mymodel = predict_with_pretrained_weights(model_name=model_name,nGPU=0,model_file=weights_file,image_size=image_size, \
           test_image_file=image_file, test_label_file=label_file, augmentation=augment, contrast_normalize=contrast_norm)
    
    pred_file =  save_folder + model_name + "_predictions.npy"
    print ("Saving predictions", pred_file)
    np.save(pred_file, mymodel.predictions)
