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
#from helper_plotutilities import *
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

GPU_CLUSTER = "4,5"
#GPU_CLUSTER = "2,3"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_CLUSTER
GPUs = len(GPU_CLUSTER.split(','))

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
# Method load pre-trained weights to unet-model and run predictions 
#
##########################


def predict_with_pretrained_weights(model_name, nGPU, model_file, image_size, test_image_file, test_label_file = "none", augmentation = False, contrast_normalize = False):
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
    img_size_list = [176, 256]
    args = sys.argv[1:]
    print ("total arguments", len(args), args)
    if len(args) != 8:
        print ("insufficient arguments ")
        print (" enter model_name, model_file (weights file), image_size, image_file, label_file(enter none if no labels), save_folder(to save predictions), augmentation (True or False), contrast_normalization (True or False)")
        sys.exit() 
    
    model_name = sys.argv[1]
    image_size = int(sys.argv[2])
    if image_size not in img_size_list:
        print ("image size %d is not supported"%image_size)
        sys.exit()
    weights_file = sys.argv[3]   
    image_file = sys.argv[4]
    label_file = sys.argv[5]
    save_folder = sys.argv[6]
    if (sys.argv[7] == "True"):
        augment = True
    else:
        augment = False
      
    if (sys.argv[8] == "True"):
        contrast_norm = True
    else:
        contrast_norm = False

          
    mymodel = predict_with_pretrained_weights(model_name=model_name,nGPU=0,model_file=weights_file,image_size=image_size, \
           test_image_file=image_file, test_label_file=label_file, augmentation=augment, contrast_normalize=contrast_norm)
    
    pred_file =  save_folder + model_name + "_predictions.npy"
    print ("Saving predictions", pred_file)
    np.save(pred_file, mymodel.predictions)
    
    
'''
Sample command : 

python3 /masvol/heartsmart/unet_model/unet_predict_sk.py pred_test 176 \
/masvol/heartsmart/unet_model/baseline/results/experiments/1_3_0_176_CLAHE_augx_bce1.hdf5 \
/masvol/output/dsb/norm/outlier_testing/m1t3/dsb_315_176_test.npy  none \
/masvol/heartsmart/unet_model/baseline/results/experiments/ False True   > runlogpred_test.txt &
'''
  
