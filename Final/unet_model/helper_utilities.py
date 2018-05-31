#########################################
#
# Helper utilities
# 
#
#########################################
import re, sys, math
import glob
import random
import json
import os 
import numpy as np
import cv2
from sklearn.metrics import log_loss
from sklearn.metrics import auc, roc_curve, accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from collections import OrderedDict
from helpers_dicom import DicomWrapper as dicomwrapper

perf_keys = ["samples", "logloss", "weighted_logloss","accuracy", "weighted_accuracy", "dice_coef", "precision","recall", \
             "f1_score", "true_positive", "false_positive","true_negative","false_negative", "zero_contour_labels", \
             "zero_contour_pred", "missed_pred_lt_05", "missed_pred_gt_25", "missed_pred_gt_50", "missed_pred_eq_100"]

perf_keys_ext=["tr_model_name","tr_nGPUs", "tr_loss_fn","tr_dropout", "tr_optimizer","tr_lrrate","tr_batchsize","tr_epoch", \
               "tr_size","tr_contrast_norm","tr_augmentation","tr_augment_count","tr_augment_shift_h", "tr_augment_shift_w", \
               "tr_augment_rotation","tr_augment_zoom","eval_loss","eval_dice_coeff","eval_binary_accuracy","logloss", \
               "accuracy","weighted_logloss", "weighted_accuracy","precision","recall","f1_score", "true_positive", \
             "true_negative", "false_positive", "false_negative"]
def get_perf_keys():
    return perf_keys

def get_perf_keys_ext():
    return perf_keys_ext

def load_images_and_labels(data, normalize= True, zscore_normalize = False, contrast_normalize= False, cliplimit=2, tilesize=8 ):
    """Function to load images and labels from .npy file into a 4d numpy array
    before we feed them into U-net model. 

    Note:
    Image files will be normalized based on pixel values by default.       

    Args:
        data (:dict): dictionary with full path to .npy files for images and labels.
        normalize (:boolean, optional): Applies pixel normalization if True. Default value is True.
    
    Returns:
       numpy arrays  of images and labels.
    """
    print('-'*30)
    print('load np arrays of images and labels...')
    print('-'*30)
    imgfile = data["images"]
    labelfile = data["labels"]
    print ("Loading files : ", imgfile, labelfile)
    
    im = np.load(imgfile)
    lb = np.load(labelfile)
    if (contrast_normalize == True):
        # Apply contrast normalization
            #create a CLAHE object (Arguments are optional).
        clahe = cv2.createCLAHE(clipLimit=cliplimit, tileGridSize=(tilesize, tilesize))
        print ("CLAHE clip limit, tile size",cliplimit, tilesize )
        imgs_equalized = np.empty(im.shape)
#         for i in range(imgs.shape[0]):
#             imgs_equalized[i,0] = clahe.apply(np.array(imgs[i,0], dtype = np.uint8))
        for i in range(im.shape[0]):
            imgs_equalized[i,:,:,0] = clahe.apply(np.array(im[i,:,:,0], dtype = np.uint16))
            
        print("shape, max, min, mean of original image set:", im.shape, im.max(), im.min(), im.mean())
        im = imgs_equalized
        
        print("shape, max, min, mean of CLAHE image set:", imgs_equalized.shape, imgs_equalized.max(), imgs_equalized.min(), imgs_equalized.mean())
        
    if (normalize == True):
        images = im.astype('float32')
        labels = lb.astype('float32')
        ##Normalize the pixel values, (between 0..1)
        x_min = images.min(axis=(1, 2), keepdims=True)
        x_max = images.max(axis=(1, 2), keepdims=True)
        images2 = (images - x_min)/(x_max-x_min)
        print("shape, max, min, mean of original image set:", images.shape, images.max(), images.min(), images.mean())
        print("shape, max, min, mean after normalization  :", images2.shape, images2.max(), images2.min(), images2.mean())
        print("shape, max, min, mean of labels :", labels.shape, labels.max(), labels.min(), labels.mean())
    elif (zscore_noramlize == True):
        images = im.astype('float32')
        labels = lb.astype('float32')
        ##Normalize the pixel values, (between 0..1)
        x_mean = images.mean(axis=(1, 2), keepdims=True)
        x_std = images.std(axis=(1, 2), keepdims=True)
        images2 = (images - x_mean)/(x_std)
        print("shape, max, min, mean of original image set:", images.shape, images.max(), images.min(), images.mean())
        print("shape, max, min, mean after zscore normalization  :", images2.shape, images2.max(), images2.min(), images2.mean())
        print("shape, max, min, mean of labels :", labels.shape, labels.max(), labels.min(), labels.mean())
    else :
        images2 = im
        labels = lb
        print("shape, max, min, mean of images :", images2.shape, images2.max(), images2.min(), images2.mean())
        print("shape, max, min, mean of labels :", labels.shape, labels.max(), labels.min(), labels.mean())
    
    return images2, labels

def load_images_and_labels_contrast(data, normalize= True, cliplimit=2, tilesize=8):
    """Function to load images and labels from .npy file into a 4d numpy array
    before we feed them into U-net model. 

    Note:
    Image files will be normalized based on pixel values by default.       

    Args:
        data (:dict): dictionary with full path to .npy files for images and labels.
        normalize (:boolean, optional): Applies pixel normalization if True. Default value is True.
    
    Returns:
       numpy arrays  of images and labels.
    """
    print('-'*30)
    print('load np arrays of images and labels...')
    print('-'*30)
    imgfile = data["images"]
    labelfile = data["labels"]
    print ("Loading files : ", imgfile, labelfile)
    
    im = np.load(imgfile)
    lb = np.load(labelfile)
    if (normalize == True):
    # Apply contrast normalization
            #create a CLAHE object (Arguments are optional).
        clahe = cv2.createCLAHE(clipLimit=cliplimit, tileGridSize=(tilesize, tilesize))
        print ("CLAHE clip limit, tile size",cliplimit, tilesize )
        imgs_equalized = np.empty(im.shape)
#         for i in range(imgs.shape[0]):
#             imgs_equalized[i,0] = clahe.apply(np.array(imgs[i,0], dtype = np.uint8))
        for i in range(im.shape[0]):
            imgs_equalized[i,:,:,0] = clahe.apply(np.array(im[i,:,:,0], dtype = np.uint16))
        #return imgs_equalized
        images = imgs_equalized.astype('float32')
        labels = lb.astype('float32')
        ##Normalize the pixel values, (between 0..1)
        x_min = images.min(axis=(1, 2), keepdims=True)
        x_max = images.max(axis=(1, 2), keepdims=True)
        images2 = (images - x_min)/(x_max-x_min)
        print("shape, max, min, mean of original image set:", images.shape, images.max(), images.min(), images.mean())
        print("shape, max, min, mean of CLAHE image set:", imgs_equalized.shape, imgs_equalized.max(), imgs_equalized.min(), imgs_equalized.mean())
        print("shape, max, min, mean after normalization  :", images2.shape, images2.max(), images2.min(), images2.mean())
        print("shape, max, min, mean of labels :", labels.shape, labels.max(), labels.min(), labels.mean())
    
    else :
        images2 = im
        labels = lb
        print("shape, max, min, mean of images :", images2.shape, images2.max(), images2.min(), images2.mean())
        print("shape, max, min, mean of labels :", labels.shape, labels.max(), labels.min(), labels.mean())
    
    return images2, labels


def load_images_and_labels2(data, normalize= True):
    """Function to load images and labels from .npy file into a 4d numpy array
    before we feed them into U-net model The data is normalized using Z-score normalization technique. 

    Note:
    Image files will be normalized based on pixel values by default.       

    Args:
        data (:dict): dictionary with full path to .npy files for images and labels.
        normalize (:boolean, optional): Applies pixel normalization if True. Default value is True.
    
    Returns:
       numpy arrays  of images and labels.
    """
    print('-'*30)
    print('load np arrays of images and labels...')
    print('-'*30)
    imgfile = data["images"]
    labelfile = data["labels"]
    print ("Loading files : ", imgfile, labelfile)
    
    im = np.load(imgfile)
    lb = np.load(labelfile)
    if (normalize == True):
        images = im.astype('float32')
        labels = lb.astype('float32')
        ##Normalize the pixel values, (between 0..1)
        x_mean = images.mean(axis=(1, 2), keepdims=True)
        x_std = images.std(axis=(1, 2), keepdims=True)
        images2 = (images - x_mean)/(x_std)
        print("shape, max, min, mean of original image set:", images.shape, images.max(), images.min(), images.mean())
        print("shape, max, min, mean after normalization  :", images2.shape, images2.max(), images2.min(), images2.mean())
        print("shape, max, min, mean of labels :", labels.shape, labels.max(), labels.min(), labels.mean())
    
    else :
        images2 = im
        labels = lb
        print("shape, max, min, mean of images :", images2.shape, images2.max(), images2.min(), images2.mean())
        print("shape, max, min, mean of labels :", labels.shape, labels.max(), labels.min(), labels.mean())
    
    return images2, labels

def load_images_and_labels_no_preproc(data):
    print('-'*30)
    print('load np arrays of images and labels...')
    print('-'*30)
    imgfile = data["images"]
    labelfile = data["labels"]
    print ("Loading files : ", imgfile, labelfile)

    images = np.load(imgfile)
    labels = np.load(labelfile)
#     im = np.load(imgfile)
#     lb = np.load(labelfile)
#     images = im.astype('float32')
#     labels = lb.astype('float32')
    
#     ##Normalize the pixel values, (between 0..1)
#     x_min = images.min(axis=(1, 2), keepdims=True)
#     x_max = images.max(axis=(1, 2), keepdims=True)
#     images2 = (images - x_min)/(x_max-x_min)

    print("shape, max, min, mean of original image set:", images.shape, images.max(), images.min(), images.mean())
#    print("shape, max, min, mean after normalization  :", images2.shape, images2.max(), images2.min(), images2.mean())
    print("shape, max, min, mean of labels :", labels.shape, labels.max(), labels.min(), labels.mean())
    return images, labels


def load_images(imgfile, normalize= True, zscore_normalize = False, contrast_normalize= False, cliplimit=2, tilesize=8 ):
    """Function to load images  from .npy file into a 4d numpy array
    before we feed them into U-net model. 

    Note:
    Image files will be normalized based on pixel values by default.       

    Args:
        imgfile (:string): .npy file name with full path for images .
        normalize (:boolean, optional): Applies pixel normalization if True. Default value is True.
    
    Returns:
       numpy array  of images.
       
    """
    print('-'*30)
    print('load np arrays of images ...')
    print('-'*30)
    print ("Loading files : ", imgfile)
    
    im = np.load(imgfile)
    if (contrast_normalize == True):
        # Apply contrast normalization
            #create a CLAHE object (Arguments are optional).
        clahe = cv2.createCLAHE(clipLimit=cliplimit, tileGridSize=(tilesize, tilesize))
        print ("CLAHE clip limit, tile size",cliplimit, tilesize )
        imgs_equalized = np.empty(im.shape)
#         for i in range(imgs.shape[0]):
#             imgs_equalized[i,0] = clahe.apply(np.array(imgs[i,0], dtype = np.uint8))
        for i in range(im.shape[0]):
            imgs_equalized[i,:,:,0] = clahe.apply(np.array(im[i,:,:,0], dtype = np.uint16))
            
        print("shape, max, min, mean of original image set:", im.shape, im.max(), im.min(), im.mean())
        im = imgs_equalized
        
        print("shape, max, min, mean of CLAHE image set:", imgs_equalized.shape, imgs_equalized.max(), imgs_equalized.min(), imgs_equalized.mean())
        
    if (normalize == True):
        images = im.astype('float32')
        ##Normalize the pixel values, (between 0..1)
        x_min = images.min(axis=(1, 2), keepdims=True)
        x_max = images.max(axis=(1, 2), keepdims=True)
        images2 = (images - x_min)/(x_max-x_min)
        print("shape, max, min, mean of original image set:", images.shape, images.max(), images.min(), images.mean())
        print("shape, max, min, mean after normalization  :", images2.shape, images2.max(), images2.min(), images2.mean())
    elif (zscore_noramlize == True):
        images = im.astype('float32')
        ##Normalize the pixel values, (between 0..1)
        x_mean = images.mean(axis=(1, 2), keepdims=True)
        x_std = images.std(axis=(1, 2), keepdims=True)
        images2 = (images - x_mean)/(x_std)
        print("shape, max, min, mean of original image set:", images.shape, images.max(), images.min(), images.mean())
        print("shape, max, min, mean after zscore normalization  :", images2.shape, images2.max(), images2.min(), images2.mean())
    else :
        images2 = im
        print("shape, max, min, mean of images :", images2.shape, images2.max(), images2.min(), images2.mean())
   
    return images2

def load_images_contrast(imgfile, normalize= True):
    """Function to load images  from .npy file into a 4d numpy array
    before we feed them into U-net model. 

    Note:
    Image files will be normalized based on pixel values by default.       

    Args:
        imgfile (:string): .npy file name with full path for images .
        normalize (:boolean, optional): Applies pixel normalization if True. Default value is True.
    
    Returns:
       numpy array  of images.
       
    """
    print('-'*30)
    print('load np arrays of images ...')
    print('-'*30)
    print ("Loading files : ", imgfile)
    
    im = np.load(imgfile)
    if (normalize == True):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        imgs_equalized = np.empty(imgs.shape)
#         for i in range(imgs.shape[0]):
#             imgs_equalized[i,0] = clahe.apply(np.array(imgs[i,0], dtype = np.uint8))
        for i in range(im.shape[0]):
            imgs_equalized[i,:,:,0] = clahe.apply(np.array(im[i,:,:,0], dtype = np.uint16))
        #return imgs_equalized
        images = imgs_equalized.astype('float32')
        ##Normalize the pixel values, (between 0..1)
        x_min = images.min(axis=(1, 2), keepdims=True)
        x_max = images.max(axis=(1, 2), keepdims=True)
        images2 = (images - x_min)/(x_max-x_min)
        print("shape, max, min, mean of original image set:", images.shape, images.max(), images.min(), images.mean())
        print("shape, max, min, mean of CLAHE image set:", imgs_equalized.shape, imgs_equalized.max(), imgs_equalized.min(), imgs_equalized.mean())
        print("shape, max, min, mean after normalization:", images2.shape, images2.max(), images2.min(), images2.mean())
    else :
        images2 = im
        print("shape, max, min, mean of images:", images2.shape, images2.max(), images2.min(), images2.mean())
   
    return images2

def load_images2(imgfile, normalize= True):
    """Function to load images  from .npy file into a 4d numpy array
    before we feed them into U-net model. 

    Note:
    Image files will be normalized based on pixel values by default.       

    Args:
        imgfile (:string): .npy file name with full path for images .
        normalize (:boolean, optional): Applies pixel normalization if True. Default value is True.
    
    Returns:
       numpy array  of images.
       
    """
    print('-'*30)
    print('load np arrays of images ...')
    print('-'*30)
    print ("Loading files : ", imgfile)
    
    im = np.load(imgfile)
    if (normalize == True):
        images = im.astype('float32')
        ##Normalize the pixel values, (between 0..1)
        x_mean = images.mean(axis=(1, 2), keepdims=True)
        x_std = images.std(axis=(1, 2), keepdims=True)
        images2 = (images - x_mean)/(x_std)
        print("shape, max, min, mean of original image set:", images.shape, images.max(), images.min(), images.mean())
        print("shape, max, min, mean after normalization:", images2.shape, images2.max(), images2.min(), images2.mean())
    else :
        images2 = im
        print("shape, max, min, mean of images:", images2.shape, images2.max(), images2.min(), images2.mean())
   
    return images2

def read_performance_statistics(file_p):
    """Function to read performance statistics captured during model training.       

    Args:
        file_p(:string): performance statistics file (.json) with full path.

    Returns:
       None.
       
    """
    perf_list = ['logloss', 'weighted_logloss', 'accuracy', 'weighted_accuracy','true_positive', 'false_positive', 'true_negative','false_negative', \
                 'precision','recall', 'f1_score' ]
    #perf = OrderedDict.fromkeys(perf_keys_ext)
    try:
        with open(file_p, 'r') as file:
            perf = json.load(file)
    except (OSError, ValueError):  # file does not exist or is empty/invalid
        print ("File does not exist")
        perf = {}
        
    print('-'*30)
    print ("Model Parameters")
    for key in perf:
        if (key.startswith("tr_") == True):
            print (key, " : ", perf[key])
    print('-'*30)
    print ("Evaluation on Test set")
    for key in perf:
        if (key.startswith("eval_") == True):
            print (key, " : ", perf[key])

    print('-'*30)
    for key in perf_list:
        if key in perf.keys():
            print (key, " : ", perf[key])
    print('-'*30) 
    print('-'*30)
    return perf
    # list all data in history
    
def get_performance_statistics(y_true_f, y_pred_f):
    """Function to plot learning history captured during model training.       

    Args:
        file_p(:string): learning history file (.json) with full path.

    Returns:
       perf(:dict): dictionary of perf statistics
       
    """   
    
#     y_true = np.load(y_true_f)
#     y_pred = np.load(y_pred_f)

    y_true = y_true_f.flatten()
    y_pred = y_pred_f.flatten()

    sample_weights = np.copy(y_true)
    sample_weights[sample_weights == 1] = 1.
    sample_weights[sample_weights == 0] = .2
    
    epsilon = 1e-7
    y_pred[y_pred<=0.] = epsilon
    y_pred[y_pred>=1.] = 1. -epsilon
    
    perf = {}
    
    score = log_loss (y_true, y_pred)
    score2 = log_loss (y_true, y_pred, sample_weight = sample_weights)
    perf["logloss"] = score
    perf["weighted_logloss"] = score2
    perf["accuracy"] = math.exp(-score)
    perf["weighted_accuracy"] = math.exp(-score2)

    y_pred = np.round(y_pred)
    perf["precision"] = precision_score(y_true, y_pred, average="binary")
    perf["recall"] = recall_score(y_true, y_pred, average="binary")
    perf["f1_score"] = f1_score(y_true, y_pred, average="binary")

    cm = confusion_matrix(y_true, y_pred)
    perf["true_positive"] = int(cm[1][1])
    perf["false_positive"] = int(cm[0][1])
    perf["true_negative"] = int(cm[0][0])
    perf["false_negative"] = int(cm[1][0])
    
    #cm.print_stats()
    return perf

def compute_roc_auc(y_true_f, y_pred_f):
    """Function to plot learning history captured during model training.       

    Args:
        file_p(:string): learning history file (.json) with full path.

    Returns:
       perf(:dict): dictionary of perf statistics
       
    """   
    
    y_true_a = np.load(y_true_f)
    y_pred_a = np.load(y_pred_f)

    y_true = y_true_a.flatten()
    y_pred = y_pred_a.flatten()
    
    fpr, tpr, threshold = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, threshold, roc_auc
    

def compute_performance_statistics (y_true_f, y_pred_f):
    """Function to compute performanc statistics using labels and prections.       

    Args:
        y_true_f(:string):  label file (.npy) with full path.
        y_pred_f(:string):  predictions file (.npy) with full path.
        
    Returns:
       perf(:dict): dictionary of perf statistics.
    
    Note:
        prection file should have the sigmoid outputs (not the rounded values).
    """
    
    y_true = np.load(y_true_f)
    y_pred = np.load(y_pred_f)
    
    y_true_o = np.load(y_true_f)
    y_pred_o = np.load(y_pred_f)
    #print (y_true.shape, y_pred.shape)
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    sample_weights = np.copy(y_true)
    sample_weights[sample_weights == 1] = 1.
    sample_weights[sample_weights == 0] = .2
    
    
    epsilon = 1e-7
    y_pred[y_pred<=0.] = epsilon
    y_pred[y_pred>=1.] = 1. -epsilon
    
    #print (y_true.shape, y_pred.shape)
    smooth = 1.
    intersection = np.sum(y_true * y_pred)
    dice_coef = (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)

    score = log_loss (y_true, y_pred)
    score2 = log_loss (y_true, y_pred, sample_weight = sample_weights)
    acc = math.exp(-score)
    acc2 = math.exp(-score2)
    y_pred = np.round(y_pred)

    prec = precision_score(y_true, y_pred, average="binary")
    rec = recall_score(y_true, y_pred, average="binary")
    f1 = f1_score(y_true, y_pred, average="binary")

    
    cm = confusion_matrix(y_true, y_pred)
    #cm.print_stats()
    true_p = cm[1][1]
    false_p = cm[0][1]
    true_n = cm[0][0]
    false_n = cm[1][0]

    
    #perf = {}
    
#     keys = ["samples", "logloss", "weighted_logloss","accuracy", "weighted_accuracy", "dice_coef", "precision","recall", "f1_score", "true_positive", \
#            "false_positive","true_negative","false_negative", "zero_contour_labels", "zero_contour_pred", \
#            "missed_pred_lt_05", "missed_pred_gt_25", "missed_pred_gt_50", "missed_pred_eq_100"]
    perf = OrderedDict.fromkeys(perf_keys)
    
    perf["logloss"] = score
    perf["weighted_logloss"] = score2
    perf["accuracy"] = acc
    perf["weighted_accuracy"] = acc2

    perf["dice_coef"] = dice_coef
    perf["precision"] = prec
    perf["recall"] = rec
    perf["f1_score"] = f1
    perf["true_positive"] = int(cm[1][1])
    perf["false_positive"] = int(cm[0][1])
    perf["true_negative"] = int(cm[0][0])
    perf["false_negative"] = int(cm[1][0])
    
    y_true = y_true_o
    y_pred = np.round(y_pred_o)
    samples, x, y, z = y_pred.shape
    y_true_sum = y_true.sum(axis=(1, 2), keepdims=True).reshape(samples)
    y_pred_sum = y_pred.sum(axis=(1, 2), keepdims=True).reshape(samples)  
    lb0 = (np.where(y_true_sum == 0))
    pd0 = (np.where(y_pred_sum == 0))
    lb0 = list(lb0[0])
    pd0 = list(pd0[0])
    perf["samples"] = samples
    perf["zero_contour_labels"] = len(lb0)
    perf["zero_contour_pred"] = len(pd0)
    
    pix_diff = (abs(y_true_sum - y_pred_sum))/(y_true_sum + epsilon)
    px1 = np.where(pix_diff <.0005)
    px1 = list(px1[0])
    px25 = np.where(pix_diff>.25)
    px25 = list(px25[0])
    px50 = np.where(pix_diff>.5)
    px50 = list(px50[0])
    px100 = np.where(pix_diff >= 1.0) 
    px100 = list(px100[0])
    perf["missed_pred_lt_05"] = len(px1)
    perf["missed_pred_gt_25"] = len(px25)
    perf["missed_pred_gt_50"] = len(px50)
    perf["missed_pred_eq_100"] = len(px100)
    return perf


    
def find_outliers_in_prediction(y_pred_f):
    """Function to find outliers such as labels with zero contours.       

    Args:
        y_pred_f(:string):  predictions file (.npy) with full path.
        
    Returns:
       None.
    
    Note:
        prection file should have the sigmoid outputs (not the rounded values)
    """
    y_pred_s = np.load(y_pred_f)
    samples, x, y, z = y_pred_s.shape
    print ("Number of Predictions : %d, image size : %d x %d "%(samples, x, y))
    y_pred = np.round(y_pred_s)
    y_pred_sum = y_pred.sum(axis=(1, 2), keepdims=True).reshape(samples)  
    pd0 = (np.where(y_pred_sum == 0))
    pd0 = list(pd0[0])
    print ("Sample Index of predictions with zero contours", pd0)
    ypr = []
    for idx in pd0:
        ypr.append(y_pred_s[idx,:,:,:].max())
    print ("max-sigmoid values with zero contours", ypr)
    print('-'*30)
    
    pd1 = (np.where(y_pred_sum <= 5))
    pd1 = list(pd1[0])
    print ("Sample Index with contour pixels <= 5", pd1)


def dump_and_sort(image_one_file, origpath, newpath):
    """ Get the minimum and the maximum contours from each slice """
    count = 0
    new_dir = os.path.dirname(newpath)

    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    with open(image_one_file, 'r') as inputs:
        jin = json.load(inputs)
        slicedict = dict()

        for i in sorted(jin):
            count += 1
            rootnode = i.split("/")
            tmp=rootnode[-1].split('_')
            sax = tmp[0] +'_'+ tmp[1]
            frame = tmp[-1]

            if sax in slicedict:
                slicedict[sax].update({frame: jin[i]})
            else:
                slicedict.update({sax: {frame: jin[i]}})

        min_max = get_min_max_slice(slicedict, origpath)
        
        with open(newpath, 'w') as output:
            output.write("{0}\n".format(json.dumps(min_max)))

def get_min_max_slice(slicedict, origpath):
    """
    Figure out the min and max of each slice

    Args:
      slicedict:  slice info
      origpath: original dcm numpy array file path

    Returns: Identified min max info in dict

    """
    identified = {}

    for i in slicedict: #i is sax
        zmin = 9999999
        zmax = 0
        zminframe = ''
        zmaxframe = ''
        zcounts = {}

        for j in slicedict[i]: #j is frame
            zcount = slicedict[i][j]['ones']

            if zcount in zcounts:
                if 'frame' in zcounts[zcount]:
                    zcounts[zcount]['frame'].append(j)
                else:
                    zcounts[zcount].update({'frame':[j]})
            else:
                zcounts.update({zcount: {'frame':[j]}})
            
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
            maxdw = dicomwrapper(origpath, maxpath)
            maxsl = maxdw.slice_location()
            maxst = maxdw.slice_thickness()
        except:
            print ('error max',origpath, maxpath)
            maxsl = None
            maxst = None

        try:
            mindw = dicomwrapper(origpath, minpath)
            minsl = mindw.slice_location()
            minst = mindw.slice_thickness()
        except:
            print ('error min',origpath, minpath)
            minsl = None
            minst = None

        identified[i] = {'zmin':zmin,
                         'zminframe': zminframe,
                         'minSL': minsl,
                         'minST': minst,
                         'zmax': zmax,
                         'zmaxframe': zmaxframe,
                         'maxSL': maxsl,
                         'maxST': maxst,
                        'zcounts': zcounts}

    return identified

    
if __name__ == "__main__":
    file_p = "/masvol/heartsmart/unet_model/data/baseline/sunnybrook_1_3_256_learning_history.json"
    plot_accuracy_and_loss(file_p)
