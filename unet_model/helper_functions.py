#########################################
#
# Methods to load 4d np array for images from ./data directory
# 4D tensor with shape: (samples, rows, cols, channels=1)
#
#########################################
import re, sys
import glob
import random
import json
import os 
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


def load_images_and_labels(data):
    print('-'*30)
    print('load np arrays of images and labels...')
    print('-'*30)
    imgfile = data["images"]
    labelfile = data["labels"]
    print ("Loading files : ", imgfile, labelfile)
    
    im = np.load(imgfile)
    lb = np.load(labelfile)
    images = im.astype('float32')
    labels = lb.astype('float32')
    
    ##Normalize the pixel values, (between 0..1)
    x_min = images.min(axis=(1, 2), keepdims=True)
    x_max = images.max(axis=(1, 2), keepdims=True)
    images2 = (images - x_min)/(x_max-x_min)

    print("shape, max, min, mean of original image set:", images.shape, images.max(), images.min(), images.mean())
    print("shape, max, min, mean after normalization  :", images2.shape, images2.max(), images2.min(), images2.mean())
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


def load_images(imgfile):
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

def plot_accuracy_and_loss(file_p):
    try:
        with open(file_p, 'r') as file:
            history = json.load(file)
    except (OSError, ValueError):  # file does not exist or is empty/invalid
        print ("File does not exist")
        
    # list all data in history
    print(history.keys())
    print('-'*30)
    print ("Values at first and last epoch")
    print('-'*30)
    for key in history:
        print (key, " : ", history[key][0], ",", history[key][-1])
    print('-'*30) 
    print('-'*30)
    # summarize history for accuracy
    if 'dice_coeff' in history.keys():
        plt.plot(history['dice_coeff'])
        plt.plot(history['val_dice_coeff'])
        plt.title('model accuracy(dice_coeff)')
    elif 'val_acc' in history.keys():
        plt.plot(history['acc'])
        plt.plot(history['val_acc'])
        plt.title('Model accuracy')
    elif 'categorical_accuracy' in history.keys():
        plt.plot(history['categorical_accuracy'])
        plt.plot(history['val_categorical_accuracy'])
        plt.title('categorical_accuracy')
    elif 'binary_accuracy' in history.keys():
        plt.plot(history['binary_accuracy'])
        plt.plot(history['val_binary_accuracy'])
        plt.title('Minary_accuracy')
    else : 
        print ("new loss function, not in the list")
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='lower right')
    plt.grid()
    plt.show()
    # summarize history for loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.grid()
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()

def show_perf_values(file_p):
    perf_list = ['logloss','log2loss','true_positive', 'false_positive', 'true_negative','false_negative', \
                 'precision','recall','accuracy', 'f1_score' ]
    try:
        with open(file_p, 'r') as file:
            perf = json.load(file)
    except (OSError, ValueError):  # file does not exist or is empty/invalid
        print ("File does not exist")

    print('-'*30)
    for key in perf_list:
        if key in perf.keys():
            print (key, " : ", perf[key])
    print('-'*30) 
    print('-'*30)
    # list all data in history
    
    
    
def get_performance_statistics(y_true, y_pred):
#     y_true = np.load(y_true_f)
#     y_pred = np.load(y_pred_f)
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

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
    perf["true_positive"] = cm[1][1]
    perf["false_positive"] = cm[0][1]
    perf["true_negative"] = cm[0][0]
    perf["false_negative"] = cm[1][0]
    #cm.print_stats()
    return perf

def compute_performance_statistics (y_true_f, y_pred_f):
    #read the numpy files
    y_true = np.load(y_true_f)
    y_pred = np.load(y_pred_f)
    #print (y_true.shape, y_pred.shape)
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    weight = 0.8
    sample_weights = np.copy(y_true)
    sample_weights[sample_weights == 1] = 1.
    sample_weights[sample_weights == 0] = .2
    
    
    epsilon = 1e-7
    y_pred[y_pred<=0.] = epsilon
    y_pred[y_pred>=1.] = 1. -epsilon
    
    #print (y_true.shape, y_pred.shape)

    score = log_loss (y_true, y_pred)
    score2 = log_loss (y_true, y_pred, sample_weight = sample_weights)
    acc = math.exp(-score)
    acc2 = math.exp(-score2)
    y_pred = np.round(y_pred)
    print ("log_loss : ", score,  "  Accuracy: ", acc)
    print ("weighted log_loss : ", score2,  "  Accuracy: ", acc2)
 
    cm = confusion_matrix(y_true, y_pred)
    print (cm)
    #cm.print_stats()
    true_p = cm[1][1]
    false_p = cm[0][1]
    true_n = cm[0][0]
    false_n = cm[1][0]
    print ("true_p = %d, false_p = %d, true_neg = %d, false_neg = %d"%(true_p, false_p, true_n, false_n))
    print("f1 score :", f1_score(y_true, y_pred, average="binary"))
    print("precision :", precision_score(y_true, y_pred, average="binary"))
    print("recall :", recall_score(y_true, y_pred, average="binary"))
    print (" ")
    plt.matshow(cm)
    plt.title('Confusion matrix of the classifier')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.colorbar()
    
    plt.show()
    
def display_images_labels (image_file, label_file,  num_images = 4, random_images = False):
    ts = np.load(image_file)
    tl = np.load(label_file)
    samples, x, y, z = tl.shape

    display_list = []

    if random_images == True:
        display_list = random.sample(range(0, samples), num_images)
    else :
        display_list = [i for i in range (num_images)]

    for i in display_list:
        f, axs = plt.subplots(1,3,figsize=(15,15))
        plt.subplot(131),plt.imshow(ts[i].reshape(x, y))
        plt.title('Image '+str(i)), plt.xticks([]), plt.yticks([])
        plt.subplot(132),plt.imshow(tl[i].reshape(x, y))
        plt.title('Label'), plt.xticks([]), plt.yticks([])
        plt.subplot(133),plt.imshow(ts[i].reshape(x, y)), plt.imshow(tl[i].reshape(x, y), 'binary', interpolation='none', alpha=0.3)
        plt.title('Overlay'), plt.xticks([]), plt.yticks([])
        plt.show()

def display_images_labels_predictions (image_file, label_file, pred_file, num_images = 4, random_images = False):
    ts = np.load(image_file)
    tl = np.load(label_file)
    pred = np.load(pred_file)
    samples, x, y, z = pred.shape
    print ("samples, max, min ", samples, pred.max(), pred.min())
    pred2 = np.round(pred)

    ##Print few images wih actual labels and predictions
    display_list = []

    if random_images == True:
        display_list = random.sample(range(0, samples), num_images)
    else :
        display_list = [i for i in range (num_images)]

    for i in display_list:
        f, axs = plt.subplots(1,4,figsize=(15,15))
        plt.subplot(141),plt.imshow(ts[i].reshape(x, y))
        plt.title('Image '+str(i)), plt.xticks([]), plt.yticks([])
        plt.subplot(142),plt.imshow(tl[i].reshape(x, y))
        plt.title('Label'), plt.xticks([]), plt.yticks([])
        plt.subplot(143),plt.imshow(pred2[i].reshape(x, y))
        plt.title('Prediction'), plt.xticks([]), plt.yticks([])
        plt.subplot(144),plt.imshow(tl[i].reshape(x, y)), plt.imshow(pred2[i].reshape(x, y), 'binary', interpolation='none', alpha=0.5)
        plt.title('Overlay'), plt.xticks([]), plt.yticks([])
        plt.show()


def display_images_predictions (image_file, pred_file,  num_images = 4, random_images = False):
    ts = np.load(image_file)
    pred = np.load(pred_file)
    samples, x, y, z = pred.shape
    print ("samples, max, min ", samples, pred.max(), pred.min())
    pred2 = np.round(pred)

    display_list = []

    if random_images == True:
        display_list = random.sample(range(0, samples), num_images)
    else :
        display_list = [i for i in range (num_images)]

    for i in display_list:
        f, axs = plt.subplots(1,3,figsize=(15,15))
        plt.subplot(131),plt.imshow(ts[i].reshape(x, y))
        plt.title('Image '+str(i)), plt.xticks([]), plt.yticks([])
        plt.subplot(132),plt.imshow(pred2[i].reshape(x, y))
        plt.title('Prediction'), plt.xticks([]), plt.yticks([])
        plt.subplot(133),plt.imshow(ts[i].reshape(x, y)), plt.imshow(pred2[i].reshape(x, y), 'binary', interpolation='none', alpha=0.3)
        plt.title('Overlay'), plt.xticks([]), plt.yticks([])
        plt.show()
    
    
if __name__ == "__main__":
    file_p = "/masvol/heartsmart/unet_model/data/baseline/sunnybrook_1_3_256_learning_history.json"
    plot_accuracy_and_loss(file_p)
