#!/usr/bin/env python


import preproc
import re, sys
import fnmatch, shutil, subprocess
import glob
import os 
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from method1 import Method1
import config

# dummy parms
config.method = 0
config.source = ''
config.path = ''
config.type = 0
m1 = Method1(config)


#Fix the random seeds for numpy (this is for Keras) and for tensorflow backend to reduce the run-to-run variance
from numpy.random import seed
seed(100)
from tensorflow import set_random_seed
set_random_seed(200)

print("\nSuccessfully imported packages!!!\n")

#Settings
IMAGE_SIZE_BIG = 256
IMAGE_SIZE_SMALL = 176
BASE_DIR = "/masvol/output/"
SOURCE = None
RE_PATTERN = None
TRAIN_IMG_DIR = None
TRAIN_LBL_DIR = None
TEST_IMG_DIR = None
PRED_RESULT_DIR = None
UNET_TRAIN_DIR = None

TRAIN_TEST_SPLIT_RATIO = 0.1  # train/test split ratio

##################################
#
# Methods to extract contour files and corresponding image files 
#
###################################


def shrink_case(case):
    toks = case.split("-")
    
    def shrink_if_number(x):
        try:
            cvt = int(x)
            return str(cvt)
        except ValueError:
            return x

    return "-".join([shrink_if_number(t) for t in toks])

class Contour(object):
    def __init__(self, ctr_path):
        self.ctr_path = ctr_path
        #print (ctr_path)
        
        match = re.search(r"/{0}".format(RE_PATTERN), ctr_path)
        self.case = shrink_case(match.group(1))
        self.record = int(match.group(2))
        self.img_no = int(match.group(3))
    
    def __str__(self):
        return "<Contour for case %s, record %d image %d>" % (self.case, self.record, self.img_no)
    
    __repr__ = __str__

def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def load_contour(contour, img_path, crop_size):
    filename = None

    if SOURCE == "sunnybrook":
        filename = preproc.filenames[SOURCE] % (preproc.sax_series_all[contour.case], contour.img_no)
    elif SOURCE == "acdc":
        filename = preproc.filenames[SOURCE] % (contour.case, contour.record, contour.img_no)

    full_path = os.path.join(img_path, contour.case, filename)
    print (full_path)
    img = np.load(full_path)
    print (img.shape)
    label = np.load(contour.ctr_path)
    img = m1.get_square_crop(img,crop_size,crop_size)
    label = m1.get_square_crop(label,crop_size,crop_size)
    #height, width = img.shape
    #height_l, width_l = label.shape
    
    #if height != crop_size or width != crop_size:
    #    img = crop_center(img,crop_size,crop_size)
    #    label = crop_center(label,crop_size,crop_size)
        
    return img, label, full_path
   
def get_all_contours(contour_path):
    contours = None

    if SOURCE == "acdc":
        contours = [os.path.join(dirpath,f) 
            for files in glob.glob(TRAIN_LBL_DIR+"/*") 
            for dirpath, dirname, infiles in os.walk(files) 
            for f in infiles if f.endswith(preproc.labels[SOURCE])]
    else:
        print ('contour_path', contour_path)
        contours = [os.path.join(dirpath, f)
            for dirpath, dirnames, files in os.walk(contour_path)
            for f in fnmatch.filter(files, preproc.labels[SOURCE])]
    
    print("Number of examples: {:d}".format(len(contours)))
    print("Shuffle data")
    
    np.random.shuffle(contours)
    print (contours[0], contours[-1])
    print("Number of examples after cleanup: {:d}".format(len(contours)))
    
    extracted = list(map(Contour, contours))
    print ("Contour 0 :", extracted[0].case, extracted[0].record, extracted[0].img_no)
    print ("Contour -1 :", extracted[-1].case, extracted[-1].record, extracted[-1].img_no) 
    return extracted

def get_contours_and_images(contours, img_path, crop_size):
    counter_img = 0
    counter_label = 0
    batchsz = len(contours)
    print("Processing {:d} images and labels...".format(len(contours)))
    
    for i in range(int(np.ceil(len(contours) / float(batchsz)))):
        batch = contours[(batchsz*i):(batchsz*(i+1))]

        if len(batch) == 0:
            break
            
        imgs, labels = [], []
        #imgs2, labels2 = [], []
        
        for idx,ctr in enumerate(batch):
            filename = None

            if SOURCE == "sunnybrook":
                filename = preproc.filenames[SOURCE] % (preproc.sax_series_all[ctr.case], ctr.img_no)
            elif SOURCE == "acdc":
                filename = preproc.filenames[SOURCE] % (ctr.case,ctr.record,ctr.img_no)

            #full_path = os.path.join(img_path, ctr.case, filename)
            #img = np.load(full_path)
            #x,y = img.shape
                    
            #if x < crop_size or y < crop_size:
            #    continue
                
            img, label, fullpath = load_contour(ctr, img_path, crop_size)
            imgs.append(img)
            labels.append(label)

            if idx % 100 == 0:
                print (fullpath)

    return imgs, labels

def extract_training_data(crop_size=256): 
    SPLIT_RATIO = TRAIN_TEST_SPLIT_RATIO  # train/test split ratio
    print("Mapping ground truth contours to images...")
    
    ctrs = get_all_contours(TRAIN_LBL_DIR)
    print("Done mapping ground truth contours to images")
    
    test_ctrs = ctrs[0:int(SPLIT_RATIO*len(ctrs))]
    train_ctrs = ctrs[int(SPLIT_RATIO*len(ctrs)):]
    print("Split train_set:%d, test_set:%d"%(len(train_ctrs), len(test_ctrs)))
    print ("Extracting Training Images and Labels")
    
    train_imgs, train_labels = get_contours_and_images(train_ctrs, TRAIN_IMG_DIR, crop_size)
    print ("Extracting Test Images and Labels")
    
    test_imgs, test_labels = get_contours_and_images(test_ctrs, TRAIN_IMG_DIR, crop_size)
    print("Extracted Images train_set:%d, test_set:%d"%(len(train_imgs), len(test_imgs)))
    
    return train_imgs, train_labels, test_imgs, test_labels 

#########################################
#
# Method to create  4d np array for images and labels and store the array in a directory
# 4D tensor with shape: (samples, rows, cols, channels=1)
#
#########################################

def create_training_data(imgs, lbls, save_file_path, file_prefix, image_size):
    rows = image_size
    cols = image_size
    i = 0
    print('-'*30)
    print("Creating training data..input size : ",len(imgs))
    print('-'*30)
    print("Converting data to np array")
    
    imgdatas = np.ndarray((len(imgs),rows,cols,1), dtype=np.int)
    imglabels = np.ndarray((len(imgs),rows,cols,1), dtype=np.uint8)
    
    for idx in range(len(imgs)):
        img = imgs[idx]
        label = lbls[idx]
        img = img_to_array(img)
        label = img_to_array(label)
        imgdatas[i] = img
        imglabels[i] = label
        i += 1
        
    imgfile = save_file_path + file_prefix +"_images.npy"
    print (imgfile)
    lblfile = save_file_path + file_prefix +"_labels.npy"
    np.save(imgfile, imgdatas)
    np.save(lblfile, imglabels)

    print ("Shape of data & label np arrays : ", imgdatas.shape, imglabels.shape)
    print (imgdatas.max(), imgdatas.min(), imglabels.max(), imglabels.min())
    print('Saved data as: ', imgfile, lblfile )


if __name__ == "__main__":
    SOURCE = 'sunnybrook'
    #SOURCE = 'acdc'
    METHOD = '1'
    TYPE = '1'
    #TYPE = '2'
    RE_PATTERN = preproc.re_patterns[SOURCE]
    TRAIN_IMG_DIR = BASE_DIR + SOURCE + "/norm/{0}/{1}/images/".format(METHOD,TYPE)
    TRAIN_LBL_DIR = BASE_DIR + SOURCE + "/norm/{0}/{1}/labels/".format(METHOD,TYPE)
    TEST_IMG_DIR = BASE_DIR + SOURCE + "/norm/{0}/{1}/images/".format(METHOD,TYPE)
    PRED_RESULT_DIR = BASE_DIR + SOURCE + "/norm/{0}/{1}/images/".format(METHOD,TYPE)
    UNET_TRAIN_DIR = BASE_DIR + SOURCE + "/norm/{0}/{1}/unet_model/data/".format(METHOD,TYPE)

    ##Get images and labels with crop from center to get 256x256 images
    train_imgs, train_labels, test_imgs, test_labels = extract_training_data(IMAGE_SIZE_BIG)

    ##Get images and labels with crop from center to get 180x180 images
    train_imgs2, train_labels2, test_imgs2, test_labels2 = extract_training_data(IMAGE_SIZE_SMALL)

    ### Create 256x256 size train/test data in 4d tensor shape and save them
    save_location = UNET_TRAIN_DIR
    tr_file_prefix = SOURCE + "_256_train_orig"
    tst_file_prefix = SOURCE + "_256_test_orig"
    create_training_data(train_imgs, train_labels, save_location, tr_file_prefix, IMAGE_SIZE_BIG)
    create_training_data(test_imgs, test_labels, save_location, tst_file_prefix, IMAGE_SIZE_BIG)

    ### Create 180x180 size train/test data in 4d tensor shape and save them
    tr_file_prefix = SOURCE + "_176_train_orig"
    tst_file_prefix = SOURCE + "_176_test_orig"

    create_training_data(train_imgs2, train_labels2, save_location, tr_file_prefix, IMAGE_SIZE_SMALL)
    create_training_data(test_imgs2, test_labels2, save_location, tst_file_prefix, IMAGE_SIZE_SMALL)
