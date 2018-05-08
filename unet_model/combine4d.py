#!/usr/bin/env python

import numpy as np

METHOD = 1
TYPE = 2
ACDC =  'acdc'
SB =  'sunnybrook'
IMAGE_SIZE = '176'
#IMAGE_SIZE = '256'
UNET_TRAIN_DIR = "/masvol/output/"
ACDC_SOURCE = "acdc/norm/{0}/{1}/unet_model/data/".format(METHOD, TYPE)
SB_SOURCE = "sunnybrook/norm/{0}/{1}/unet_model/data/".format(METHOD, TYPE)
UNET_DATA = "{0}/unet_training/".format(UNET_TRAIN_DIR)

def load_images_and_labels_no_preproc(data):
    print('-'*30)
    print('load np arrays of images and labels...')
    print('-'*30)
    imgfile = data["images"]
    labelfile = data["labels"]
    print ("Loading files : ", imgfile, labelfile)

    images = np.load(imgfile)
    labels = np.load(labelfile)

    print("shape, max, min, mean of original image set:", images.shape, images.max(), images.min(), images.mean())
    print("shape, max, min, mean of labels :", labels.shape, labels.max(), labels.min(), labels.mean())
    return images, labels

def combine_acdc_sunnybrook_data():
    global METHOD
    global TYPE
    global ACDC
    global SB
    global IMAGE_SIZE
    global UNET_TRAIN_DIR
    global ACDC_SOURCE
    global SB_SOURCE
    global UNET_DATA

    acdc_train_data = {}
    acdc_train_data["images"] = UNET_TRAIN_DIR + ACDC_SOURCE + "{0}_{1}_train_orig_images.npy".format(ACDC, IMAGE_SIZE)
    acdc_train_data["labels"] = UNET_TRAIN_DIR + ACDC_SOURCE + "{0}_{1}_train_orig_labels.npy".format(ACDC, IMAGE_SIZE)

    acdc_train_img, acdc_train_lbl = load_images_and_labels_no_preproc(acdc_train_data)

    acdc_test_data = {}
    acdc_test_data["images"] = UNET_TRAIN_DIR + ACDC_SOURCE + "{0}_{1}_test_orig_images.npy".format(ACDC, IMAGE_SIZE)
    acdc_test_data["labels"] = UNET_TRAIN_DIR + ACDC_SOURCE + "{0}_{1}_test_orig_labels.npy".format(ACDC, IMAGE_SIZE)

    acdc_test_img, acdc_test_lbl = load_images_and_labels_no_preproc(acdc_test_data)

    sb_train_data = {}
    sb_train_data["images"] = UNET_TRAIN_DIR + SB_SOURCE + "{0}_{1}_train_orig_images.npy".format(SB, IMAGE_SIZE)
    sb_train_data["labels"] = UNET_TRAIN_DIR + SB_SOURCE + "{0}_{1}_train_orig_labels.npy".format(SB, IMAGE_SIZE)
    sb_train_img, sb_train_lbl = load_images_and_labels_no_preproc(sb_train_data)

    sb_test_data = {}
    sb_test_data["images"] = UNET_TRAIN_DIR + SB_SOURCE + "{0}_{1}_test_orig_images.npy".format(SB, IMAGE_SIZE)
    sb_test_data["labels"] = UNET_TRAIN_DIR + SB_SOURCE + "{0}_{1}_test_orig_labels.npy".format(SB, IMAGE_SIZE)
    sb_test_img, sb_test_lbl = load_images_and_labels_no_preproc(sb_test_data)

    combined_train_img = np.concatenate((acdc_train_img, sb_train_img), axis=0)
    combined_train_lbl = np.concatenate((acdc_train_lbl, sb_train_lbl), axis=0)

    combined_test_img = np.concatenate((acdc_test_img, sb_test_img), axis=0)
    combined_test_lbl = np.concatenate((acdc_test_lbl, sb_test_lbl), axis=0)
    print (combined_train_img.shape, combined_train_lbl.shape,combined_test_img.shape,combined_test_lbl.shape)
    print ("Saving combined files.......")

    tr_img_file = UNET_DATA + "combined_{0}_{1}_{2}_train_images.npy".format(METHOD,TYPE,IMAGE_SIZE)
    tr_lbl_file = UNET_DATA + "combined_{0}_{1}_{2}_train_labels.npy".format(METHOD,TYPE,IMAGE_SIZE)
    tst_img_file = UNET_DATA + "combined_{0}_{1}_{2}_test_images.npy".format(METHOD,TYPE,IMAGE_SIZE)
    tst_lbl_file = UNET_DATA + "combined_{0}_{1}_{2}_test_labels.npy".format(METHOD,TYPE,IMAGE_SIZE)

    np.save(tr_img_file, combined_train_img)
    np.save(tr_lbl_file, combined_train_lbl)
    np.save(tst_img_file, combined_test_img)
    np.save(tst_lbl_file, combined_test_lbl)


if __name__ == "__main__":
    combine_acdc_sunnybrook_data()
