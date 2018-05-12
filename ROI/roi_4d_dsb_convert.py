#!/usr/bin/env python

import pydicom as dm
import numpy as np
from sklearn.cluster import KMeans
import scipy.misc 
import glob
from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
from skimage.util import crop, pad   
import os
from method1 import Method1
import pickle
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import config
# dummy settings
config.method = 0
config.source = ''
config.path = ''
config.type = 0
m1 = Method1(config)

import cv2 as c

def pad_image(img, img_size):
    pad_x=0
    pad_y=0
    x,y = img.shape

    if (x<img_size):
        pad_x = img_size - x

    if (y<img_size):
        pad_y = img_size - y

    process_img = np.pad(img, pad_width=((pad_x//2, ((pad_x//2) + (pad_x % 2))), (pad_y//2, ((pad_y//2) + (pad_y % 2)))), mode = 'constant', constant_values = 0)
    return process_img

def fix_acdc(im):    
    max_val=im.max()

    if max_val!=0: 
        im[im<max_val]=0
        im[im==max_val]=1
    return im

def dislplay_list(L,binary,image_size):
    imgs = []
    
    for i in L:
        t= check_image(np.load(i),image_size)*binary
        imgs.append(t)

    return imgs

def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def check_image(img,crop_size):
    x,y = img.shape

    if x < crop_size or y < crop_size:
        img = pad_image(img, crop_size)
    elif x > crop_size or y > crop_size:
        return crop_center(img,crop_size,crop_size)
    else:
        print('Error')

    return crop_center(img,crop_size,crop_size)  

def crop_image(image, crop_mask):
    ''' Crop the non_zeros pixels of an image  to a new image 
      Need to rerfernce it    
         
    '''
    pxlst = np.where(crop_mask.ravel())[0]
    dims = crop_mask.shape
    imgwidthy = dims[1]   #dimension in y, but in plot being x
    imgwidthx = dims[0]   #dimension in x, but in plot being y
    #x and y are flipped???
    #matrix notation!!!
    pixely = pxlst%imgwidthy
    pixelx = pxlst//imgwidthy

    minpixelx = np.min(pixelx)
    minpixely = np.min(pixely)
    maxpixelx = np.max(pixelx)
    maxpixely = np.max(pixely) 
    crops = crop_mask*image
    img_crop = crop(crops, ((minpixelx, imgwidthx -  maxpixelx -1 ), (minpixely, imgwidthy -  maxpixely -1 )))
    return img_crop 

def cluster_image(img,rand=0):
    image_reshape=img.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=rand,max_iter=100).fit(image_reshape)
    clustered_image=kmeans.labels_

    if kmeans.labels_[0]>0:
        clustered_image[clustered_image==1]=-1
        clustered_image[clustered_image==0]=1
        clustered_image[clustered_image==-1]=0
    else:
        clustered_image

    clustered_image=clustered_image.reshape(img.shape[0],img.shape[1])
    return clustered_image

def hough_circles(img,start,end,step,peaks):
    hough_radii = np.arange(start,end, step)
    hough_res = hough_circle(img, hough_radii)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                               total_num_peaks=peaks)
    image = color.gray2rgb(img)
    radii=radii*1.2
    radii=radii.astype(int)

    for center_y, center_x, radius in zip(cy, cx, radii):
        circy, circx = circle_perimeter(center_y, center_x, radius)
        try:
            image[circy, circx] = (220, 20, 20)
        except:
            continue 

    return np.abs(image[:,:,1]-image[:,:,0])

def create_filter_binary(image,edges):
    height=image.shape[0]
    width=image.shape[1]
    binary = np.zeros([height,width])

    for row in range(0,height):
        for col in range(0, width):
            if edges.sum(axis=0)[col]*edges.sum(axis=1)[row]>0 :
                binary[row][col]=1
                
    return binary

def fourier_transform_pod(L):
    t =np.load(L[0])
    pylab.imshow(t)
    pylab.show()
    finalArray = np.zeros((256, 256, len(L)))

    for index, image in enumerate(L):
        t= np.load(L[index])
        x=check_image(t,256)
        toFill = x
        finalArray[:, :, index] = toFill

    ff=np.fft.fft(finalArray)
    return ff

def harmonic_number_pod(L,n):
    B=fourier_transform_pod(L)
    harmonic_n = np.abs(B[:,:,n])
    return  harmonic_n

def high_variance_pod(L):
    x=0
    y=0

    for i in L:
        x=check_image(np.load(i),256)+x

    x=x/(len(L))

    for j in L:
        t = check_image(np.load(j),256)
        y=(t-x)**2+y

    v=y/(len(L))
    return v

def convert_images_to_nparray_and_save (imgs, save_file, image_size):
    rows = image_size
    cols = image_size
    i = 0
    print('-'*30)
    print("Converting data to np array, Input size : ",len(imgs))
    print('-'*30)

    imgdatas = np.ndarray((len(imgs),rows,cols,1), dtype=np.int)

    for idx in range(len(imgs)):
        img = imgs[idx]
        img = img_to_array(img)
        try:
            imgdatas[i] = img
            i += 1
        except Exception as e:
            print (e)
            continue

    np.save(save_file, imgdatas)

    print ("Shape of image array : ", imgdatas.shape)
    print ("Max, min, mean values", imgdatas.max(), imgdatas.min(), imgdatas.mean())
    print('Saved data as: ', save_file)

if __name__ == "__main__":
    datasource='dsb'
    path='/masvol/output/{0}/norm/opt/output'.format(datasource)
    method='1'
    typenumber='3'
    #source_type = "test"
    source_type = "train"
    image_size = 176
    #image_size = 256
    outpath = "/masvol/output/{0}/norm/{1}/{2}/unet_model_{3}_roi/data".format(datasource, method, typenumber, source_type)
    fullpath='/masvol/output/{0}/norm/{1}/{2}/{3}'.format(datasource, method, typenumber, source_type)

    with open('{0}/{1}/norm/{2}/{3}/{4}/filters/all_processed_train_DSB_{5}.pickle'.format(path, datasource, method, typenumber,
                                                                                           source_type, image_size), 'rb') as handle:
        inputs = pickle.load(handle)
    
        for i in inputs:
            ppath = "{0}/{1}/sax_*".format(fullpath, i)
            files = glob.glob(ppath)
            imgs = dislplay_list(files,inputs[i], image_size)
            outfile = "{0}/{1}_{2}_{3}_{4}.npy".format(outpath, datasource, i, image_size, source_type)
            convert_images_to_nparray_and_save(imgs, outfile, image_size)
        
            filepath = "{0}/{1}_{2}_image_path.txt".format(outpath, datasource, i)
            # creating files to store source file names
            #with open(filepath, 'w') as outfile:
            #    outfile.write("{0}\n".format(",".join(files)))
