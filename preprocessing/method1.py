#!/usr/bin/env python

import sys
import dicom
import os
import numpy as np
import glob
import cv2 
import preproc
from helpers_dicom import DicomWrapper as dicomwrapper


class Method1(object):
	def __init__(self, arg):
            self.arg = arg
            print self.arg
            self.start = self.arg.start
            self.end = self.arg.end
            self.method = self.arg.method
            self.path = self.arg.path
            self.source = self.arg.source
            self.sourceinfo = None
            self.inputfiles = None
            self.filesource = dict()

        def get_config(self):
            if self.method != 1 and self.method != '1':
                print "method 1"
                sys.exit()

            if self.path != 'train' and self.path != 'validate':
                print "train or validate path"
                sys.exit()

            if self.source not in preproc.sources.keys():
                sys.exit()

            if int(self.end) < int(self.start):
                self.end = self.start

        """ main function """
        def main_process(self):
            self.get_config()
            self.get_init_path()

            if self.source == 'dsb':
                self.update_filesource('source', self.source)
                self.get_dsb_files()
                #print self.filesource

        def update_filesource(self, key, value, append=0):
            #print 'k', key, 'v', value, 'a', append

            if key not in self.filesource:
                if append:
                    self.filesource.update({key: [value]})
                    return

                self.filesource.update({key: value})
                return

            elif append:
                self.filesource[key].append(value)
                return
              
            self.filesource.update({key: value})

        def get_init_path(self):
            self.sourceinfo = preproc.sources[self.source]

            if self.path not in self.sourceinfo['paths']:
                print "valid paths {0}".format(self.sourceinfo['paths'])
                sys.exit() 

            path = "{0}/{1}/*".format(self.sourceinfo['dir'], self.path)
            self.update_filesource('path', path)
            self.inputfiles = glob.glob(path)

        def get_dsb_files(self):
            for f in self.inputfiles:
                #print 'f', f
                nodes = f.split('/')

                patient = int(nodes[-1])

                if patient < int(self.start):
                    continue

                #print 'patient range', patient, self.start, self.end

                if patient > int(self.end):
                    continue
                
                inputs = glob.glob("{0}/{1}/{2}*".format(f,self.sourceinfo['string'],self.sourceinfo['pattern']))
                #print 'inputs', inputs

                for i in inputs:
                    patientslices = dict()

                    for root, _, files in os.walk(i):

                        rootnode = root.split("/")[-1] # sax file
                        patientslices.update({root: []})

                        for f in files:
                            if not f.endswith('.dcm'):
                                continue

                            #print root, f
                            dw = dicomwrapper(root+'/', f)

                            if int(dw.patient_id) != patient:
                                print 'Error'
                                sys.exit()

                            patientframe = dict()
                            patientframe.update({'filename': f})
                            patientframe.update({'InPlanePhaseEncodingDirection': dw.in_plane_encoding_direction})

                            patientslices.update({'image_position_patient': dw.image_position_patient})
                            patientslices.update({'image_orientation_patient': dw.image_orientation_patient})
                            patientslices.update({'PixelSpacing': dw.spacing})
                            patientslices.update({'PatientAge': dw.PatientAge})

                            patientslices[root].append(patientframe)

                            img = self.InPlanePhaseEncoding(dw)
                            rescaled = self.reScale(img, dw.spacing[0])
                            outfilename = "{0}_{1}.npy".format(rootnode, f)
                            outpath =  "{0}/{1}/{2}".format(preproc.normoutputs[self.source]['dir'], self.method, dw.patient_id)

                            if not os.path.isdir(outpath):
                                os.mkdir(outpath)

                            np.save("{0}/{1}".format(outpath, outfilename), rescaled)

                    self.update_filesource(patient, {'patientfiles':patientslices}, 1)

		
	#Function that uses the InPlanephaseEncoding to determine if COL or ROW based and then transposes and flips the image. 
	def InPlanePhaseEncoding (self, img):
	    if img.in_plane_encoding_direction == 'COL':
	        new_img = cv2.transpose(img.pixel_array)
	        #py.imshow(img_new)
	        new_img = cv2.flip(new_img, 0)
	        return new_img
	    else:
	        #print 'Row Oriented'
	        return img.pixel_array

	#Function of Rescaling the pixels
	def reScale(self, img, scale):
	    return cv2.resize(img, (0, 0), fx=scale, fy=scale)

	#Function to crop the image into a square
	def get_square_crop(self, img, base_size=256, crop_size=256):
	    res = img
	    height, width = res.shape

	    if height < base_size:
	        diff = base_size - height
	        extend_top = diff / 2
	        extend_bottom = diff - extend_top
	        res = cv2.copyMakeBorder(res, extend_top, extend_bottom, 0, 0, 
	                                 borderType=cv2.BORDER_CONSTANT, value=0)
	        height = base_size

	    if width < base_size:
	        diff = base_size - width
	        extend_top = diff / 2
	        extend_bottom = diff - extend_top
	        res = cv2.copyMakeBorder(res, 0, 0, extend_top, extend_bottom, 
	                                 borderType=cv2.BORDER_CONSTANT, value=0)
	        width = base_size

	    crop_y_start = (height - crop_size) / 2
	    crop_x_start = (width - crop_size) / 2
	    res = res[crop_y_start:(crop_y_start + crop_size), crop_x_start:(crop_x_start + crop_size)]
	    return res

	#Contrast Normalizaiton
	def CLAHEContrastNorm(self, img, tile_size=(1,1)):
	    clahe = cv2.createCLAHE(tileGridSize=tile_size)
	    return clahe.apply(sq_img_0)


if __name__ == "__main__":
    arg = {'start':2,
           'end':2,
           'method': 1,
           'path': 'train'
          }

    method = Method1(arg)
    method.get_config()
    method.get_files()
