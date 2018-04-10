#!/usr/bin/env python

import sys
import dicom
import os
import numpy as np
import glob
import cv2 
from scipy.misc import imrotate
import preproc
from helpers_dicom import DicomWrapper as dicomwrapper


class Method2(object):
	def __init__(self, arg):
            self.arg = arg
            self.start = self.arg.start
            self.end = self.arg.end
            self.method = self.arg.method
            self.path = self.arg.path
            self.source = self.arg.source
            self.sourceinfo = None
            self.inputfiles = None
            self.filesource = dict()

        def get_config(self):
            if self.method != 2 and self.method != '2':
                print "method 2"
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
                print self.filesource
                return

            if self.source == 'sunnybrook':
                self.update_filesource('source', self.source)
                self.get_sunnybrook_files()
                return

        def update_filesource(self, key, value, append=0):
            print 'k', key, 'v', value, 'a', append

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
                print 'f', f
                nodes = f.split('/')

                patient = int(nodes[-1])

                if patient < int(self.start):
                    continue

                print 'patient range', patient, self.start, self.end

                if patient > int(self.end):
                    continue
                
                print self.sourceinfo
                inputs = glob.glob("{0}/{1}/{2}*".format(f,self.sourceinfo['string'],self.sourceinfo['pattern']))
                print 'inputs', inputs

                for i in inputs:
                    patientslices = dict()

                    for root, _, files in os.walk(i):
                        rootnode = root.split("/")[-1] # sax file
                        patientslices.update({root: []})

                        for f in files:
                            if not f.endswith('.dcm'):
                                continue

                            print root, f
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

                            img = self.getAlignImg(dw)
                            cropped = self.crop_size(img)
                            outfilename = "{0}_{1}_{2}.npy".format(dw.patient_id, rootnode, f)
                            np.save("{0}/{1}/{2}".format(preproc.normoutputs[self.source]['dir'], self.method, outfilename), cropped)

                    self.update_filesource(patient, {'patientfiles':patientslices}, 1)
        
        def get_sunnybrook_files(self):
            for i in self.inputfiles:
                if i.endswith('pdf'):
                    continue

                patientslices = dict()

                for root, _, files in os.walk(i):
                    rootnode = root.split("/")[-1] # sax file
                    patientslices.update({root: []})

                    for f in files:
                        if not f.endswith('.dcm'):
                            continue

                        dw = None

                        try:
                            dw = dicomwrapper(root+'/', f)
                        except:
                            continue

                        patientframe = dict()
                        patientframe.update({'filename': f})
                        patientframe.update({'InPlanePhaseEncodingDirection': dw.in_plane_encoding_direction})

                        patientslices.update({'image_position_patient': dw.image_position_patient})
                        patientslices.update({'image_orientation_patient': dw.image_orientation_patient})
                        patientslices.update({'PixelSpacing': dw.spacing})
                        #patientslices.update({'PatientAge': dw.PatientAge})

                        patientslices[root].append(patientframe)

                        img = self.getAlignImg(dw)
                        cropped = self.crop_size(img)
                        outfilename = "{0}.npy".format(f)
                        outpath =  "{0}/{1}/{2}/{3}".format(preproc.normoutputs[self.source]['dir'], self.method, self.path, rootnode)

                        if not os.path.isdir(outpath):
                            os.mkdir(outpath)

                        np.save("{0}/{1}".format(outpath, outfilename), cropped)

        def get_ACDC_files(self):
            for i in self.inputfiles:
                
                patientslices = dict()
                
                for root, _, files in os.walk(i):
                    rootnode = root.split("/")[-1] # sax file
                    patientslices.update({root: []})
                    
                    for f in files:
                        if not f.endswith('.nii'):
                            continue

                        dw = None
                        
                        try:
                            dw = dicomwrapper(root+'/', f)
                        except:
                            continue
                        patientframe = dict()
                        patientframe.update({'filename': f})
                        

	def getAlignImg(self, img, label = None):#!!!notice, only take uint8 type for the imrotate function!!!
	    f = lambda x:np.asarray([float(a) for a in x]);
	    o = f(img.image_orientation_patient);
	    o1 = o[:3];
	    o2 = o[3:];
	    oh = np.cross(o1,o2);
	    or1 = np.asarray([0.6,0.6,-0.2]);
	    o2new = np.cross(oh,or1);
	    theta = np.arccos(np.dot(o2,o2new)/np.sqrt(np.sum(o2**2)*np.sum(o2new**2)))*180/3.1416;
	    theta = theta * np.sign(np.dot(oh,np.cross(o2,o2new)));
	    im_max = np.percentile(img.pixel_array.flatten(),99);
	    res = imrotate(np.array(np.clip(np.array(img.pixel_array,dtype=np.float)/im_max*256,0,255),dtype=np.uint8),theta);
	    if label is None:
		return res;
	    else:
		lab = imrotate(label,theta);
		return res,lab
	    
	#Crop the image
	def crop_size(self, res):
	    shift  = np.array([0,0])
	    img_L=int(np.min(180)) #NEED TO UPDATE BASED ON COMMON IMAGE 
	    if res.shape[0]>res.shape[1]:
		s = (res.shape[0]-res.shape[1])//2;
		res = res[s:s+res.shape[1],:];
		shift[1] = s;
	    else:
		s = (res.shape[1]-res.shape[0])//2;
		res = res[:,s:s+res.shape[0]];
		shift[0] = s;

		#crop or stretch to the same size
	    if img_L>0 and (res.shape[0] != img_L):
		#print("crop or fill",filename);
		if res.shape[0]>img_L:#because h=w after crop
		    s = (res.shape[0]-img_L)//2;
		    res = res[s:s+img_L,s:s+img_L];
		    shift = shift + s;
		else:
		    s = (img_L-res.shape[0])//2;
		    res2 = np.zeros((img_L,img_L));
		    res2[s:s+res.shape[0],s:s+res.shape[0]] = res;
		    res = res2;
		    shift = shift - s;
	    return res

