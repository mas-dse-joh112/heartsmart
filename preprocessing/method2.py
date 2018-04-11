#!/usr/bin/env python

import sys
import dicom
import os
import numpy as np
import glob
import cv2 
from scipy.misc import imrotate
import preproc
import nibabel as nib
from helpers_dicom import DicomWrapper as dicomwrapper


class Method2(object):
	def __init__(self, arg):
            self.arg = arg
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

            if self.source == 'acdc':
                self.update_filesource('source', self.source)
                self.get_acdc_files()
                return

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

        def orientation_flip180(self, img):
            return cv2.flip(img,-1)

        def contrast(self, img):
            im_max=np.percentile(img.flatten(),99)
            return np.array(np.clip(np.array(img,dtype=np.float)/im_max*256,0,255),dtype=np.uint8)

        def get_dsb_files(self):
            for f in self.inputfiles:
                print 'f', f
                nodes = f.split('/')

                patient = int(nodes[-1])

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
                            outfilename = "{0}_{1}.npy".format(rootnode, f)
                            outpath = "{0}/{1}/{2}".format(preproc.normoutputs[self.source]['dir'], self.method, dw.patient_id)

                            if not os.path.isdir(outpath):
                                os.mkdir(outpath)

                            np.save("{0}/{1}".format(outpath, outfilename), cropped)

                    self.update_filesource(patient, {'patientfiles':patientslices}, 1)
        
        def get_acdc_files(self):
            for f in self.inputfiles:
                nodes = f.split('/')

                inputs = glob.glob("{0}/{1}/{2}*".format(f,self.sourceinfo['string'],self.sourceinfo['pattern']))
                print 'inputs', inputs

                patient = None

                for i in inputs:
                    print 'i', i

                    if i.endswith('labels'):
                        for root, _, files in os.walk(i):
                            rootnode = root.split("/")
                            label = rootnode[-1]
                            tempdir = i.replace('/'+label,'')
                            patient = rootnode[-2]
                            print 'rootnode', rootnode, label, patient, files, tempdir

                            for f in files:
                                nim1dir = root.replace('/'+label,'')
                                nim1label = nib.load(root+'/'+f)
                                flippedlabel = self.orientation_flip180(nim1label.get_data())
                                cropped = self.crop_size(flippedlabel)
                                outfilename = "{0}.npy".format(f)
                                outpath = "{0}/{1}/{2}".format(preproc.normoutputs[self.source]['dir'], self.method, patient)

                                if not os.path.isdir(outpath):
                                    os.mkdir(outpath)

                                np.save("{0}/{1}".format(outpath, outfilename), cropped)

                                outfilenamenodes = outfilename.split('_')
                                slicepath = "{0}_{1}".format(outfilenamenodes[0], outfilenamenodes[1])
                                slicedir = "{0}/{1}".format(nim1dir, slicepath)

                                for root2, _, files2 in os.walk(slicedir):
                                    for f2 in files2:
                                        nim1 = nib.load(root2+'/'+f2)
                                        flipped = self.orientation_flip180(nim1.get_data())
                                        contrast_img2 = self.contrast(flipped)
                                        cropped2 = self.crop_size(contrast_img2)
                                        outfilename2 = "{0}.npy".format(f2)
                                        np.save("{0}/{1}".format(outpath, outfilename2), cropped2)

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
                            print root, f
                            img=dicom.read_file(root+'/'+f)
                            print img.BitsStored, img.BitsAllocated
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

if __name__ == "__main__":
    import config
    method = Method2(config)
    method.main_process()
