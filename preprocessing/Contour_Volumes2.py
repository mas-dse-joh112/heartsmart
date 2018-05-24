import dicom
import sys
import os
import numpy as np
import glob
import cv2 
import re
#import preproc
import nibabel as nib
from helpers_dicom import DicomWrapper as dicomwrapper

BASE_DIR = "/opt/output/"
SOURCE = "sunnybrook"
SCD_LBL_PATH = BASE_DIR + SOURCE + "/norm/1/3/labels/"
SCDimgpathT = "/opt/data/" + SOURCE + "/challenge_training/" 
SCDimgpathV = "/opt/data/" + SOURCE + "/challenge_validation/"

print(SCDimgpathV)

    #print (rootnode, patient_path)
    #results.append([filename, patientVolume(patient_path)])
#print (results)

def getDICOM(path, patient, img):
    imgpath = path + 'IM-' +patient +'-'+ img + '.dcm'
    return imgpath

def getfrac(img):
    frac = (sum(i == 1 for i in img.flatten()))/float(len(img.flatten()))
    return frac

def slicelocation(img):
    sl= np.dot(img.ImagePositionPatient,
                   np.cross(img.ImageOrientationPatient[:3],img.ImageOrientationPatient[3:]))
    return sl

def getVolumes(slicedict):
    ESF = []
    ESVlist = []
    EDF = []
    EDVlist = []

    for key in slicedict:
        frames = list(slicedict[key].keys())
        #print (frames)
        if len(frames) > 2:
            print ('A slice has more than two frames')
        
        if len(frames) < 2:
            print ('A slice has one frame')
            
        tmp = 0
        for i in range(len(frames)-1):
            #print (slicedict[key][frames[i]]['fraction'], slicedict[key][frames[i+1]]['fraction'] )
            if slicedict[key][frames[i]]['fraction'] < slicedict[key][frames[i+1]]['fraction']:
                ESF.append(frames[i])
                ESVlist.append(slicedict[key][frames[i]]['Volume'])
                EDF.append(frames[i+1])
                EDVlist.append(slicedict[key][frames[i+1]]['Volume'])
            else:
                ESF.append(frames[i+1])
                ESVlist.append(slicedict[key][frames[i+1]]['Volume'])
                EDF.append(frames [i])
                EDVlist.append(slicedict[key][frames[i]]['Volume'])
    ESV = sum(ESVlist)
    EDV = sum(EDVlist)
    return ESF, ESV, EDF, EDV

def patientVolume(patient_path):
    for root,_, files in os.walk(patient_path):
        
            norm = None
            slicedict={}
            print (patient_path)
        
            for i in files:
                #print (i)
                filenode = i.strip('.dcm.label.npy')
                filenode = re.split('-',filenode)
                #print (filenode)
                patient = filenode[1]
                #slicenum = 
                #frame = int(filenode[2])
                lblpath = root +'/'+ i
            
                imgpath = getDICOM(origimgpath, patient, filenode[2])

                img = dicom.read_file(imgpath)
            
                if norm is not None:
                    ps = img.PixelSpacing[0]
                else:
                    ps = 1
            
                thickness = img.SliceThickness
                gap = img.SpacingBetweenSlices
                sliceLocation = slicelocation(img)
            
                lbl = np.load(lblpath)
            
                frac = getfrac(lbl)
                area = frac * ps * len(lbl)
                volume = thickness * area

                if sliceLocation in slicedict:
                    slicedict[sliceLocation].update({filenode[2]:
                                                {'fraction': frac,
                                                 'Area': area, 
                                                 'Volume': volume}})
                else:
                    slicedict[sliceLocation] = {filenode[2]:
                                        {'fraction': frac,
                                        'Area': area, 
                                        'Volume': volume}}
            
            ESF, ESV, EDF, EDV = getVolumes(slicedict)
            return (ESV, EDV)
        
results = []
for root, _, files in os.walk(SCD_LBL_PATH):
    rootnode = root.split("/")
    #label = rootnode[-2]
    filename = rootnode[-1]
    #patient_path = SCDlblpath + filename + '/'

    if os.path.isdir(SCDimgpathT + filename + '/') is True:
        print ('True')
        origimgpath = SCDimgpathT + filename + '/'
        patient_path = SCD_LBL_PATH + filename + '/'
        print (patient_path)
    
    elif os.path.isdir(SCDimgpathV + filename + '/') is True:
        print ('True')
        origimgpath = SCDimgpathV + filename + '/'
        patient_path = SCD_LBL_PATH + filename + '/'
        print (patient_path)
    
    #else:
     #   print (SCDimgpathT + filename + '/')
      #  print('Challenge_Online Directory')  
    results.append([filename, patientVolume(patient_path)])
