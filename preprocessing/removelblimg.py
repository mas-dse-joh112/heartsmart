#!/usr/bin/env python

import os

lblfound = 0
imgfound = 0

with open ('t', 'r') as inputs:
    for i in  inputs:
        lbl = i.strip().replace('rame0','rame')
        img = lbl.replace('labels','images')
        img = img.replace('_label_fix.nii','.nii')
     
        print (lbl, img)
        if os.path.isfile(lbl):
            lblfound += 1
            os.remove(lbl)
        if os.path.isfile(img):
            imgfound += 1
            os.remove(img)

print (lblfound, imgfound)
