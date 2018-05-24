#!/usr/bin/env python

import preproc
import argparse
import os
import glob
import numpy as np

def fix_acdc_no_delete(im):
    max_val=im.max()

    if max_val != 0:
        im[im<max_val]=0
        im[im==max_val]=1

    return im, 0

def fix_acdc(im):
    max_val=im.max()

    if max_val != 0:
        if max_val < 3:
            return im, 1

        im[im<max_val]=0
        im[im==max_val]=1

    if np.count_nonzero(im) > 0:
        return im, 0

    return im, 1

def do_delete(method, type):
    imgpath = "/masvol/output/acdc/norm/{0}/{1}/*".format(method, type)
    lblpath = "/masvol/output/acdc/norm/{0}/{1}L".format(method, type)

    lblcount = 0
    lblfound = 0
    removeimg = 0

    for i in glob.glob(imgpath):
        print (i)

        for j in glob.glob("{0}/*".format(i)):
            if 'label' in j:
                print ('removing old lbl', j)
                os.remove(j)
                lblcount += 1
                continue

            nodes = j.split('/')
            lblfile = "{0}/{1}/{2}".format(lblpath,nodes[-2],nodes[-1])
            lblfile = lblfile.replace(".nii","_label_fix.nii")
            #print ('lblfile', lblfile)

            if os.path.isfile(lblfile):
                print ('found', lblfile)
                lblfound += 1
                continue

            removeimg += 1
            #print ('removing image ', j)
            os.remove(j)

    print ('lblcount', lblcount)
    print ('lblfound', lblfound)
    print ('removeimg', removeimg)

def do_convert(method, type):
    #imgpath = "/opt/output/acdc/norm/{0}/{1}/*".format(method, type)
    imgpath = "/masvol/output/acdc/norm/{0}/{1}/*".format(method, type)
    outpath = "/masvol/output/acdc/norm/{0}/{1}L".format(method, type)
    nolv = 0
    print ('imgpath', imgpath)

    for i in glob.glob(imgpath):
        print (i)

        for j in glob.glob("{0}/*".format(i)):
            if 'label' not in j:
                continue

            print (j)
            img, delete = fix_acdc_no_delete(np.load(j)) # keep everything
            #img, delete = fix_acdc(np.load(j)) # delete the ones with all zeroes and no LV
            image = j.replace('_label','')

            if delete:
                nolv += 1
                print ('deleting, no LV or empty', j, image)
                os.remove(j)
                image = image.replace('rame0','rame')
                os.remove(image)
                continue

            outfile = j.replace('label','label_fix')
            outfile = outfile.replace('rame0','rame')
            nodes = outfile.split('/')
            print ('nodes', nodes)
            newpath = "{0}/{1}".format(outpath,nodes[-2])
            print ('newpath', newpath)

            if not os.path.isdir(newpath):
                os.mkdir(newpath)

            outfile = "{0}/{1}".format(newpath,nodes[-1])
            #print (outfile)
            np.save(outfile, img)

    print ('total nolv and empty found ', nolv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', help="normalize methods:{0}".format(", ".join(preproc.methods)))
    parser.add_argument('--type', help="normalize types:{0}".format(", ".join(preproc.types)))

    args = parser.parse_args()
    method = args.method
    type = args.type
    do_convert(method, type)
    do_delete(method, type)
