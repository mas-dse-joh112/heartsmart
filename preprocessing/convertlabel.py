#!/usr/bin/env python

import preproc
import argparse
import os
import glob
import numpy as np

def fix_acdc(im):
    max_val=im.max()
    im[im<max]=0
    im[im==max]=1
    return im

def do_convert(method, type):
    imgpath = "/opt/output/acdc/norm/{0}/{1}/*".format(method, type)
    outpath = "/opt/output/acdc/norm/{0}/{1}L".format(method, type)

    for i in glob.glob(imgpath):
	print (i)

	for j in glob.glob("{0}/*".format(i)):
	    if 'label' not in j:
		continue

	    print (j)
	    img = fix_acdc(np.load(j))
	    outfile = j.replace('label','label_fix')
	    nodes = outfile.split('/')
	    newpath = "{0}/{1}".format(outpath,nodes[-2])

	    if not os.path.isdir(newpath):
		os.mkdir(newpath)

	    outfile = "{0}/{1}".format(newpath,nodes[-1])
	    print (outfile)
	    np.save(outfile, img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', help="normalize methods:{0}".format(", ".join(preproc.methods)))
    parser.add_argument('--type', help="normalize types:{0}".format(", ".join(preproc.types)))

    args = parser.parse_args()
    method = args.method
    type = args.type
    do_convert(method, type)
