#!/usr/bin/env python

import sys
import dicom
import os
import numpy as np
import glob
import cv2 
import preproc
import argparse
from method1 import *


class FindMethod(object):
    def __init__(self, arg):
        self.arg = arg
        print self.arg
        self.start = self.arg.start
        self.end = self.arg.end
        self.method = self.arg.method
        self.path = self.arg.path
        self.train_files = None

    def get_method(self):
        if self.method not in preproc.methods:
            print "not a valid method"
            print ", ".join(preproc.methods)
            sys.exit()

        if self.path not in preproc.paths:
            print "not a valid path"
            print ", ".join(preproc.paths)
            sys.exit()

        if self.start == '':
            print "not a valid start"
            sys.exit()

        try:
            if int(self.end) < int(self.start):
                self.end = self.start
                print "starting from {0}".format(self.start)
                print "ending at {0}".format(self.end)
        except:
            print "not a valid start or end"
            sys.exit() 

        if self.method == '1':
            method = Method1(self.arg)
            method.main_process()
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', help='patient folder start number')
    parser.add_argument('--end', help='patient folder end number')
    parser.add_argument('--method', help="normalize methods:{0}".format(", ".join(preproc.methods)))
    parser.add_argument('--path', help="folder paths:{0}".format(", ".join(preproc.paths)))
    parser.add_argument('--source', help="sources:{0}".format(", ".join(preproc.sources.keys())))

    args = parser.parse_args()

    fm = FindMethod(args) 
    fm.get_method()
