#!/usr/bin/env python

import dicom, cv2, re, sys
import os, fnmatch, shutil, subprocess
import numpy as np
import method1 as m1
import method2 as m2

# Mapping between the Contour and the Image that the Contour is for with in the patient images
SAX_SERIES = {
    # challenge training
    "SC-HF-I-1": "0004",
    "SC-HF-I-2": "0106",
    "SC-HF-I-4": "0116",
    "SC-HF-I-10": "0024",
    "SC-HF-I-40": "0134",
    "SC-HF-NI-3": "0379",
    "SC-HF-NI-4": "0501",
    "SC-HF-NI-34": "0446",
    "SC-HF-NI-36": "0474",
    "SC-HYP-1": "0550",
    "SC-HYP-3": "0650",
    "SC-HYP-38": "0734",
    "SC-HYP-40": "0755",
    "SC-N-2": "0898",
    "SC-N-3": "0915",
    "SC-N-40": "0944",
    "SC-HF-I-5": "0156",
    "SC-HF-I-6": "0180",
    "SC-HF-I-7": "0209",
    "SC-HF-I-8": "0226",
    "SC-HF-NI-11": "0270",
    "SC-HF-NI-31": "0401",
    "SC-HF-NI-33":"0424",
    "SC-HF-NI-7": "0523",
    "SC-HYP-37": "0702",
    "SC-HYP-6": "0767",
    "SC-HYP-7": "0007",
    "SC-HYP-8": "0796",
    "SC-N-5": "0963",
    "SC-N-6": "0981",
    "SC-N-7": "1009",
    "SC-HF-I-11": "0043",
    "SC-HF-I-12": "0062",
    "SC-HF-I-9": "0241",
    "SC-HF-NI-12": "0286",
    "SC-HF-NI-13": "0304",
    "SC-HF-NI-14": "0331",
    "SC-HF-NI-15": "0359",
    "SC-HYP-10": "0579",
    "SC-HYP-11": "0601",
    "SC-HYP-12": "0629",
    "SC-HYP-9": "0003",
    "SC-N-10": "0851",
    "SC-N-11": "0878",
    "SC-N-9": "1031"
}

#Update with Sunnybrook data path
SUNNYBROOK_ROOT_PATH = "/opt/data/sunnybrook/"

#Update with Train Contour Path
TRAIN_CONTOUR_PATH = os.path.join(SUNNYBROOK_ROOT_PATH, "SCD_ManualContours")

#Update with Train Image Path
TRAIN_IMG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH, "challenge")


def shrink_case(case):
    toks = case.split("-")
    def shrink_if_number(x):
        try:
            cvt = int(x)
            return str(cvt)
        except ValueError:
            return x
    return "-".join([shrink_if_number(t) for t in toks])

class Contour(object):
    def __init__(self, ctr_path):
        self.ctr_path = ctr_path
        match = re.search(r"/([^/]*)/contours-manual/IRCCI-expert/IM-0001-(\d{4})-icontour-manual.txt", ctr_path)
        self.case = shrink_case(match.group(1))
        self.img_no = int(match.group(2))
    
    def __str__(self):
        return "<Contour for case %s, image %d>" % (self.case, self.img_no)
    
    __repr__ = __str__

def load_contour(contour, img_path,method = None):
    filename = "IM-%s-%04d.dcm" % (SAX_SERIES[contour.case], contour.img_no)
    full_path = os.path.join(img_path, contour.case, filename)
    
    if not os.path.isfile(full_path):
        return [], [], full_path
    
    f = dicom.read_file(full_path)
    img = f.pixel_array.astype(np.int)
    ctrs = np.loadtxt(contour.ctr_path, delimiter=" ").astype(np.int)
    label = np.zeros_like(img, dtype="uint8")
    cv2.fillPoly(label, [ctrs], 1)
    #ADD IN NORMALIZATION HERE
    if method == 1:
        label = m1.InPlanePhaseEncoding(f, label)
        label = m1.rescale(label, f.PixelSpacing[0])
        label = m1.get_square_crop(label)
        
    if method == 2:
        label = m2.getAlignImg(f, label)
        label = m2.crop_size(label)
        
    return img, label, full_path
    
def get_all_contours(contour_path):
    contours = [os.path.join(dirpath, f)
        for dirpath, dirnames, files in os.walk(contour_path)
        for f in fnmatch.filter(files, 'IM-0001-*-icontour-manual.txt')]
    print("Shuffle data")
    np.random.shuffle(contours)
    print("Number of examples: {:d}".format(len(contours)))
    extracted = map(Contour, contours)
    return extracted

def export_all_contours(contours, img_path, lmdb_img_name, lmdb_label_name, m1=False, m2=False):
    for lmdb_name in [lmdb_img_name, lmdb_label_name]:
        db_path = os.path.abspath(lmdb_name)
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
    counter_img = 0
    counter_label = 0
    batchsz = 100
    if m1 is False & m2 is False:
        outpath = '/opt/output/sunnybrook/challenge'
    if m1 is True:
        outpath = '/opt/output/sunnybrook/norm/1/contours'
    if m2 is True:
        m2outpath = '/opt/output/sunnybrook/norm/2/contours'
    
    print("Processing {:d} images and labels...".format(len(contours)))
    
    for i in xrange(int(np.ceil(len(contours) / float(batchsz)))):
        batch = contours[(batchsz*i):(batchsz*(i+1))]
        if len(batch) == 0:
            break
        
        for idx,ctr in enumerate(batch):
            if m1 is False & m2 is False:
                img, label, full_path = load_contour(ctr, img_path)
            if m1 is True:
                img, label, full_path = load_contour(ctr,img_path, 1)
            if m2 is True:
                img, label, full_path = load_contour(ctr, img_path, 2)
    
            if len(img) == 0:
                print 'missing ', full_path
                continue
                
            filepath, filename = full_path.split('/')[-2:]
            #print 'full_path', full_path
            #print 'img', img
            #print 'label', label
            outfullpath = "{0}/{1}".format(outpath, filepath)
            print outfullpath
            
            if not os.path.exists(outfullpath):
                os.mkdir(outfullpath) 
            
            np.save("{0}/{1}.img.npy".format(outfullpath, filename), img)
            np.save("{0}/{1}.label.npy".format(outfullpath, filename), label)
                
            if idx % 20 == 0:
                print ctr

    print 'missing', missing


if __name__== "__main__":
    SPLIT_RATIO = 0.1
    print("Mapping ground truth contours to images...")
    ctrs = get_all_contours(TRAIN_CONTOUR_PATH)
    print("Done mapping ground truth contours to images")
    export_all_contours(ctrs, TRAIN_IMG_PATH, "train_images_lmdb", "train_labels_lmdb")
