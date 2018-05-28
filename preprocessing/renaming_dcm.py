#!/usr/bin/env python

import os
import glob

patient = 499
source = "train"
#source = "validate"
#patient = 597
#patient = 234
#patient = 123
mainpath = "/masvol/data/dsb/{0}/{1}/study".format(source,patient)
origpath = "{0}/sax*/*".format(mainpath)

#/masvol/data/dsb/train/234/study/sax_19/IM-3097-0008-0001.dcm
#['', 'masvol', 'data', 'dsb', 'train', '234', 'study', 'sax_19', 'IM-3097-0008-0001.dcm']

count = 0

for i in glob.glob(origpath):
   if not i.endswith('.dcm'):
       continue

   print (i)
   nodes = i.split('/')
   print (nodes)
   filename = nodes[-1]
   print (filename)
   filenodes = filename.split('-')
   if len(filenodes) != 4:
       continue

   sax = filenodes[-1].replace('.dcm','')

   newdir = "{0}/sax_{1}".format(mainpath, int(sax))
   print (newdir)
   newname = newdir + '/' + '-'.join(filenodes[:-1]) + '.dcm'
   print (newname)

   newdirpath = os.path.dirname(newname)

   if not os.path.exists(newdirpath):
       os.makedirs(newdirpath)
   
   #os.rename(i, newname)
   os.popen("cp {0} {1}".format(i, newname))
   count += 1

   #if count > 5:
   #    break
