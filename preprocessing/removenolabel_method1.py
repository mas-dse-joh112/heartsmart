
import os
import glob

imgpath = "/masvol/output/sunnybrook/norm/1/1/images/*"

for i in glob.glob(imgpath):
    print (i)
    found = 0
    notfound = 0

    for j in glob.glob("{0}/*".format(i)):
        labelfile = j.replace('images','labels')
        labelfile = labelfile.replace('dcm','dcm.label')

        if os.path.isfile(labelfile):
            found += 1
            continue

        print (j)
        os.remove(j)
        notfound += 1

    print (found, notfound)
