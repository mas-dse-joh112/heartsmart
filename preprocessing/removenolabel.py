
import os
import glob

imgpath = "/opt/output/sunnybrook/norm/1/3/images/*"

for i in glob.glob(imgpath):
    print (i)
    found = 0
    notfound = 0

    for j in glob.glob("{0}/*".format(i)):
        labelfile = j.replace('images','challenge')
        labelfile = labelfile.replace('dcm','dcm.label')
        if os.path.isfile(labelfile):
            found += 1
            continue

        print (j)
        os.remove(j)
        notfound += 1

    print (found, notfound)
