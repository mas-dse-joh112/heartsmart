
import os
import glob

imgpath = "/opt/output/sunnybrook/norm/1/2/images/*"
lblpath = "/opt/output/sunnybrook/norm/1/2/challenge/*"

total = 0

for i in glob.glob(lblpath):
    found = 0
    notfound = 0

    for j in glob.glob("{0}/*".format(i)):
        print (j)
        ifile = j.replace('challenge','images')
        ifile = ifile.replace('dcm.label','dcm')
        if os.path.isfile(ifile):
            found += 1
            total += 1
            print (ifile)
            continue

        print (j)
        #os.remove(j)
        notfound += 1

    print (found, notfound)

print (total)
