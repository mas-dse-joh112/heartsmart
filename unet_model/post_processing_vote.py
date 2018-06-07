#!/usr/bin/env python

""" Apply majority vote post processing """

import glob
from helper_utilities import *


sources = ['validate','train']
method = 1
Type = 3
image_size = 176
test_source_path = "/masvol/output/dsb/norm/{0}/{1}".format(method, Type)

def do_post_processing(volpath, ensampath):
    global sources;
    global image_size
    global test_source_path
    global method
 
    for source in sources:
        data_source = "unet_model_{0}".format(source)

        for patient in glob.glob('/masvol/data/dsb/{0}/*'.format(source)):
            nodes = patient.split('/')
            patient = nodes[-1]
     
            for sys in glob.glob(ensampath+'*'):
                syspatient = sys.split('/')[-1].split('_')[0]

                if 'round_predict.npy' not in sys:
                    continue
 
                try:
                    if int(patient) != int(syspatient): 
                        continue
                except:
                    print ('error ', syspatient, patient)
                    continue

                print (syspatient, patient)
                print (np.load(sys).shape)
        
                predictions = remove_contour(np.load(sys))
                topdir = "{0}/{1}".format(test_source_path, data_source)
                image_source_file = "{0}/data/dsb_{1}_image_path.txt".format(topdir, patient)
                sourcedict = get_ones(predictions, image_source_file)

                pred_file_CR = "{0}/dsb_{1}_{2}_CR4d_predictions_cleaned.npy".format(ensampath, patient, image_size)
                np.save(pred_file_CR, predictions)                     

                image_one_file = "{0}/dsb_{1}_{2}_one_count.json".format(ensampath, patient, image_size)

                with open(image_one_file, 'w') as output:
                    output.write("{0}\n".format(json.dumps(sourcedict)))

                origpath = '/masvol/data/dsb/{0}/{1}/study/'.format(source,patient)
                newpath = '/masvol/output/dsb/volume/{0}/{1}/{2}_{3}_{4}.json'.format(method,volpath,source,patient,image_size)

                dirname = os.path.dirname(newpath)
                print ('dirname', dirname)

                if not os.path.exists(dirname):
                    os.makedirs(dirname)

                dump_and_sort(image_one_file, origpath, newpath)

if __name__ == "__main__":
    syspath = 'systolic_vote'
    sys_pred = '/masvol/output/dsb/norm/1/3/ensamble/systolic_vote/'
    do_post_processing(syspath, sys_pred)

    diapath = 'diastolic_vote'
    dia_pred = '/masvol/output/dsb/norm/1/3/ensamble/diastolic_vote/'
    do_post_processing(diapath, dia_pred)
