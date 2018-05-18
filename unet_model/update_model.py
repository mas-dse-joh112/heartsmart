#!/usr/bin/env python

import json
import MySQLdb
import ast
import sys


db = {'host':'localhost', 'user':'root', 'password':'mastf','db':'mastf','port':3306}

conn = MySQLdb.connect(host=db['host'],user=db['user'],password=db['password'],db=db['db'],port=db['port'])
curs = conn.cursor()

def insert_model(row):
    sql = """insert into models (tr_model_name, tr_lrrate, tr_epoch, tr_size,
             tr_dropout, tr_optimizer, tr_loss_fn, tr_nGPUs, tr_batchsize,
             true_negative, true_positive, false_positive, false_negative,
             recall, `precision`, f1_score, accuracy, weighted_accuracy,
             eval_dice_coeff, logloss, weighted_logloss, eval_binary_accuracy,
             eval_loss, learning_history, predictions, image_size, training_images,
             training_labels, test_images, test_labels, model_path, 
             augmentation, model_summary, performance, modelfile)
             values ({0}) """.format(",".join(["'"+str(x)+"'" for x in row]))

    print (sql)
    curs.execute(sql)
    curs.execute("commit;")

if __name__ == "__main__":
    #inputfile = "/opt/unet_model/data/combine1_1_256_dice_aug_drop_32B.log"
    inputfile = sys.argv[1]
    print (inputfile)

    linestr = 'total arguments 15 '

    with open(inputfile, 'r') as inputs:
        for i in inputs:
            i = i.strip()
            if linestr in i:
                nodes = i.replace(linestr,'')
                parms = ast.literal_eval(nodes)

                model_path = parms[6] # model_path
                model_name = parms[0] # model_name
                performance = "{0}/{1}_performance.json".format(model_path, model_name) # performance

                row = []

                with open(performance, 'r') as perfin:
                    pjson = json.load(perfin)
                    row.append(model_name)
                    row.append(pjson['tr_lrrate'])
                    row.append(pjson['tr_epoch'])
                    row.append(pjson['tr_size'])
                    tr_dropout = pjson['tr_dropout']

                    if tr_dropout == 'True':
                        row.append(1)
                    else:
                        row.append(0)

                    row.append(pjson['tr_optimizer'])
                    row.append(pjson['tr_loss_fn'])

                    try:
                        row.append(pjson['tr_nGPUs'])
                    except:
                        row.append(0)

                    row.append(pjson['tr_batchsize'])
                    row.append(pjson['true_negative'])
                    row.append(pjson['true_positive'])
                    row.append(pjson['false_positive'])
                    row.append(pjson['false_negative'])
                    row.append(pjson['recall'])
                    row.append(pjson['precision'])
                    row.append(pjson['f1_score'])
                    row.append(pjson['accuracy'])
                    row.append(pjson['weighted_accuracy'])
                    row.append(pjson['eval_dice_coeff'])
                    row.append(pjson['logloss'])
                    row.append(pjson['weighted_logloss'])
                    row.append(pjson['eval_binary_accuracy'])
                    row.append(pjson['eval_loss'])
             
                learning_history = "{0}/{1}_learning_history.json".format(model_path, model_name) # learning_history
                row.append(learning_history)
                predictions = "{0}/{1}_predictions.npy".format(model_path, model_name) # predictions
                row.append(predictions)
                row.append(parms[1]) # image size
                training_images = parms[2] # training_images
                row.append(training_images)
                training_labels = parms[3] # training_labels
                row.append(training_labels)
                test_images = parms[4] # test_images
                row.append(test_images)
                test_labels = parms[5] # test_labels
                row.append(test_labels)
                row.append(model_path)
                augmentation = parms[13] # augmentation

                if augmentation == 'True':
                    row.append(1)
                else:
                    row.append(0)

                model_summary = parms[14] # model_summary

                if model_summary == 'True':
                    row.append(1)
                else:
                    row.append(1)

                modelfile = "{0}/{1}.hdf5".format(model_path, model_name) # predictions
                row.append(performance)
                row.append(modelfile)

                insert_model(row)
                #print (parms)
                break 
