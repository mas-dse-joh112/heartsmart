import json
import numpy as np
import dicom
import os
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics
import random
from helpers_dicom import DicomWrapper as dicomwrapper



def computeVolume(path,imgsize, vtype = 'orig', STway = 'sub',ty = None):
    Volumes = {}
    for root, dirs, files in os.walk(path):   

        for i in files:
#             print (root+i)
            if os.path.isfile(root+i):
#                 print ('******True')
                if ('_'+str(imgsize)+'.') or ('_'+str(imgsize)+'_') in i:
                    split = i.split('_')#.strip('.json')
#                     print (split)
                    source = split[0]
                    if ('roi') in i:
                        patient = split[-2]
#                         print ('*****',split[-2])
                    elif ('CR') in i:
                        patient = split[1]
#                         print (patient)
                    else:
                        patient = split[1].strip('.json')
#                     print (source, patient)
#                     print (i)
                    patientdict = json.load(open(root + i))
#                     print (patientdict)
        
                    if vtype == 'orig':
                        ESV, EDV, issues = getVolume(patient, patientdict)
                    elif vtype == 'ST':
                        ESV, EDV, issues = getVolume(patient, patientdict, vtype, STway)
                    elif vtype == 'zeros':
#                         print ('*** ****Patient: ', patient)
                        ESV, EDV, issues = getVolume(patient, patientdict, vtype, ty)

        
                    Volumes[source+'_'+patient] = {'ESV': ESV, 
                                                  'EDV': EDV,
                                                   'issues': issues}
    return Volumes

def getVolume(patient,patientdict, v_type = 'orig', ty = None):
    es= {}
    ed = {}
    noSL = []
    noMin = []
    noMax = []
    numSlices = 0
    issues = {}
    minframes = []
    maxframes = []
    sl = 0
#     print (patient)
    if int(patient) < 500 or int(patient) == 500:
        orig_path = '/masvol/data/dsb/train/'+patient+'/study/'
    elif int(patient) > 500 and int(patient) <= 700:
        orig_path = '/masvol/data/dsb/validate/'+patient+'/study/'
    else:
        orig_path = '/masvol/data/dsb/test/'+patient+'/study/'
#     print (patientdict)
    for i in patientdict:
        numSlices = len(patientdict.keys())
        if len(patientdict.keys()) < 5:
#              print ('Less than five slices')
                sl +=1

        if patientdict[i]['minSL'] is None: 
#             print ('MinSL: ', patientdict[i])
            noSL.append(patientdict[i]['zminframe'])
        else: 
            if patientdict[i]['zmin'] ==0:
#                 print ('zmin: ', patientdict[i])
                noMin.append(patientdict[i]['zminframe'])
            es[patientdict[i]['minSL']] = {'zmin': patientdict[i]['zmin'],
                                          'minST':patientdict[i]['minST'],
                                          'zcounts': patientdict[i]['zcounts']}
            minframes.append([i, patientdict[i]['zminframe'], patientdict[i]['minSL']])
    
        if patientdict[i]['maxSL'] is None:
#             print ('MaxSL: ', patientdict[i])
            noSL.append(patientdict[i]['zmaxframe'])
        else:
            if patientdict[i]['zmax'] ==0:
#                 print ('zmax: ', patientdict[i])
                noMax.append(patientdict[i]['zmaxframe'])
            ed[patientdict[i]['maxSL']] = {'zmax': patientdict[i]['zmax'],
                                          'maxST':patientdict[i]['maxST'],
                                          'zcounts': patientdict[i]['zcounts']}
            maxframes.append([i,patientdict[i]['zmaxframe'], patientdict[i]['maxSL']])
#     print (len(maxframes), len(minframes))
    if len(maxframes) > 0:
        age_img = maxframes[0][0]+'/'+maxframes[0][1].strip('.npy')
        age = getAge(orig_path,age_img)
#     print (age_img)
    elif len(minframes) > 0:
        age_img = minframes[0][0]+'/'+minframes[0][1].strip('.npy')
        age = getAge(orig_path,age_img)
    else:
        age = None
    issues[patient] = {'numSlices': numSlices,
                        'noMinValue': noMin,
                       'noMaxValue': noMax,
                       'noSL': noSL,
                      'minframes': minframes,
                      'maxframes': maxframes,
                      'age': age}
    
    if v_type == 'orig':
        ESV, EDV = origVolume(es,ed)
        return (ESV, EDV, issues)
    if v_type == 'ST':
        ESV, EDV = STVolume(es,ed)
        return(ESV, EDV, issues)
    if v_type == 'zeros':
        ESV, EDV, issues = zerosVolume(es,ed, issues, ty)
        return (ESV, EDV, issues)
    
def origVolume(es, ed):
    ESV = 0
    EDV = 0
    
    a = sorted(es)
    b = sorted(ed)
#     print(len(a), len(b))
    for i in range(len(a)-1):
        ESVi = (es[a[i]]['zmin'] + es[a[i+1]]['zmin']) * ((abs(a[i]-a[i+1]))/2)
#         print ('********ESVi:',a[i],a[i+1],ESVi)
        ESV = ESV + ESVi
#         print ('********ESV: ', ESV)
    
    for i in range(len(b)-1):
        EDVi = (ed[b[i]]['zmax'] + ed[b[i+1]]['zmax']) * ((abs(b[i] - b[i+1])/2))
        EDV = EDV + EDVi
    
    ESV = ESV / 1000
    EDV = EDV / 1000
    return (ESV, EDV)

def zero_Top_Bottom(es,ed):
    ESV = 0
    EDV = 0
    
    a = sorted(es)
    b = sorted(ed)
    
    for i in range(len(a)-1):
        if (i == 0):
            ESVi = (0 + es[a[i+1]]['zmin']) * ((abs(a[i]-a[i+1]))/2)
            ESV = ESV + ESVi
        elif (i == (len(a)-1)):
            ESVi = (es[a[i]]['zmin'] + 0) * ((abs(a[i]-a[i+1]))/2)
            ESV = ESV + ESVi
        else:
            ESVi = (es[a[i]]['zmin'] + es[a[i+1]]['zmin']) * ((abs(a[i]-a[i+1]))/2)
            ESV = ESV + ESVi
    for i in range(len(b)-1):
        if (i == 0):
            EDVi = (0 + ed[b[i+1]]['zmax']) * ((abs(b[i]-b[i+1]))/2)
            EDV = EDV + EDVi
        elif (i == (len(b)-1)):
            EDVi = (ed[b[i]]['zmax'] + 0) * ((abs(b[i]-b[i+1]))/2)
            EDV = EDV + EDVi
        else:
            EDVi = (ed[b[i]]['zmax'] + ed[b[i+1]]['zmax']) * ((abs(b[i] - b[i+1])/2))
            EDV = EDV + EDVi
    
    ESV = ESV / 1000
    EDV = EDV / 1000
    return (ESV, EDV)
        
def STVolume(es, ed, STway = 'sub'):
    ESV = 0
    EDV = 0
    
    a = sorted(es)
    b = sorted(ed)
    
    if (len(a) > 2) and (len(b) > 2):
        filtered_a = dict((k, es[k]) for k in a[:-1] if k in es)
        filtered_b = dict((k, ed[k]) for k in b[:-1] if k in ed)
    
        ESV, EDV = origVolume(filtered_a,filtered_b)
    
        if STway == 'sub':
            ESVi = (es[a[-2]]['zmin'] + es[a[-1]]['zmin']) * ((abs(a[-2]-a[-1]-es[a[-1]]['minST']))/2)
            EDVi = (ed[b[-2]]['zmax'] + ed[b[-1]]['zmax']) * ((abs(b[-2]-b[-1]-ed[b[-1]]['maxST']))/2)
        if STway == 'add':
            ESVi = (es[a[-2]]['zmin'] + es[a[-1]]['zmin']) * ((abs(a[-2]-a[-1]))/2)
            ESVi = ESVi + (es[a[-1]]['zmin']*es[a[-1]]['minST'])
            EDVi = (ed[b[-2]]['zmax'] + ed[b[-1]]['zmax']) * ((abs(b[-2]-b[-1]))/2)
            EDVi = EDVi + (ed[b[-1]]['zmax']*ed[b[-1]]['maxST'])
        ESV = ESV + (ESVi / 1000)
        EDV = EDV + (EDVi / 1000)
    else:
        if len(a) == 2:
            for i in range(len(a)-1):
#                 print ('a', len(a), len(a-1), i)
                ESVi = (es[a[i]]['zmin'] + es[a[i+1]]['zmin']) * ((abs(a[i]-a[i+1]-es[a[i+1]]['minST']))/2)
            ESV = (ESV + ESVi)/1000
        if len(b) == 2:
            for i in range(len(b)-1):
                EDVi = (ed[b[i]]['zmax'] + ed[b[i+1]]['zmax']) * ((abs(b[i]-b[i+1]-ed[b[i+1]]['maxST']))/2)
            EDV = (EDV + EDVi)/1000
        if len(a) == 1:
            ESV = (es[a[0]]['zmin']*es[a[0]]['minST'])/1000
        if len(b) == 1:
            EDV = (ed[b[0]]['zmax']*ed[b[0]]['maxST'])/1000
    
    return (ESV, EDV)

# def zerosVolume(es,ed, ty = None):
#     ESV = 0
#     EDV = 0
    
#     a = sorted(es)
#     count = 0 
#     for i in range(len(a)):
#         count +=1
#         print ('*******SAX Number: ', a[i])
#         print (count,len(a))
#         if (es[a[i]]['zmin'] == 0) and (count > 1) and (count < len(a)):
#             print ('************** Min Zero: ',es[a[i]])
#             lst = [x for x in es[a[i]]['zcounts'] if x >0]
#             print ('Non zero Values: ',lst)
#             if len(lst) > 0:
#                 secmin = min(lst)
#                 print ('Min Update: ', secmin)
#                 es[a[i]].update({'zmin':secmin})
#                 print (es[a[i]])
#             else:
#                 del es[a[i]]
#         if (es[a[i]]['zmin'] == 0) and (len(a) == 2):
#             print ('************** Min Zero: ',es[a[i]])
#             lst = [x for x in es[a[i]]['zcounts'] if x >0]
#             print ('Non zero Values: ',lst)
#             if len(lst) > 0:
#                 secmin = min(lst)
#                 print ('Min Update: ', secmin)
#                 es[a[i]].update({'zmin':secmin})
#                 print (es[a[i]])
#     b = sorted(ed)
#     for i in range(len(b)):
#         if ed[b[i]]['zmax'] == 0:
#             del ed[b[i]]
    
#     if ty is None:
#         ESV, EDV = origVolume(es, ed)
#     else: 
#         ESV, EDV = STVolume(es,ed)
#     return (ESV, EDV)     

def zerosVolume(es,ed, issues,ty = None):
#     print ('***********************', ty)
    ESV = 0
    EDV = 0
#     for i in issues:
#         print ('**************************')
#         print (len(issues[i]['minframes']))
    a = sorted(es)
#     print (len(a))
#     print ('********ES: ',es)
    count = 0 
#     print (es)
#     print (type(es), len(a))
    if len(a) == 2:
        for i in range(len(a)):
#             print ('**********',es[a[i]]['zcounts'])
            keys = list(es[a[i]]['zcounts'].keys())
            lst = [int(x) for x in keys if int(x) >0]
            if len(lst) > 0:
                secmin = min(lst)
                secminframe = es[a[i]]['zcounts'][str(secmin)]['frame']
                es[a[i]].update({'zmin':secmin})
                for j in issues:
                    for k in range(len(issues[j]['minframes'])):
                        if a[i] == issues[j]['minframes'][k][2]:
                            issues[j]['minframes'][k][1] = secminframe
#                             issues[j]['minframes'][k].append(secmin)
    else:
#         print ('ELSE**************************')
        for i in range(len(a)):
            count +=1
#         print ('*******SL Number: ', a[i])
#         print (count,len(a))    
            if (es[a[i]]['zmin'] == 0):# and (count > 1) and (count < len(a)):
    #                 print ('************** Min Zero: ',es[a[i]])
                    keys = list(es[a[i]]['zcounts'].keys())
                    lst = [int(x) for x in keys if int(x) >0]
    #             print ('Non zero Values: ',lst)
                    if len(lst) > 0:
                        secmin = min(lst)
                        secminframe = es[a[i]]['zcounts'][str(secmin)]['frame']
#                     print ('Min Update: ', secmin)
                        es[a[i]].update({'zmin':secmin})
                        for j in issues:
                            for k in range(len(issues[j]['minframes'])):
                                if a[i] == issues[j]['minframes'][k][2]:
#                                     print ('Frame Update: ', issues[j]['minframes'][k][1], secminframe)
                                    issues[j]['minframes'][k][1] = secminframe
#                                     issues[j]['minframes'][k].append(secmin)
#                                 print (issues[j]['minframes'][k])
    #                 es[a[i]].update({'minframes': secminframe})
    #                 print (es[a[i]])
                    else:
                        del es[a[i]]
            
#         if (es[a[i]]['zmin'] == 0) and (len(a) == 2):
# #             print ('************** Min Zero: ',es[a[i]])
#             keys = list(es[a[i]]['zcounts'].keys())
#             lst = [int(x) for x in keys if int(x) >0]
# #             print ('Non zero Values: ',lst)
#             if len(lst) > 0:
#                 secmin = min(lst)
#                 secminframe = es[a[i]]['zcounts'][str(secmin)]['frame']
# #                 print ('Min Update: ', secmin)
#                 es[a[i]].update({'zmin':secmin})
#                 for j in issues:
#                     for k in range(len(issues[j]['minframes'])):
#                         if a[i] == issues[j]['minframes'][k][2]:
# #                             print ('Frame Update: ', issues[j]['minframes'][k][1], secminframe)
#                             issues[j]['minframes'][k][1] = secminframe
#                             print (issues[j]['minframes'][k])
#                 es[a[i]].update({'minframes': secminframe})
#                 print (es[a[i]])
                
    b = sorted(ed)
    for i in range(len(b)):
        if ed[b[i]]['zmax'] == 0:
            del ed[b[i]]
    
    if ty is None:
        ESV, EDV = origVolume(es, ed)
    if ty == 'ST': 
        ESV, EDV = STVolume(es,ed)
    if ty == 'TB':
#         print ('*************TB***************')
        ESV, EDV = zero_Top_Bottom(es,ed)
    return (ESV, EDV, issues)

def getSliceOutliers(Volumes, data = None):
    few_slices = []
    
    for i in Volumes:
        if data is None:
            if ('train') in i:
                p = i.strip('train_')
                if Volumes[i]['issues'][p]['numSlices'] < 5:
                    few_slices.append(p)
            if ('validate') in i: 
                p = i.strip('validate_')
                if Volumes[i]['issues'][p]['numSlices'] < 5:
                    few_slices.append(p)
        elif data == 'train':
            p = i.strip('train_')
            if Volumes[i]['issues'][p]['numSlices'] < 5:
                few_slices.append(p)
        elif data == 'validate':
            p = i.strip('validate_')
            if Volumes[i]['issues'][p]['numSlices'] < 5:
                few_slices.append(p)
            
    return few_slices

def getAge(origpath, agepath):
    img= dicomwrapper(origpath, agepath)
    age = img.PatientAge
    return age

def removeOutliers(df, Volumes, data):
    few_slices = getSliceOutliers(Volumes, data)
#     print (few_slices)
    df = df
    for i in range(len(few_slices)):
        df = df.drop(int(few_slices[i]))
    return df

def removeNoVolume(df):
    tmp = df
    all_values = tmp[(tmp['Systole_P'] > 0) & (tmp['Diastole_P'] >0)]

    return all_values

def getbest(df, actual, predicted, dev):
    diff = df[actual]-df[predicted]
#     top_low = diff.index[diff < dev].tolist()
#     top_high = diff.index[diff > -dev].tolist()
    top = diff.index[(diff > -dev) & (diff < dev)].tolist()
#     print ('**********************')
#     print (diff)
    return (top)

def removeLowHigh(df, dev, val):
    col = val+'_diff'
    tmp = df
    print (tmp.shape)
    orig = tmp.shape[0]
    low_idx = tmp[tmp[col] > dev].index.tolist()
    tmp = tmp[(tmp[col] < dev)]
    print (tmp.shape)
    high_idx = tmp[tmp[col] < -dev].index.tolist()
    tmp = tmp[(tmp[col] > -dev)]
    print (tmp.shape)
    new = tmp.shape[0]
    #print (tmp)
    return (tmp, orig-new, low_idx, high_idx)

def create_df(Volumes):
    train_gt = pd.read_csv('/masvol/data/dsb/train.csv')
    validate_gt = pd.read_csv('/masvol/data/dsb/validate.csv')
    
    train_pred = []
    validate_pred = []

    for i in Volumes:
        if ('train') in i:
            ID = i.strip('train_')
            ESV = Volumes[i]['ESV']
            EDV = Volumes[i]['EDV']
            age = Volumes[i]['issues'][ID]['age']
            age_unit = age[-1]
            age_int = int(age[:-1])
            train_pred.append([int(ID), ESV, EDV,age, age_int, age_unit])
#             print (i,ID)
#             train_pred.append([int(ID), ESV, EDV])
        if ('validate') in i:
            ID = i.strip('validate_')
            ESV = Volumes[i]['ESV']
            EDV = Volumes[i]['EDV']
            age = Volumes[i]['issues'][ID]['age']
            age_unit = age[-1]
            age_int = int(age[:-1])
            validate_pred.append([int(ID), ESV, EDV,age, age_int, age_unit])
#             print (i,ID)
#             validate_pred.append([int(ID), ESV, EDV])
    train_pred_df = pd.DataFrame(train_pred, columns= ['Id','Systole_P', 'Diastole_P',
                                                       'Age','Age_Int','Age Unit'])
    validate_pred_df = pd.DataFrame(validate_pred, columns= ['Id','Systole_P', 'Diastole_P',
                                                            'Age','Age_Int','Age Unit'])
    
    train_df = pd.concat([train_gt.set_index('Id'),train_pred_df.set_index('Id')], axis=1, join='inner')
    train_df['Systole_diff'] = train_df['Systole'] - train_df['Systole_P']
    train_df['Diastole_diff'] = train_df['Diastole'] - train_df['Diastole_P']
    train_df['EF_P'] = (train_df['Diastole_P']-train_df['Systole_P'])/train_df['Diastole_P']
    train_df['EF'] = (train_df['Diastole']-train_df['Systole'])/train_df['Diastole']
    validate_df = pd.concat([validate_gt.set_index('Id'),validate_pred_df.set_index('Id')], axis=1, join='inner')
    validate_df['Systole_diff'] = validate_df['Systole'] - validate_df['Systole_P']
    validate_df['Diastole_diff'] = validate_df['Diastole'] - validate_df['Diastole_P']
    validate_df['EF_P'] = (validate_df['Diastole_P']-validate_df['Systole_P'])/validate_df['Diastole_P']
    validate_df['EF'] = (validate_df['Diastole']-validate_df['Systole'])/validate_df['Diastole']
    all_df = pd.concat([train_df, validate_df], axis=0)
    
    return (train_df, validate_df, all_df)

def estimateCheck(df, actual, predicted, dev):
    diff = df[actual]-df[predicted]
#     print (diff)
    if dev == 0: 
        low = diff[diff > 0].count()
        correct = diff[diff == 0].count()
        high = diff[diff < 0].count()
        print (actual,': % Low Estimate:', round((low/len(df)),2)*100, '% Correct Estimate: ',
               round((correct/len(df)),2)*100,'% High Estimate: ', (round((high/len(df)),2))*100)
        
    else:
        low = diff[diff > dev].count()
        correct = diff[(diff < 5) & (-5 < diff)].count()
        high = diff[diff < -dev].count()
        print (actual,': % Low Estimate:', round((low/len(df)),2)*100, '% Correct Estimate: ',
               round((correct/len(df)),2)*100,'% High Estimate: ', (round((high/len(df)),2))*100)
        print ('Lows: ', diff.index[diff > dev].tolist())
        print ('Highs: ', diff.index[diff < -dev].tolist())
        low = round((low/len(df)),2)*100
        high = round((high/len(df)),2)*100
        return (low, high,diff.index[diff > dev].tolist(),diff.index[diff < -dev].tolist())
#         return (low, high,diff.index[diff < dev].tolist(),diff.index[diff > -dev].tolist())

def plot_outlier_imgs(img_file, pred_file, method, Type,img_txt, v, image, frames):
    idx = []
    
    f = open(img_txt, 'r')
    x = f.readlines()
    f.close()
    imgs = x[0].split(',')
    orig = []
#     print (imgs)
    if int(image) <=500:
        t = 'train_'+image
        s = 'train'
    if int(image) > 500:
        t = 'validate_'+image
        s = 'validate'
    f = v[t]['issues'][image][frames]
#     if method == '1' and Type == '3':
#         print ('**********True')
#         base_dir = '/opt/output/dsb/norm/'
#     else:
    base_dir = '/masvol/output/dsb/norm/'
    f = sorted(f, key=lambda frame: frame[2])
    for i in range(len(f)):
#         img_f = '/masvol/output/dsb/norm/1/3/'+s+'/'+image+'/'+f[i][0]+'_'+f[i][1]
        if isinstance(f[i][1], list):
#             print ('True', f[i][0])
            f_0 = f[i][0]
            f_1 = f[i][1][0]
        else:
#             print ('False', f[i][0], f[i][1])
            f_0 = f[i][0]
            f_1 = f[i][1]
        img_f_t = base_dir+method+'/'+Type+'/'+s+'/'+image+'/'+f_0+'_'+f_1
#         img_o = '/masvol/output/dsb/norm/1/3/'+s+'/'+image+'/'+f[i][0]+'_'+f[i][1]
        img_o_t = base_dir+method+'/'+Type+'/'+s+'/'+image+'/'+f_0+'_'+f_1
#         print (img_f, img_f_t)
        idx.append(imgs.index(img_f_t))
        orig.append(img_o_t)

    display_images_predictions (img_file, pred_file, image_list = idx, orig=orig)

def plotting (df, actual, predicted):
    plt.figure(figsize=(8, 6))
    plt.scatter(df[predicted], df[actual],)
    plt.plot(range(len(df[predicted])), range(len(df[actual])), color = 'green')
    plt.xlabel(predicted)
    plt.ylabel(actual)
    plt.xlim(0,600)
    plt.ylim(0, 600)
    
def display_images_predictions (image_file, pred_file,  num_images=4, image_list=False, random_images=False, orig=None):
    """Function to display images,predictions and overlays of images and predictions.       

    Args:
        image_file(:string):  image file (.npy) with full path.
        pred_file(:string):  prediction file (.npy) with full path.
        image_list (:list, optional) : list images to be displayed, if this field is present then num_images and random_images will be ignored.
        num_images (:int, optional) : number of images to be displayed, default is 4.
        random_images (:boolean, optional) : if True pick images randomly, else display first n images, default is False.
        
    Returns:
       None.
    
    Note:
        prection file should have the sigmoid outputs (not the rounded values).
    """
    ts = np.load(image_file)
    pred = np.load(pred_file)
    samples, x, y, z = pred.shape
    print ("samples, max, min ", samples, pred.max(), pred.min())
    pred2 = np.round(pred)

    display_list = []
    if image_list == False:
        if random_images == True:
            display_list = random.sample(range(0, samples), num_images)
        else :
            display_list = [i for i in range (num_images)]
    else:
        display_list = image_list
    count = 0
    for i in display_list:
        if orig is not None:
            print (orig[count])
        f, axs = plt.subplots(1,3,figsize=(15,15))
        print (i)
        plt.subplot(131),plt.imshow(ts[i].reshape(x, y))
#         plt.savefig('Original'+ str(i) +'.png')
        plt.title('Image '+str(i)), plt.xticks([]), plt.yticks([])
        plt.subplot(132),plt.imshow(pred2[i].reshape(x, y))
#         plt.savefig('Prediction'+ str(i) +'.png')
        plt.title('Prediction'), plt.xticks([]), plt.yticks([])
        plt.subplot(133),plt.imshow(ts[i].reshape(x, y)), plt.imshow(pred2[i].reshape(x, y), 'binary', interpolation='none', alpha=0.3)
#         plt.savefig('Overlay'+ str(i) +'.png')
        plt.title('Overlay'), plt.xticks([]), plt.yticks([])
        plt.show()
        count +=1

def compute_rmse(df, source, val = None):
    if val is None:
        sys = np.sqrt(sklearn.metrics.mean_squared_error(df['Systole'],df['Systole_P']))
        dia = np.sqrt(sklearn.metrics.mean_squared_error(df['Diastole'],df['Diastole_P']))
        ef = np.sqrt(sklearn.metrics.mean_squared_error(df['EF'],df['EF_P']))
#         print (source,': Systole RMSE: ', sys, 'Diastole RMSE: ', dia, ': EF RMSE: ', ef)
        print (source,": ","Systole RMSE: %.2f" % round(sys,2), "ml ", "Diastole RMSE: '%.2f" % round(dia,2),
               "ml ","EF RMSE: %.2f" % (round(ef,2)*100), "%")
        return (round(sys,2),round(dia,2), (round(ef,2)*100))
    elif ('D') in val:
        dia = np.sqrt(sklearn.metrics.mean_squared_error(df['Diastole'],df['Diastole_P']))
        print (source,'Diastole RMSE: ', dia)
    elif ('S') in val:
        sys = np.sqrt(sklearn.metrics.mean_squared_error(df['Systole'],df['Systole_P']))
        print (source,': Systole RMSE: ', sys)
    elif ('E') in val:
        ef = np.sqrt(sklearn.metrics.mean_squared_error(df['EF'],df['EF_P']))
        print (source, ': EF RMSE: ', ef)
        
        
