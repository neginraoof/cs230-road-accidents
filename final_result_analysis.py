###############################################################################
# Project CS230: Result Analysis
# Authors: Matias, Negin, Alex

# date: March 24, 2020

## Final Version
###############################################################################

import json
import numpy as np
from shapely.geometry import Polygon
import pandas as pd
from pyproj import Proj, transform
from sklearn.cluster import KMeans
import numpy.ma as ma
from scipy import stats
import random
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

def calculate_results_old(file,label_file):
    
    '''
    This function calculates train set evaluation metrics for the 6 models using Approach 1 as
    described in the report.

    Args
    - file: csv file including results 
    - label_file: file including true labels 
    
    Output
    - Includes aggregate as well as class-specific accuracies and confusion
    matrices both for the average probability and majority voting approach
    as described in the report 
    '''
    
    resA_1fps = pd.read_csv(file)
    resA_1fps = resA_1fps.sort_values(by=['video_id'])
    
    print('Total Number of Video Clips is {}'.format(len(resA_1fps)))
    print("Number of Videos is {}".format(len(resA_1fps.video_id.unique())))
    
    len_vids= len(resA_1fps.video_id.unique())
    
    #Average Probability Approach
    
    average_prob =resA_1fps.groupby('video_id').mean()
    average_prob['max_prob'] = average_prob.loc[:,'p1':'p4'].idxmax(axis=1)
    average_prob['pred_label'] = 0
    
    print("Number of Videos (Average) is {}".format(len(average_prob)))
    
    for i in range(len(average_prob)):
        if average_prob['max_prob'].iloc[i] == 'p1':
            average_prob['pred_label'].iloc[i]= 1
        elif average_prob['max_prob'].iloc[i] == 'p2':
            average_prob['pred_label'].iloc[i]= 2
        elif average_prob['max_prob'].iloc[i] == 'p3':
            average_prob['pred_label'].iloc[i]= 3
        elif average_prob['max_prob'].iloc[i] == 'p4':
            average_prob['pred_label'].iloc[i]= 4
            
    #Majority Probability Approach
    
    resA_1fps['max_prob'] = resA_1fps.loc[:,'p1':'p4'].idxmax(axis=1)
    
    resA_1fps['most_p1'] = (resA_1fps['max_prob']=='p1')
    resA_1fps['most_p2'] = (resA_1fps['max_prob']=='p2')
    resA_1fps['most_p3'] = (resA_1fps['max_prob']=='p3')
    resA_1fps['most_p4'] = (resA_1fps['max_prob']=='p4')
    
    
    major_prob = resA_1fps.groupby('video_id').sum()
    major_prob['max_prob'] = major_prob.loc[:,'most_p1':'most_p4'].idxmax(axis=1)
    
    major_prob['pred_label'] = 0
    
    for i in range(len(average_prob)):
        if major_prob['max_prob'].iloc[i] == 'most_p1':
            major_prob['pred_label'].iloc[i]= 1
        elif major_prob['max_prob'].iloc[i] == 'most_p2':
            major_prob['pred_label'].iloc[i]= 2
        elif major_prob['max_prob'].iloc[i] == 'most_p3':
            major_prob['pred_label'].iloc[i]= 3
        elif major_prob['max_prob'].iloc[i] == 'most_p4':
            major_prob['pred_label'].iloc[i]= 4
            
    def_dif = np.sum(major_prob['pred_label']==average_prob['pred_label'])/len_vids
    print('Definitions coincide in {} percent of cases'.format(def_dif*100))
    
    equiv_pred = major_prob['pred_label']==average_prob['pred_label']
    index_false = equiv_pred[equiv_pred==False].index
    
    #Merge with True Labels
    true_labels = pd.read_csv(label_file, header=0)
    true_labels = true_labels.drop(['vid_name'],axis=1)
    
    true_labels['true_lab2'] = 5
    for i in range(len(true_labels)):
        if true_labels['true_lab'].iloc[i] == ' label:  0':
            true_labels['true_lab2'].iloc[i]= 1
        elif true_labels['true_lab'].iloc[i] == ' label:  1':
            true_labels['true_lab2'].iloc[i]= 2
        elif true_labels['true_lab'].iloc[i] == ' label:  2':
            true_labels['true_lab2'].iloc[i]= 3
        elif true_labels['true_lab'].iloc[i] == ' label:  3':
            true_labels['true_lab2'].iloc[i]= 4
            
    true_labels =true_labels.drop(['true_lab'],axis=1)
    true_labels = true_labels.groupby('video_id').mean()
    
    merged_labels = true_labels.merge(average_prob,left_on='video_id', right_on='video_id')
    merged_labels2 = true_labels.merge(major_prob,left_on='video_id', right_on='video_id')

    #Accuracies
    accuracy_average = np.sum(merged_labels['true_lab2'] == merged_labels['pred_label'])/len(merged_labels)
    accuracy_major = np.sum(merged_labels2['true_lab2'] == merged_labels2['pred_label'])/len(merged_labels)

    print('Accuracy for the average probability approach is {} percent'.format(accuracy_average*100))
    print('Accuracy for the majority probability approach is {} percent'.format(accuracy_major*100))

    
    
    equiv_lab = merged_labels['true_lab2'] == merged_labels['pred_label']
    index_labfalse = equiv_lab[equiv_lab==False].index
    
    equiv_lab2 = merged_labels2['true_lab2'] == merged_labels2['pred_label']
    index_labfalse2 = equiv_lab2[equiv_lab2==False].index
    
    #Confusion Matrix
    expected = list(merged_labels['true_lab2'])
    predicted = list(merged_labels['pred_label'])
    conf_matrix = confusion_matrix(expected, predicted)
    
    expected2 = list(merged_labels2['true_lab2'])
    predicted2 = list(merged_labels2['pred_label'])
    conf_matrix2 = confusion_matrix(expected2, predicted2)
    
    print('The confusion matrix for the average approach is')
    print(conf_matrix)
    
    print('The confusion matrix for the majority approach is')
    print(conf_matrix2)
    
    print('Additional Accuracy Metrics:Average Prob')
    print('True distribution of Labels')
    print((np.sum(conf_matrix,axis=1)/np.sum(conf_matrix))*100)
    print('Predicted distribution of Labels')
    print((np.sum(conf_matrix,axis=0)/np.sum(conf_matrix))*100)
    print('Class Specific Accuracy')
    print((np.diag(conf_matrix)/np.sum(conf_matrix,axis=1))*100)

    
    print('Additional Accuracy Metrics:Major Prob')
    print('True distribution of Labels')
    print((np.sum(conf_matrix2,axis=1)/np.sum(conf_matrix2))*100)
    print('Predicted distribution of Labels')
    print((np.sum(conf_matrix2,axis=0)/np.sum(conf_matrix2))*100)
    print('Class Specific Accuracy')
    print((np.diag(conf_matrix2)/np.sum(conf_matrix2,axis=1))*100)
    
    return accuracy_average,accuracy_major,conf_matrix,conf_matrix2,index_labfalse,index_labfalse2,index_false,major_prob,average_prob

files = ['catA_1fps_train_epoch_9','catA_10fps_train_epoch_9','catB_1fps_train_epoch_9','catB_10fps_train_epoch_7','catC_1fps_train_epoch_9','catC_10fps_train_epoch_6']
#Get Train Set Results 

for file in files:
    print('Results for model'+file)
    accuracy_average,accuracy_major,conf_matrix,conf_matrix2,index_labfalse,index_labfalse2,index_false,major_prob,average_prob = calculate_results_old('results/train/'+file+'.csv','temp.csv')

#Get Validation Set Results

#Get true labels for validation set 
true_labels = pd.read_csv('conv3d_10fps_validation_labels.csv', header=0)
true_labels = true_labels.sort_values(by=['vid_name'])
len(true_labels)
true_labels = true_labels.iloc[:3013] #This is getting rid of redundant rows 
valid_videos= ['road_h148_s4_a_2', 'road_h106_s3_b', 'road_h119_s1_b',
 'road_h139_s7,8,9_a_1', 'road_h14_s2_a', 'road_h184_s4_b', 'road_h157_s2_a',
 'road_h153_s4_ar_bl', 'road_h62_s5_a_1', 'road_h35_s1_a_2',
 'road_h29_s12_a_1', 'road_h90_s3_a_1', 'road_h162_s6_a_b', 'road_h134_1,2_b',
 'road_h98_s3_b', 'road_h98_s3_a', 'road_h201_s2_b', 'road_h121_s5_a,b',
 'road_h14_s2_b', 'road_h57_s7&8_a_1', 'road_h171_s1_2_3_b', 'road_h137_s2_a',
 'road_h90_s3_b_2', 'road_h243_s1_a', 'road_h90_s4_b_1', 'road_h121_s1_a,b',
 'road_h177_s2,3,4_a_2', 'road_h106_s3_a', 'road_h188_s4_a,b',
 'road_h147_s4_a', 'road_h123_s5a', 'road_h128_s2_b,a', 'road_h162_s4_a_b',
 'road_h110_s1,2,3_a', 'road_h95_s8', 'road_h40_s1_a', 'road_h164_s2_s3_b',
 'road_h127_s1_a', 'road_h72_s5_a_b', 'road_h64_s3_b', 'road_h127_s2_b',
 'road_h60_s2,3_b', 'road_h20_s1_b', 'road_h95_s4_a', 'road_h153_s7_b',
 'road_h109_s1,2_a', 'road_h148_s3_a_1', 'road_h80_s3,4,5_a_2',
 'road_h25_s6_aright_bleft_1', 'road_h42_s1_s2_b', 'road_h26_s1_b.3gp',
 'road_h72_s6_b', 'road_h52_s2_s1_b', 'road_h94_s1_ar_bl', 'road_h40_s1,2_b',
 'road_h53_s3_b', 'road_h47_s2_b', 'road_h76_s1_b[2]', 'road_h59_s3_b',
 'road_h177_s5_a_2', 'road_h130_s2_aright_bleft', 'road_h72_s2_b',
 'road_h65_s2_b', 'road_h145_s2,3_b', 'road_h34_s1_a', 'road_h95_s2_b',
 'road_h100_s4_b', 'road_h34_s2_b', 'road_h12_s1_a_2', 'road_h12_s2_b_2',
 'road_h18_s7_a', 'road_h43_s2_a', 'road_h153_s9_b', 'road_h153_s9_a',
 'road_h6_s1_aa', 'road_h28_s6_a_1.3gp', 'road_h14_s4_a', 'road_h86_s1_b',
 'road_h4_s1_a', 'road_h17_s2_b', 'road_h144_s7_b_1', 'road_h10_s2_b',
 'road_h2_s2_s3_a', 'road_h144_s6_afar_bclose_1', 'road_h10_s5_a',
 'road_h7_s5_b', 'road_h150_s1_a_1']

#Selecting the Validation Set Videos
true_labels['valid'] = 0 
for i in range(len(true_labels)):
    #print(i)
    for name in valid_videos:
        if name in true_labels['vid_name'].iloc[i]:
            true_labels['valid'].iloc[i] = 1

true_labels = true_labels[true_labels.valid==1]
true_labels = true_labels.sort_values(by=['video_id'])
true_labels = true_labels.drop(['vid_name'],axis=1)
true_labels = true_labels.groupby('video_id').mean()
len(true_labels)
true_labels['vid_id2'] =true_labels.index
true_labels.vid_id2 = true_labels.vid_id2.astype('int64')


def calculate_results_valid_old(file,true_labels):
    
    '''
    This function calculates validation set evaluation metrics for the 6 models using Approach 1 as
    described in the report.

    Args
    - file: csv file including results 
    - true_labels: file including true labels 
    
    Output
    - Includes aggregate as well as class-specific accuracies and confusion
    matrices both for the average probability and majority voting approach
    as described in the report 
    '''

    
    resA_1fps = pd.read_csv(file)
    resA_1fps = resA_1fps.sort_values(by=['video_id'])
    
    print('Total Number of Video Clips is {}'.format(len(resA_1fps)))
    print("Number of Videos is {}".format(len(resA_1fps.video_id.unique())))
    
    len_vids= len(resA_1fps.video_id.unique())
    
    #Average Probability Approach
    
    average_prob =resA_1fps.groupby('video_id').mean()
    average_prob['max_prob'] = average_prob.loc[:,'p1':'p4'].idxmax(axis=1)
    average_prob['pred_label'] = 0
    
    print("Number of Videos (Average) is {}".format(len(average_prob)))
    
    for i in range(len(average_prob)):
        if average_prob['max_prob'].iloc[i] == 'p1':
            average_prob['pred_label'].iloc[i]= 1
        elif average_prob['max_prob'].iloc[i] == 'p2':
            average_prob['pred_label'].iloc[i]= 2
        elif average_prob['max_prob'].iloc[i] == 'p3':
            average_prob['pred_label'].iloc[i]= 3
        elif average_prob['max_prob'].iloc[i] == 'p4':
            average_prob['pred_label'].iloc[i]= 4
            
    #Majority Probability Approach
    
    resA_1fps['max_prob'] = resA_1fps.loc[:,'p1':'p4'].idxmax(axis=1)
    
    resA_1fps['most_p1'] = (resA_1fps['max_prob']=='p1')
    resA_1fps['most_p2'] = (resA_1fps['max_prob']=='p2')
    resA_1fps['most_p3'] = (resA_1fps['max_prob']=='p3')
    resA_1fps['most_p4'] = (resA_1fps['max_prob']=='p4')
    
    
    major_prob = resA_1fps.groupby('video_id').sum()
    major_prob['max_prob'] = major_prob.loc[:,'most_p1':'most_p4'].idxmax(axis=1)
    
    major_prob['pred_label'] = 0
    
    for i in range(len(average_prob)):
        if major_prob['max_prob'].iloc[i] == 'most_p1':
            major_prob['pred_label'].iloc[i]= 1
        elif major_prob['max_prob'].iloc[i] == 'most_p2':
            major_prob['pred_label'].iloc[i]= 2
        elif major_prob['max_prob'].iloc[i] == 'most_p3':
            major_prob['pred_label'].iloc[i]= 3
        elif major_prob['max_prob'].iloc[i] == 'most_p4':
            major_prob['pred_label'].iloc[i]= 4
            
    def_dif = np.sum(major_prob['pred_label']==average_prob['pred_label'])/len_vids
    print('Definitions coincide in {} percent of cases'.format(def_dif*100))
    
    equiv_pred = major_prob['pred_label']==average_prob['pred_label']
    index_false = equiv_pred[equiv_pred==False].index
    
    #Merge with True Labels    
    true_labels['true_lab2'] = 5
    for i in range(len(true_labels)):
        if true_labels['true_lab'].iloc[i] == 0:
            true_labels['true_lab2'].iloc[i]= 1
        elif true_labels['true_lab'].iloc[i] == 1:
            true_labels['true_lab2'].iloc[i]= 2
        elif true_labels['true_lab'].iloc[i] == 2:
            true_labels['true_lab2'].iloc[i]= 3
        elif true_labels['true_lab'].iloc[i] == 3:
            true_labels['true_lab2'].iloc[i]= 4
            
    true_labels =true_labels.drop(['valid'],axis=1)
    true_labels =true_labels.drop(['true_lab'],axis=1)
    
    merged_labels = true_labels.merge(average_prob,left_on='vid_id2', right_on='video_id')
    merged_labels2 = true_labels.merge(major_prob,left_on='vid_id2', right_on='video_id')

    #Accuracies
    accuracy_average = np.sum(merged_labels['true_lab2'] == merged_labels['pred_label'])/len(merged_labels)
    accuracy_major = np.sum(merged_labels2['true_lab2'] == merged_labels2['pred_label'])/len(merged_labels)

    print('Accuracy for the average probability approach is {} percent'.format(accuracy_average*100))
    print('Accuracy for the majority probability approach is {} percent'.format(accuracy_major*100))

    
    
    equiv_lab = merged_labels['true_lab2'] == merged_labels['pred_label']
    index_labfalse = equiv_lab[equiv_lab==False].index
    
    equiv_lab2 = merged_labels2['true_lab2'] == merged_labels2['pred_label']
    index_labfalse2 = equiv_lab2[equiv_lab2==False].index
    
    #Confusion Matrix
    expected = list(merged_labels['true_lab2'])
    predicted = list(merged_labels['pred_label'])
    conf_matrix = confusion_matrix(expected, predicted)
    
    expected2 = list(merged_labels2['true_lab2'])
    predicted2 = list(merged_labels2['pred_label'])
    conf_matrix2 = confusion_matrix(expected2, predicted2)
    
    print('The confusion matrix for the average approach is')
    print(conf_matrix)
    
    print('The confusion matrix for the majority approach is')
    print(conf_matrix2)
    
    print('Additional Accuracy Metrics:Average Prob')
    print('True distribution of Labels')
    print((np.sum(conf_matrix,axis=1)/np.sum(conf_matrix))*100)
    print('Predicted distribution of Labels')
    print((np.sum(conf_matrix,axis=0)/np.sum(conf_matrix))*100)
    print('Class Specific Accuracy')
    print((np.diag(conf_matrix)/np.sum(conf_matrix,axis=1))*100)

    
    print('Additional Accuracy Metrics:Major Prob')
    print('True distribution of Labels')
    print((np.sum(conf_matrix2,axis=1)/np.sum(conf_matrix2))*100)
    print('Predicted distribution of Labels')
    print((np.sum(conf_matrix2,axis=0)/np.sum(conf_matrix2))*100)
    print('Class Specific Accuracy')
    print((np.diag(conf_matrix2)/np.sum(conf_matrix2,axis=1))*100)

    
    return accuracy_average,accuracy_major,conf_matrix,conf_matrix2,index_labfalse,index_labfalse2,index_false,major_prob,average_prob

files_valid = ['catA_1fps_test','catA_10fps_test','catB_1fps_test','catB_10fps_test','catC_1fps_test','catC_10fps_test']

#Calculating Validation Set Results
for file in files_valid:
    print('Results for model'+file)
    accuracy_average,accuracy_major,conf_matrix,conf_matrix2,index_labfalse,index_labfalse2,index_false,major_prob,average_prob = calculate_results_valid_old('results/valid/'+file+'.csv',true_labels)

def calculate_results_new(file):
    
    '''
    This function calculates evaluation metrics using Approach 2 as
    described in the report.

    Args
    - file: csv file including results and true labels
    
    Output
    - Includes aggregate as well as class-specific accuracies and confusion
    matrices both for the average probability and majority voting approach
    as described in the report 
    '''
    
    
    resA_1fps = pd.read_csv(file)
    resA_1fps = resA_1fps.sort_values(by=['video_id'])
    
    print('Total Number of Video Clips is {}'.format(len(resA_1fps)))
    print("Number of Videos is {}".format(len(resA_1fps.video_id.unique())))
    
    len_vids= len(resA_1fps.video_id.unique())
    
    #Average Probability Approach
    
    average_prob =resA_1fps.groupby('video_id').mean()
    average_prob['max_prob'] = average_prob.loc[:,'p1':'p4'].idxmax(axis=1)
    average_prob['pred_label'] = 0
    
    print("Number of Videos (Average) is {}".format(len(average_prob)))
    
    for i in range(len(average_prob)):
        if average_prob['max_prob'].iloc[i] == 'p1':
            average_prob['pred_label'].iloc[i]= 1
        elif average_prob['max_prob'].iloc[i] == 'p2':
            average_prob['pred_label'].iloc[i]= 2
        elif average_prob['max_prob'].iloc[i] == 'p3':
            average_prob['pred_label'].iloc[i]= 3
        elif average_prob['max_prob'].iloc[i] == 'p4':
            average_prob['pred_label'].iloc[i]= 4
            
    #Majority Probability Approach
    
    resA_1fps['max_prob'] = resA_1fps.loc[:,'p1':'p4'].idxmax(axis=1)
    
    resA_1fps['most_p1'] = (resA_1fps['max_prob']=='p1')
    resA_1fps['most_p2'] = (resA_1fps['max_prob']=='p2')
    resA_1fps['most_p3'] = (resA_1fps['max_prob']=='p3')
    resA_1fps['most_p4'] = (resA_1fps['max_prob']=='p4')
    
    
    major_prob = resA_1fps.groupby('video_id').sum()
    major_prob['max_prob'] = major_prob.loc[:,'most_p1':'most_p4'].idxmax(axis=1)
    
    major_prob['pred_label'] = 0
    
    for i in range(len(average_prob)):
        if major_prob['max_prob'].iloc[i] == 'most_p1':
            major_prob['pred_label'].iloc[i]= 1
        elif major_prob['max_prob'].iloc[i] == 'most_p2':
            major_prob['pred_label'].iloc[i]= 2
        elif major_prob['max_prob'].iloc[i] == 'most_p3':
            major_prob['pred_label'].iloc[i]= 3
        elif major_prob['max_prob'].iloc[i] == 'most_p4':
            major_prob['pred_label'].iloc[i]= 4
            
    def_dif = np.sum(major_prob['pred_label']==average_prob['pred_label'])/len_vids
    print('Definitions coincide in {} percent of cases'.format(def_dif*100))
    
    equiv_pred = major_prob['pred_label']==average_prob['pred_label']
    index_false = equiv_pred[equiv_pred==False].index
    
    #True Labels
    average_prob['true_lab2'] = 5
    for i in range(len(average_prob)):
        if average_prob['labels'].iloc[i] == 0:
            average_prob['true_lab2'].iloc[i]= 1
        elif average_prob['labels'].iloc[i] == 1:
            average_prob['true_lab2'].iloc[i]= 2
        elif average_prob['labels'].iloc[i] == 2:
            average_prob['true_lab2'].iloc[i]= 3
        elif average_prob['labels'].iloc[i] == 3:
            average_prob['true_lab2'].iloc[i]= 4
            
        
        
    major_prob['true_lab2'] = 5
    for i in range(len(major_prob)):
        if major_prob['labels'].iloc[i] == 0:
            major_prob['true_lab2'].iloc[i]= 1
        elif major_prob['labels'].iloc[i] == 20:
            major_prob['true_lab2'].iloc[i]= 2
        elif major_prob['labels'].iloc[i] == 100:
            major_prob['true_lab2'].iloc[i]= 3
        elif major_prob['labels'].iloc[i] == 300:
            major_prob['true_lab2'].iloc[i]= 4
    
    
    #Accuracies
    accuracy_average = np.sum(average_prob['true_lab2'] == average_prob['pred_label'])/len(average_prob)
    accuracy_major = np.sum(major_prob['true_lab2'] == major_prob['pred_label'])/len(major_prob)

    print('Accuracy for the average probability approach is {} percent'.format(accuracy_average*100))
    print('Accuracy for the majority probability approach is {} percent'.format(accuracy_major*100))

    
    
    equiv_lab = average_prob['true_lab2'] == average_prob['pred_label']
    index_labfalse = equiv_lab[equiv_lab==False].index
    
    equiv_lab2 = major_prob['true_lab2'] == major_prob['pred_label']
    index_labfalse2 = equiv_lab2[equiv_lab2==False].index
    
    #Confusion Matrix
    expected = list(average_prob['true_lab2'])
    predicted = list(average_prob['pred_label'])
    conf_matrix = confusion_matrix(expected, predicted)
    
    expected2 = list(major_prob['true_lab2'])
    predicted2 = list(major_prob['pred_label'])
    conf_matrix2 = confusion_matrix(expected2, predicted2)
    
    print('The confusion matrix for the average approach is')
    print(conf_matrix)
    
    print('The confusion matrix for the majority approach is')
    print(conf_matrix2)
    
    print('Additional Accuracy Metrics:Average Prob')
    print('True distribution of Labels')
    print((np.sum(conf_matrix,axis=1)/np.sum(conf_matrix))*100)
    print('Predicted distribution of Labels')
    print((np.sum(conf_matrix,axis=0)/np.sum(conf_matrix))*100)
    print('Class Specific Accuracy')
    print((np.diag(conf_matrix)/np.sum(conf_matrix,axis=1))*100)

    
    print('Additional Accuracy Metrics:Major Prob')
    print('True distribution of Labels')
    print((np.sum(conf_matrix2,axis=1)/np.sum(conf_matrix2))*100)
    print('Predicted distribution of Labels')
    print((np.sum(conf_matrix2,axis=0)/np.sum(conf_matrix2))*100)
    print('Class Specific Accuracy')
    print((np.diag(conf_matrix2)/np.sum(conf_matrix2,axis=1))*100)
    
    return accuracy_average,accuracy_major,conf_matrix,conf_matrix2,index_labfalse,index_labfalse2,index_false,major_prob,average_prob

files_train_new = ['train_epoch_6_res_total','train_epoch_9_res_total']

#Calculating Train Set Results using Approach 2
for file in files_train_new:
    print('Results for model'+file)
    accuracy_average,accuracy_major,conf_matrix,conf_matrix2,index_labfalse,index_labfalse2,index_false,major_prob,average_prob = calculate_results_new('results/train/'+file+'.csv')

files_valid_new = ['test_res_tot_6','test_res_tot_9','test_']

#Calculating Validation Set Results using Approach 2
for file in files_valid_new:
    print('Results for model'+file)
    accuracy_average,accuracy_major,conf_matrix,conf_matrix2,index_labfalse,index_labfalse2,index_false,major_prob,average_prob = calculate_results_new('results/valid/'+file+'.csv')