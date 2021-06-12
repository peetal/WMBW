# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp
import random, sys, os, multiprocessing
from itertools import chain
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, sem
from functools import partial

# function for computing cohen's D 
def _cohenD(df, parcel_name):
    dat = list(df[parcel_name])
    d = np.mean(dat)/np.std(dat, ddof = 1)  
    return d

# svr pipeline for each sampling bin 
def _svr_per_bin(merged_df, lab_type, parcel_list):
    
    """
    Main function that use SVR to measure a list of parcel's predictive accuracy for the selected beh performance
    
    Parameters:
    merged_df: a pandas dataframe that include neural and behaviral data. Column names include parcel name and beh measure names
    parcel_list: a list of parcel names (i.e., string), that will match the column names in merged_df
    lab_type: a string indicating the type of behaviral measure could be any of the following 5 strings: 
              1) WM_Task_2bk_Acc 2) ListSort_AgeAdj 3) PMAT24_A_CR 4) PicVocab_AgeAdj 5) ReadEng_AgeAdj
    
    Retrun:
    cur_bin_pred_acc: A single number that is the averaged of the 10 fold CV, measuring the predictive acc of the input parcel list. 
    
    """
    
    kf = KFold(n_splits=10, shuffle = False) # 10 fold
    feature = merged_df[parcel_list].to_numpy() # features of the current parcel_list
    label = merged_df[lab_type].to_numpy() # beh score

    pred_acc_list = [] 
    for train_index, test_index in kf.split(merge_df):

        train_feature, train_label = feature[train_index], label[train_index] # training feature and label
        test_feature, test_label = feature[test_index], label[test_index] # testing feature and label

        svr_pipeline = make_pipeline(StandardScaler(), SVR(kernel='linear',degree=1,
                                                       C=1.0,epsilon=0.1)) # svr pipeline
        svr_pipeline.fit(train_feature, train_label)  # train the model 
        pred_test_label = svr_pipeline.predict(test_feature) # test the model 
        pred_acc = pearsonr(test_label, pred_test_label)[0] # check pred acc 
        pred_acc_list.append(pred_acc) 
    
    cur_bin_pred_acc = np.mean(pred_acc_list) # mean of 10 folds
    return(cur_bin_pred_acc)

# randomly sample 10 parcels from the given bin list for N iterations. Getting N sampling data
def _sample_from_bin(parcel_bin, sample_size, iteration):
    
    """
    Function to do random sampling from each bin for N iteration 
    
    Parameters:
    bin_list: a list  containing the parcel names (i.e., string) of parcels in this bin
    iteration: int, how many samples to draw
    
    Retrun:
    iteration_sampling:  A list of all samplings. Each sampling is a list 10 parcels. 
    
    """
    
    seed_list = random.sample(range(1, 10000), iteration) # get seed
    if len(set(seed_list)) != iteration: # make sure seeds are unique
        print("seed duplication")  
    else:
        iteration_sampling = []
        for seed in seed_list: # for each given seed
            random.seed(seed)
            cur_iteration = random.sample(parcel_bin,sample_size)
            iteration_sampling.append(cur_iteration)
    
    return(iteration_sampling)

neural = sys.argv[1]
beh_lab = sys.argv[2]
output_dir = sys.argv[3]

if __name__ == '__main__':

    # read in the csv for neural and behaviral data
    if neural == 'schaefer':
        neural_data = pd.read_csv('/home/peetal/hulacon/nested_permutation/schaefer400_cope11_rm_outlier.csv') # neural
    else: 
        neural_data = pd.read_csv('/home/peetal/hulacon/nested_permutation/gordon333_cope11_rm_outlier.csv')
        
    full_beh = pd.read_csv('/home/peetal/hulacon/nested_permutation/HCP_behavioral_data.csv') # behavioral
    beh = full_beh[['Subject','WM_Task_2bk_Acc', 'ListSort_AgeAdj',
                    'PMAT24_A_CR', 'PicVocab_AgeAdj', 'ReadEng_AgeAdj']] # ID, in-scanner, and out-scanner task

    # Effect size (Cohen's D) information
    parcel_es = pd.DataFrame(columns = ['parcel_name','es']) # new df for cohen's D per parcel
    parcel_es['parcel_name'] = list(neural_data.columns[1:])
    parcel_es['es'] = [_cohenD(neural_data, parcel) for parcel in list(neural_data.columns[1:])]
    parcel_es['abs_es'] = abs(parcel_es['es'])
    parcel_es = parcel_es.sort_values(by=['abs_es'], ascending = False)

    # join neural and behaviral data by subject id
    merge_df = pd.merge(neural_data, beh, how='left', on=['Subject']).dropna() # join with ID

    pool = multiprocessing.Pool(processes = 28)
    svr_per_shuffle_partial = partial(_svr_per_bin, merge_df, beh_lab)

    all_parcel_bin_list = list(parcel_es['parcel_name'])
    sig_lower = []; sig_mean = []; sig_upper = []; real_data = []

    for sample_size in range(1,61):
        null_sampling = _sample_from_bin(all_parcel_bin_list, sample_size, 1000)
        null_pred_acc = pool.map(svr_per_shuffle_partial, null_sampling)
        sig_lower.append(np.mean(null_pred_acc) - 1.65*np.std(null_pred_acc)) # 95%
        sig_mean.append(np.mean(null_pred_acc)) # 50%
        sig_upper.append(np.mean(null_pred_acc) + 1.65*np.std(null_pred_acc)) # 5%

        real_data.append(pool.map(svr_per_shuffle_partial, [all_parcel_bin_list[0:sample_size]]))

    real_data = list(chain(*real_data))
    data={'observed':real_data, 'lower':sig_lower, 'mean':sig_mean, 'upper':sig_upper}
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_dir, f"{neural}_feature_selection_permutation_{beh_lab}.csv"), index=False)

