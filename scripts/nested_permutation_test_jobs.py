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
from scipy.stats import pearsonr
from functools import partial

neural = sys.argv[1]
beh_lab = sys.argv[2]
output_dir = sys.argv[3]

# ---------------
# Helper function
# ---------------

# function for computing cohen's D 
def _cohenD(df, parcel_name):
    dat = list(df[parcel_name])
    d = np.mean(dat)/np.std(dat, ddof = 1)  
    return d
# function for one sample ttest given parcel name
def _oneSample_ttest(df, parcel_name):
    dat = list(df[parcel_name])
    result = [ttest_1samp(dat,0)[0], ttest_1samp(dat,0)[1]]
    return result
# function for choosing parcel based on Cohen's D
def _selectParcel(parcel_es_df, parcel_type, es_min, es_max):
    df = parcel_es_df[parcel_es_df['parcel_name'].isin(parcel_type)].reset_index(drop=True)
    bin_parcel = df['parcel_name'][np.where((df['es'] >= es_min) & (df['es'] < es_max))[0]]
    return list(bin_parcel)

# randomly sample 10 parcels from the given bin list for N iterations. Getting N sampling data
def _sample_from_bin(bin_list, iteration):
    
    """
    Function to do random sampling from each bin for N iteration 
    
    Parameters:
    bin_list: a list of list. Each sub list is a effect size bin, containing the parcel names (i.e., string) of parcels in this bin
    iteration: int, how many samples to draw
    
    Retn:
    iteration_sampling: nested list. A list of all samplings. Each sampling is a list of 7 bins. Each bin is a list of 10 parcels. 
    
    """
    
    seed_list = random.sample(range(1, 10000), iteration) # get seed
    if len(set(seed_list)) != iteration: # make sure seeds are unique
        print("seed duplication")  
    else:
        iteration_sampling = []
        for seed in seed_list: # for each given seed
            random.seed(seed)
            cur_iteration = [random.sample(cur_bin,10) for cur_bin in bin_list]
            iteration_sampling.append(cur_iteration)
    
    return(iteration_sampling)

# 1000 iteration permutation shuffle for each sampling data
def _shuffle_iteration_sampling(iteration_sampling, iteration):
    
    """
    Function to shuffle a given (list of) sampling N times to create a null distribution for the current sampling
    
    Parameters:
    iteration_sampling: a nested list, output of _sample_from_bin
    iteration: int, how many shuffles to run for each sampling
    
    Retrun:
    null data: nested list. A list of all sampling. Each sampling is a list of N shuffles. Each shuffle is a list of 7 bins. 
            Each bin is a list of 10 parcels. This is the null data for each sampling. 
    
    """
    
    seed_list = random.sample(range(1, 10000), iteration) # get seed
    if len(set(seed_list)) != iteration: # make sure seeds are unique
        print("seed duplication")  
    
    iter_num = len(iteration_sampling) # total number of iteration
    bin_num = len(iteration_sampling[0]) # number of bin 
    null_data = []
    for sampling in iteration_sampling:
        cur_sampling = list(chain(*sampling)) # unlist for the current iteartion
        if len(set(cur_sampling)) != bin_num*10:
            print("duplication within a sampling")
        cur_sampling_shuffle = [list(shuffle(np.array(cur_sampling), random_state = seed)) for seed in seed_list] # null distribution for the current data (sampling)
        cur_sampling_shuffle_sublist = [[cur_shuffle[x:x+10] for x in range(0, 10*bin_num, 10)] for cur_shuffle in cur_sampling_shuffle]
        if len(cur_sampling_shuffle_sublist[0]) != bin_num:
            print("bin number does not match")
        null_data.append(cur_sampling_shuffle_sublist)
        
    if len(null_data) != iter_num:
        print("random sampling iteration number does not match")
    if len(null_data[0][0]) != bin_num:
        print("bin number does not match")
    
    return(null_data)
# svr pipeline for each bin 
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

# svr pipeline for each shuffle  
def _svr_per_shuffle(merged_df, lab_type, list_of_parcel_list):
    bin_acc = [_svr_per_bin(merged_df, lab_type,  parcel_list) for parcel_list in list_of_parcel_list]
    return(bin_acc)

# svr pipeline for each sample
def _svr_per_sampling(merged_df, lab_type, list_of_shuffle_list):
    shuffle_acc = [_svr_per_shuffle(merged_df, lab_type, shuffle_list) for shuffle_list in list_of_shuffle_list]
    return(shuffle_acc)

if __name__ == '__main__':
    
    # ---------------
    # Load in data 
    # ---------------

    # read in the csv for neural and behaviral data
    if neural == 'schaefer':
        neural_data = pd.read_csv('/home/peetal/hulacon/nested_permutation/schaefer400_cope11_rm_outlier.csv') # neural
    else: 
        neural_data = pd.read_csv('/home/peetal/hulacon/nested_permutation/gordon333_cope11_rm_outlier.csv')
    full_beh = pd.read_csv('/home/peetal/hulacon/nested_permutation/HCP_behavioral_data.csv') # behavioral
    beh = full_beh[['Subject','WM_Task_2bk_Acc', 'ListSort_AgeAdj',
                    'PMAT24_A_CR', 'PicVocab_AgeAdj', 'ReadEng_AgeAdj']] # ID, in-scanner, and out-scanner task

    # join neural and behaviral data by subject id
    merge_df = pd.merge(neural_data, beh, how='left', on=['Subject']).dropna() # join with ID

    # ---------------
    # get parcel bins 
    # ---------------

    # Effect size (Cohen's D) information
    parcel_es = pd.DataFrame(columns = ['parcel_name','es']) # new df for cohen's D per parcel
    parcel_es['parcel_name'] = list(neural_data.columns[1:])
    parcel_es['es'] = [_cohenD(neural_data, parcel) for parcel in list(neural_data.columns[1:])]

    # parcel type information (i.e., load-activate, load-reverse-activated or load-insensitive)
    ttest_result = np.stack([_oneSample_ttest(neural_data, parcel) for parcel in list(neural_data.columns[1:])], axis = 0) # ttest with bonferroini correction
    load_activated = np.array(list(neural_data.columns[1:]))[np.where((ttest_result[:,0] > 0) & (ttest_result[:,1] < 0.05/len(parcel_es)))] # load_activated
    load_reverse_activated = np.array(list(neural_data.columns[1:]))[np.where((ttest_result[:,0] < 0) & (ttest_result[:,1] < 0.05/len(parcel_es)))] # load-reverse-activated
    load_insensitive = list(set(neural_data.columns[1:]) - set(load_activated) - set(load_reverse_activated)) # load_insensitive full bin
    
    # bin parcels based on their effect size and parcel types
    act_01_03, act_03_05, act_05_07, act_07_09, act_09_11, act_11_13, act_13_15  = _selectParcel(parcel_es, load_activated, 0.1, 0.3), _selectParcel(parcel_es, load_activated, 0.3, 0.5), _selectParcel(parcel_es, load_activated, 0.5, 0.7), _selectParcel(parcel_es, load_activated, 0.7, 0.9), _selectParcel(parcel_es, load_activated, 0.9, 1.1), _selectParcel(parcel_es, load_activated, 1.1, 1.3), _selectParcel(parcel_es, load_activated, 1.3, 1.9)  
    deact_01_03, deact_03_05, deact_05_07, deact_07_09 = _selectParcel(parcel_es, load_reverse_activated, -0.3, -0.1), _selectParcel(parcel_es, load_reverse_activated, -0.5, -0.3), _selectParcel(parcel_es, load_reverse_activated, -0.7, -0.5), _selectParcel(parcel_es, load_reverse_activated, -1.2, -0.7)
    load_insensitive = _selectParcel(parcel_es, load_insensitive, -0.2, 0.2)

    num_check = act_01_03 + act_03_05 + act_05_07 + act_07_09 + act_09_11 + act_11_13 + act_13_15 + deact_01_03 + deact_03_05 + deact_05_07 + deact_07_09 + load_insensitive
    if len(num_check) != len(parcel_es):
        print("something is wrong with bin")
    # ---------------------------
    # Do nested permutation test 
    # ---------------------------
    bin_list = [act_01_03, act_03_05, act_05_07, act_07_09, act_09_11, act_11_13, act_13_15, deact_01_03, deact_03_05, deact_05_07, deact_07_09, load_insensitive]
    bin_name = ["act_01_03", "act_03_05", "act_05_07", "act_07_09", "act_09_11", "act_11_13", "act_13_15", "deact_01_03", "deact_03_05", "deact_05_07", "deact_07_09", "load_insensitive"]

    real_sample = _sample_from_bin(bin_list, 100) # 100 real sample
    null_sample = _shuffle_iteration_sampling(real_sample, 1000) # 1000 shuffled data for each real sample. 

    # enable hyperthreading
    pool = multiprocessing.Pool(processes = 28)
    partial_svr_per_shuffle = partial(_svr_per_shuffle, merge_df, beh_lab)
    partial_svr_per_sampling = partial(_svr_per_sampling, merge_df, beh_lab)

    # real data
    real_pred_acc = pool.map(partial_svr_per_shuffle, real_sample) # real score for each sample
    real_pred_acc_np = np.stack(real_pred_acc, axis = 0) # 2d array (sample_num, bin_num)
    real_pred_acc_mean = np.mean(real_pred_acc_np, axis = 0) # average of all the real samples (the averaged real data, 1 x bin_num)
    real_pred_acc_mean_expand = np.expand_dims(real_pred_acc_mean, 0)

    # null data
    null_bin_pred_acc = pool.map(partial_svr_per_sampling, null_sample) # null score for each sample 
    null_bin_pred_acc_mean = np.mean(np.array(null_bin_pred_acc), axis = 0) # 2d array (shuffle_num, bin_num)

    # real + null
    full_data = np.concatenate([real_pred_acc_mean_expand, null_bin_pred_acc_mean])

    # prepare and save output data: 
    output_df = pd.DataFrame(full_data, columns = bin_name)
    #output_df = pd.DataFrame(real_pred_acc_np, columns = bin_name)
    output_df.to_csv(os.path.join(output_dir, f"{neural}_nested_permutation_test_12bin_{beh_lab}.csv"), index=False)

    
