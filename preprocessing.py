# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 12:51:49 2020

@author: Yen-Chieh, Huang

Preprocessing for MNIST data
"""

import pandas as pd
import random

def MNIST_preprocessing():
    ### read MNIST training data
    train_df = pd.read_csv('mnist-in-csv/mnist_train.csv')
    training_data = train_df.drop(['label'],1).values
    training_true_label = train_df['label'].values
    
    ### read testing data
    test_df = pd.read_csv('mnist-in-csv/mnist_test.csv')
    testing_data = test_df.drop(['label'],1).values
    testing_true_label = test_df['label'].values
    
    ### Normalize
    ### Normalize
    mean = 0.1307
    var = 0.3081
    training_data = (training_data - mean)/var
    testing_data = (testing_data - mean)/var
    
    
    return training_data, training_true_label, testing_data, testing_true_label


def random_split_label_data(label_num,training_data, training_true_label):
    
    total_label_num = label_num
    single_label_num = int(total_label_num/10)
    label_data_idx = []
    unlabel_data_idx = []

    for num in range(0,10):
        num_idx = [i for i,v in enumerate(training_true_label) if v==num]
        random.shuffle(num_idx)
        label_data_idx = label_data_idx + num_idx[:single_label_num]
        unlabel_data_idx = unlabel_data_idx + num_idx[single_label_num:]

    random.shuffle(label_data_idx)
    random.shuffle(unlabel_data_idx)

    label_data = training_data[label_data_idx]
    label_label = training_true_label[label_data_idx]
    unlabel_data = training_data[unlabel_data_idx]
    unlabel_label = training_true_label[unlabel_data_idx]

    label_label = [i for i in label_label]
    unlabel_label = [i for i in unlabel_label]
    
    return label_data, label_label, unlabel_data, unlabel_label
    
    
    