# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 14:01:02 2020

@author: Yen-Chieh, Huang

main.py for MNIST dataset

"""
import numpy as np
import argparse
from training_and_testing import train_and_test

def process_command():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment",'-ex', type=int, default=10, help="number of experiments")
    parser.add_argument("--labelnum",'-ln', type=int, default=100, help="number of labeled samples")
    parser.add_argument("--cluster",'-clus', type=int, default=300, help="dimension of cluster embedding latent space")
    parser.add_argument("--kmargin",'-km', type=int, default=10, help="number of k for margin loss")
    parser.add_argument("--batch",'-b', type=int, default=200, help="batch number for training data")
    parser.add_argument("--test_batch",'-tb', type=int, default=100, help="batch number for testing data")
    parser.add_argument("--label_batchsize",'-lb', type=int, default=50, help="batch size for labeled data")
    parser.add_argument("--learning_rate",'-lr', type=int, default=0.001, help="learning rate")
    parser.add_argument("--epoch",'-e', type=int, default=30, help="epoch num for every experiment")
    
    return parser.parse_args()


if __name__ == '__main__':
    args = process_command()
    
    k_final_testing_acc, k_final_unlabel_acc = train_and_test(args.experiment, args.labelnum, args.cluster, args.kmargin, 
                                                              args.batch, args.test_batch, args.label_batchsize, args.epoch, args.learning_rate)
    test_final_acc = [k_final_testing_acc[k][-1] for k in range(len(k_final_testing_acc))]
    print('testing acc results : ', test_final_acc)
    print('mean : ',np.mean(test_final_acc), ' / var : ',np.var(test_final_acc)**0.5)
    