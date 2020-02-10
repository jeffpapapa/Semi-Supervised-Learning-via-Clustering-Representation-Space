# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 13:32:09 2020

@author: Yen-Chieh, Huang

training and testing for MCMC

"""

from random import shuffle
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable

from preprocessing import MNIST_preprocessing, random_split_label_data

training_data, training_true_label, testing_data, testing_true_label = MNIST_preprocessing()

from network import Net
from loss_functions import DBI, margin

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train_and_test(k_val, total_label_num, cluster_dim, k, batch, label_batch_size, test_batch, epoch, lr):
      
    k_final_testing_acc = []
    k_final_unlabel_acc = []
    
    for val_times_k in range(k_val):
        print('Start running ', val_times_k, ' round...')
        label_data, label_label, unlabel_data, unlabel_label = random_split_label_data(total_label_num, training_data, training_true_label)
        # input_dim = 2
        output_dim = 10
        #cluster_dim = 300
        #k = 10
        
        #batch = 200
        # label_batch_size = int(len(label_data)/label_batch)
        #label_batch_size = 50
        total_batch_size = int(len(training_data)/batch)
        train_total = len(training_data)
        
        #test_batch = 100
        test_batch_size = int(len(testing_data)/test_batch)
        test_total = len(testing_data)
        
        ### check gpu
        cuda_gpu = torch.cuda.is_available()   #判斷GPU是否存在可用
        
        if(cuda_gpu):
            print('gpu')
        else:
            print('cpu')
        
        
        loss_record = []
        #label_loss_record = []
        #total_train_loss_record = []
        #margin_loss_record = []
        accuracy_record = []
        unlabel_accuracy_record = []
        # wrap up with Variable in pytorch
        if(cuda_gpu):
            train_X = (torch.from_numpy(label_data).type(torch.LongTensor).view(-1, 1,28,28).float().cuda())
            test_X = (torch.from_numpy(testing_data).type(torch.LongTensor).view(-1, 1,28,28).float().cuda())
            train_y = (torch.Tensor(label_label).long().cuda())
            test_y = (torch.Tensor(testing_true_label).long().cuda())
        
        #     unlabel_X = Variable(torch.from_numpy(unlabel_data).type(torch.LongTensor).view(-1, 1,28,28).float().cuda())
            #unlabel_y = Variable(torch.Tensor(unlabel_label).long().cuda())
            unlabel_y = (torch.Tensor(training_true_label).long().cuda())
        #     unlabel_X = Variable(torch.Tensor(unlabel_unb_data).float().cuda())
        #     unlabel_y = Variable(torch.Tensor(unlabel_unb_label).float().cuda())
        
            total_train = torch.from_numpy(training_data).type(torch.LongTensor).view(-1, 1,28,28).float().cuda()
            #total_train = Variable(torch.from_numpy(unlabel_data).type(torch.LongTensor).view(-1, 1,28,28).float().cuda(), requires_grad=True)
            #     total_train = Variable(torch.Tensor(new_total_data).float().cuda(), requires_grad=True)
        else:
            train_X = (torch.from_numpy(label_data).type(torch.LongTensor).view(-1, 1,28,28).float())
            test_X = (torch.from_numpy(testing_data).type(torch.LongTensor).view(-1, 1,28,28).float())
            train_y = (torch.Tensor(label_label).long())
            test_y = (torch.Tensor(testing_true_label).long())
        
        #     unlabel_X = Variable(torch.from_numpy(unlabel_data).type(torch.LongTensor).view(-1, 1,28,28).float())
            #unlabel_y = Variable(torch.Tensor(unlabel_label).long())
            unlabel_y = torch.Tensor(training_true_label).long()
        #     unlabel_X = Variable(torch.Tensor(unlabel_unb_data).float())
        #     unlabel_y = Variable(torch.Tensor(unlabel_unb_label).float())
        
        
            total_train = torch.from_numpy(training_data).type(torch.LongTensor).view(-1, 1,28,28).float()
            #total_train = Variable(torch.from_numpy(unlabel_data).type(torch.LongTensor).view(-1, 1,28,28).float(), requires_grad=True)
        #     total_train = Variable(torch.Tensor(new_total_data).float(), requires_grad=True)
        
        net = Net(output_dim, cluster_dim)
        if(cuda_gpu):
            net = torch.nn.DataParallel(net).cuda()  
        
        criterion = nn.CrossEntropyLoss()# cross entropy loss
        

        learning_rate = lr
        learning_rate_min = 0.00001
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.99))
        #optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.25)
        
        first_back = 1
        
        loss_weight = 1
        DBI_label_weight = 1
        DBI_all_weight = 1
        margin_weight = 2
        
        label_batch_mode = True
        label_train_idx = list(range(len(train_X)))
        total_train_idx = list(range(len(total_train)))
        
        total_epoch = epoch
        
        change_training_mode_epoch = -1
        for epoch in range(total_epoch):
            ## shuffle data
        
        #     if(label_batch_mode==True):
            train_X_shuffle = train_X[label_train_idx]    
            train_y_shuffle = train_y[label_train_idx]
        
            shuffle(total_train_idx)
            total_train_shuffle = total_train[total_train_idx]
            unlabel_y_shuffle = unlabel_y[total_train_idx]
            ### minibatch
        
            if(epoch>=change_training_mode_epoch):
                decay_rate = 0.5
        
                ### exponential decrease
                if(epoch%1==0 and epoch>0):
                    learning_rate = learning_rate*(decay_rate**(epoch/50))
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = learning_rate
        
                    learning_rate = max(learning_rate, learning_rate_min)
                    

                for batch_idx in tqdm(range(batch)):
                    if(batch_idx*total_batch_size >= len(total_train_shuffle)):
                        pass
                    else:
        
                        optimizer.zero_grad()
        
                    #         optimizer.zero_grad()
                        #out, hidden = net(train_X)
                        if(label_batch_mode==True):
                            shuffle(label_train_idx)
                            train_X_shuffle = train_X[label_train_idx]    
                            train_y_shuffle = train_y[label_train_idx]  
                        if(label_batch_mode==True):
                            train_input = Variable(train_X_shuffle[:label_batch_size])
                        else:
                            train_input = Variable(train_X_shuffle)
                        out, z = net(train_input)
                        if(label_batch_mode==True):
                            truth = train_y_shuffle[:label_batch_size]
                        else:
                            truth = train_y_shuffle
                        loss = criterion(out, truth)
        
                        total_input = Variable(total_train_shuffle[batch_idx*total_batch_size : (batch_idx+1)*total_batch_size])
                        predict_total_out, total_z = net(total_input)
                        _, predict_total_y = torch.max(predict_total_out, 1)
        
                        label_DB = DBI(output_dim, cluster_dim, cuda_gpu)
                        total_DB = DBI(output_dim, cluster_dim, cuda_gpu)
                        ML = margin(output_dim, k, cuda_gpu)
        
        
                        label_DB_loss = label_DB(z, truth).cuda()
        
                        total_DB_loss = total_DB(total_z, predict_total_y).cuda()
        
                        margin_loss = ML(total_z, predict_total_y)
                        del label_DB, total_DB, ML, truth, train_input, total_input
        
                        final_loss = (loss_weight*loss + DBI_label_weight*label_DB_loss + DBI_all_weight*total_DB_loss + margin_weight*margin_loss)

                        if(first_back==1):
                            final_loss.backward(retain_graph=True)
                            optimizer.step()
                        else:
                            final_loss.backward()
                            optimizer.step()
        
                    if(batch_idx == batch-1):
                        #print('number of epoch',epoch,'loss',loss.data.item())
                        print('number of epoch', epoch, 'loss', loss.data.item(), ' / ', label_DB_loss.data.item(), 
                                  ' / ', total_DB_loss.data.item(),  ' / ', margin_loss.data.item())
                        #print('number of epoch', epoch, 'loss', loss.data.item(),
                        #      ' / ', total_DB_loss.data.item(),  ' / ', margin_loss.data.item())
        
                        loss_record.append(loss.data.item())
                        #label_loss_record.append(label_DB_loss.data.item())
                        #total_train_loss_record.append(total_DB_loss.data.item())
                        #margin_loss_record.append(margin_loss.data.item())
        
        #             del out, z, predict_total_out, total_z, final_loss, loss, label_DB_loss, total_DB_loss, margin_loss
    
            del train_X_shuffle, train_y_shuffle
            if epoch % 1 == 0:
                net.eval()
                #push_weight = push_weight/change_rate
        
        #         print('number of epoch', epoch, 'loss', loss.data.item(), ' / ', label_DB_loss.data.item(), 
        #               ' / ', total_DB_loss.data.item(),  ' / ', margin_loss.data.item())
        #         loss_record.append(loss.data.item())
        #         label_loss_record.append(label_DB_loss.data.item())
        #         total_train_loss_record.append(total_DB_loss.data.item())
        #         margin_loss_record.append(margin_loss.data.item())
                print('learning rate : ', learning_rate)
                correct = 0
                test_count = 0
                for test_batch_idx in range(test_batch):
                    if(len(test_X[test_batch_idx*test_batch_size : (test_batch_idx+1)*test_batch_size])==0):
                        break
                    predict_out, z = net(test_X[test_batch_idx*test_batch_size : (test_batch_idx+1)*test_batch_size])
                    #predict_out, hid = net(test_X)
                    _, predict_y = torch.max(predict_out, 1)
        
                    true_y = test_y[test_batch_idx*test_batch_size : (test_batch_idx+1)*test_batch_size].cpu().numpy()
        
                    if(cuda_gpu):
                        predict_y = predict_y.cpu().numpy()
                    else:
                        predict_y = predict_y.data
                    correct += (predict_y==true_y).sum().item()
                    test_count+=len(predict_y)
                    del predict_out, z, predict_y, true_y
                #print(count)
                if((test_batch_idx+1)*test_batch_size < test_total):
                    
                    predict_out, z = net(test_X[(test_batch_idx+1)*test_batch_size :])
                    #predict_out, hid = net(test_X)
                    _, predict_y = torch.max(predict_out, 1)
        
                    true_y = test_y[(test_batch_idx+1)*test_batch_size :].cpu().data
        
                    if(cuda_gpu):
                        predict_y = predict_y.cpu().numpy()
                    else:
                        predict_y = predict_y.numpy()
                    correct += (predict_y==true_y).sum().item()
                    test_count+=len(predict_y)
                    del predict_out, z, predict_y, true_y
                print('testing count : ', test_count)
                acc = correct/test_total
                print('prediction accuracy', acc)
                accuracy_record.append(acc)    
        
                ### compute unlabeled data acc
                correct = 0
                check_total = 0
                for test_batch_idx in range(batch):
                    total_input = total_train_shuffle[test_batch_idx*total_batch_size : (test_batch_idx+1)*total_batch_size]
                    if(len(total_input)==0):
                        del total_input
                        break
                    predict_out, z = net(total_input)
                    _, predict_y = torch.max(predict_out, 1)
        
                    true_y = unlabel_y_shuffle[test_batch_idx*total_batch_size : (test_batch_idx+1)*total_batch_size].cpu().numpy()
        
                    if(cuda_gpu):
                        predict_y = predict_y.cpu().numpy()
                    else:
                        predict_y = predict_y.numpy()
                    correct += (predict_y==true_y).sum().item()
                    check_total+=len(predict_y)
                    del predict_out, z, predict_y, true_y, total_input
                #print(check_total)
                if((test_batch_idx+1)*total_batch_size < train_total):
        
                    total_input = total_train_shuffle[(test_batch_idx+1)*total_batch_size : ]
                    if(len(total_input)==0):
                        del total_input
                    else:
                        predict_out, z = net(total_input)
                        _, predict_y = torch.max(predict_out, 1)
        
                        true_y = unlabel_y_shuffle[(test_batch_idx+1)*total_batch_size : ].cpu().numpy()
        
                        if(cuda_gpu):
                            predict_y = predict_y.cpu().numpy()
                        else:
                            predict_y = predict_y.numpy()
                        correct += (predict_y==true_y).sum().item()
                        check_total+=len(predict_y)
                        del predict_out, z, predict_y, true_y, total_input
                acc = correct/train_total
                print('check total : ', check_total)
                print('unlabeled prediction accuracy', acc)
                unlabel_accuracy_record.append(acc) 
        
        
                del total_train_shuffle
                net.train()
        k_final_testing_acc.append(accuracy_record)
        k_final_unlabel_acc.append(unlabel_accuracy_record)
        print('final testing acc : ', accuracy_record[-1])
        print('final unlabeled acc : ', unlabel_accuracy_record[-1])
        print('Finish running ', val_times_k, ' round')

    return k_final_testing_acc, k_final_unlabel_acc
