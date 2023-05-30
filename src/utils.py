#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy
import torch
from torchvision import datasets, transforms
from dataset import MyDataset, getDataClient, getDataClientSubset, getDataClient_HAM10000, getDataClientSubset_HAM10000
from dataset import getDataClient_Aptos, getDataClientSubset_Aptos
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid
from sampling import ham10000_iid, ham10000_noniid, ham10000_noniid_unequal
from sampling import aptos_iid, aptos_noniid, aptos_noniid_unequal

def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'mnist':

        if(args.subset == True):
            train_client1_raw, train_client2_raw , test_client1_raw,test_client2_raw = getDataClientSubset(0.5)

        else:
            train_client1_raw, train_client2_raw , test_client1_raw,test_client2_raw = getDataClient()

        train_client1_dataset = MyDataset(train_client1_raw)
        train_client2_dataset = MyDataset(train_client2_raw)
        test_client1_dataset = MyDataset(test_client1_raw)
        test_client2_dataset = MyDataset(test_client2_raw)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_client1_dataset,train_client2_dataset)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user TODO: to be changed later
                user_groups = mnist_noniid_unequal(train_client2_dataset, args.num_users)
            else:
                # Chose euqal splits for every user TODO: to be changed later
                user_groups = mnist_noniid(train_client2_dataset, args.num_users)

    elif args.dataset == 'ham10000':

        if(args.subset == True):
            train_client1_raw, train_client2_raw , test_client1_raw,test_client2_raw = getDataClientSubset_HAM10000(0.5)

        else:
            train_client1_raw, train_client2_raw , test_client1_raw,test_client2_raw = getDataClient_HAM10000()

        train_client1_dataset = MyDataset(train_client1_raw)
        train_client2_dataset = MyDataset(train_client2_raw)
        test_client1_dataset = MyDataset(test_client1_raw)
        test_client2_dataset = MyDataset(test_client2_raw)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = ham10000_iid(train_client1_dataset,train_client2_dataset)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user TODO: to be changed later
                user_groups = ham10000_noniid_unequal(train_client2_dataset, args.num_users)
            else:
                # Chose euqal splits for every user TODO: to be changed later
                user_groups = ham10000_noniid(train_client2_dataset, args.num_users)

    elif args.dataset == 'aptos':
    
        if(args.subset == True):
            train_client1_raw, train_client2_raw , test_client1_raw,test_client2_raw = getDataClientSubset_Aptos(0.5)

        else:
            train_client1_raw, train_client2_raw , test_client1_raw,test_client2_raw = getDataClient_Aptos()

        train_client1_dataset = MyDataset(train_client1_raw)
        train_client2_dataset = MyDataset(train_client2_raw)
        test_client1_dataset = MyDataset(test_client1_raw)
        test_client2_dataset = MyDataset(test_client2_raw)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = aptos_iid(train_client1_dataset,train_client2_dataset)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user TODO: to be changed later
                user_groups = aptos_noniid_unequal(train_client2_dataset, args.num_users)
            else:
                # Chose euqal splits for every user TODO: to be changed later
                user_groups = aptos_noniid(train_client2_dataset, args.num_users)

    #return train_dataset_client1, train_dataset_client2, test_dataset_client1,test_dataset_client2, user_groups
    return train_client1_dataset,train_client2_dataset, test_client1_dataset,test_client2_dataset, user_groups

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('   IID')
    else:
        print('    Non-IID')
    print(f'    Number of users  : 2')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
