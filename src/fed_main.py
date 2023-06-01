#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNDermaMnist, CNNCOVID, CNNAptos
from utils import get_dataset, average_weights, exp_details

if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    #if args.gpu:
     #   torch.mps.set_device(args.gpu)
    device = 'mps' if args.gpu else 'cpu'

    target_name_client1 = []
    target_name_client2 = []

    if(args.dataset == 'COVID'):
        target_name_client1 = ["Negative Corona","Positive Corona"]
        target_name_client2 = ["Negative Lung Op","Positive Lung Op"]

    elif(args.dataset == 'ham10000'):
        target_name_client1 = ["Negative mel", "Positive mel"]
        target_name_client2 = ["Negative nv", "Positive nv"]
    elif(args.dataset == 'aptos'):
        target_name_client1 = ["Normal", "Mild"]
        target_name_client2 = ["Normal", "Severe"]
    else:
        raise  Exception("Dataset '%s' is not supported" % args.dataset)
    
    # load dataset and user groups
    train_dataset, test_dataset ={}, {}

    train_dataset_client1, train_dataset_client2, test_dataset_client1,test_dataset_client2, user_groups = get_dataset(args)
    # BUILD MODEL
    train_dataset[1] = train_dataset_client1
    train_dataset[2] = train_dataset_client2
    test_dataset[1] = test_dataset_client1
    test_dataset[2] = test_dataset_client2

    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'COVID':
            global_model = CNNCOVID(args=args)
        elif args.dataset == 'ham10000':
            global_model = CNNDermaMnist(args=args)
        elif args.dataset == 'aptos':
            global_model = CNNAptos(args=args)
        else: # TODO: Add support for other datasets
            raise NotImplementedError
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)

        idxs_users = np.random.choice(range(1,3),1, replace=False)
        print("----------------")
        
        for idx in idxs_users:
            print("user chosen", idx)
            print("----------------")
            print("----------------")

            local_model = LocalUpdate(args=args,userId = idx, dataset=train_dataset[idx],
                                      idxs=user_groups[idx], logger=logger)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for userId in range(1,3):
            acc, loss = local_model.inference(args, userId = userId, model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))
        
        print("Training accuracy",train_accuracy)
        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

    # Test inference after completion of training
    test_acc1, test_acc1_on_other, test_acc2,test_acc2_on_other = test_inference(args, global_model, test_dataset_client1, test_dataset_client2)
    print("========================================\n")
    print("========================================\n")

    print('Test on original client distribution for client 1 : {:.2f}%'.format(100*test_acc1))
    print('Test on client 2 distribution for client 1 : {:.2f}%'.format(100*test_acc1_on_other))
    print('Test on original client distribution for client 2 : {:.2f}%'.format(100*test_acc2))
    print('Test on client 1 distribution for client 2 : {:.2f}%'.format(100*test_acc2_on_other))

    """
       
    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
    """