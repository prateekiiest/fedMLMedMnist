#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from utils import get_dataset
from options import args_parser
from update import test_inference, test_inference_base2
from models import MLP, CNNDermaMnist, CNNMnist, CNNAptos

if __name__ == '__main__':
    args = args_parser()
    #if args.gpu:
     #   torch.cuda.set_device(args.gpu)
    device = 'cpu' if args.gpu else 'cpu'

    target_name_client1 = []
    target_name_client2 = []

    if(args.dataset == 'mnist'):
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
    
    # load datasets
    train_dataset, test_dataset = {}, {}
    train_dataset_client1, train_dataset_client2, test_dataset_client1,test_dataset_client2, user_groups = get_dataset(args)
    # BUILD MODEL
    train_dataset[1] = train_dataset_client1
    train_dataset[2] = train_dataset_client2
    test_dataset[1] = test_dataset_client1
    test_dataset[2] = test_dataset_client2
    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'ham10000':
            global_model = CNNDermaMnist(args=args)
            #TODO: Add support for different datasets
        elif args.dataset == 'aptos':
            global_model = CNNAptos(args=args)
        else:
            raise  Exception("Dataset '%s' is not supported" % args.dataset)
    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # Training
    # Set optimizer and criterion
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr,
                                    momentum=0.5)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr,
                                     weight_decay=1e-4)
        
    for id in range(1,3):
        if(id==1):
            target_names = target_name_client1

            trainloader = DataLoader(train_dataset[id], batch_size=64, shuffle=True)
            criterion = torch.nn.NLLLoss().to(device)
            epoch_loss = []

            for epoch in tqdm(range(args.epochs)):
                batch_loss = []

                for batch_idx, (images, labels) in enumerate(trainloader):
                    images, labels = images.to(device), labels.to(device)

                    optimizer.zero_grad()
                    outputs = global_model(images)
                    labels = labels.long()
                    labels[labels==2]= 1
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    if batch_idx % 50 == 0:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch+1, batch_idx * len(images), len(trainloader.dataset),
                            100. * batch_idx / len(trainloader), loss.item()))
                        
                        _, pred_labels = torch.max(outputs, 1)
                        pred_labels = pred_labels.view(-1)
                        y_pred =  pred_labels
                        y_true = labels
                        y_pred_d = y_pred.detach().numpy()
                        y_true_d = y_true.detach().numpy()
                        rep = classification_report(y_pred_d, y_true_d, target_names=target_names)
                        print(rep)
                    batch_loss.append(loss.item())

                loss_avg = sum(batch_loss)/len(batch_loss)
                print('\nTrain loss:', loss_avg)
                epoch_loss.append(loss_avg)
        else:
            target_names = target_name_client2
            trainloader = DataLoader(train_dataset[id], batch_size=64, shuffle=True)
            criterion = torch.nn.NLLLoss().to(device)
            epoch_loss = []

            for epoch in tqdm(range(args.epochs)):
                batch_loss = []

                for batch_idx, (images, labels) in enumerate(trainloader):
                    images, labels = images.to(device), labels.to(device)

                    optimizer.zero_grad()
                    outputs = global_model(images)
                    labels = labels.long()
                    labels[labels==1]= 0
                    labels[labels==3]= 1
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    if batch_idx % 50 == 0:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch+1, batch_idx * len(images), len(trainloader.dataset),
                            100. * batch_idx / len(trainloader), loss.item()))
                        
                        _, pred_labels = torch.max(outputs, 1)
                        pred_labels = pred_labels.view(-1)
                        y_pred =  pred_labels
                        y_true = labels
                        y_pred_d = y_pred.detach().numpy()
                        y_true_d = y_true.detach().numpy()
                        rep = classification_report(y_pred_d, y_true_d, target_names=target_names)
                        print(rep)
                    batch_loss.append(loss.item())

                loss_avg = sum(batch_loss)/len(batch_loss)
                print('\nTrain loss:', loss_avg)
                epoch_loss.append(loss_avg)


         # testing
    test_acc1, test_acc1_on_other, test_acc2,test_acc2_on_other = test_inference_base2(args, global_model, test_dataset[1], test_dataset[2])
    
    print("========================================\n")
    print("========================================\n")

    print('Test on original client distribution for client 1 : {:.2f}%'.format(100*test_acc1))
    print('Test on client 2 distribution for client 1 : {:.2f}%'.format(100*test_acc1_on_other))
    print('Test on original client distribution for client 2 : {:.2f}%'.format(100*test_acc2))
    print('Test on client 1 distribution for client 2 : {:.2f}%'.format(100*test_acc2_on_other))



"""

    # Plot loss
    plt.figure()
    plt.plot(range(len(epoch_loss)), epoch_loss)
    plt.xlabel('epochs')
    plt.ylabel('Train loss')
    plt.savefig('../save/nn_{}_{}_{}.png'.format(args.dataset, args.model,
                                                 args.epochs))
"""

   