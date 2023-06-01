#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
from sklearn.metrics import classification_report
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, userId, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.device = 'mps' if args.gpu else 'cpu'
        self.userId = userId
        # Default criterion set to NLL loss function
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10), shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                labels = labels.long()
                if(self.userId == 1):
                    labels[labels==2]= 1
                else:
                    labels[labels==1]= 0
                    labels[labels==3]= 1
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, userId, model):
        """ Returns the inference accuracy and loss.
        """

        device = 'mps' 


        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        target_names = []
        if(userId==2):
            # target_names = ["Negative Lung Op","Positive Lung Op"]
            # target_names = ["Normal", "Severe"]
            target_names = ["NORMAL", "DRUSEN"]
        else:
            # target_names= ["Negative Corona","Positive Corona"]
            # target_names = ["Normal", "Mild"]
            target_names = ["NORMAL", "DME"]


        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            labels= labels.long()
            if(3 in labels):

                labels[labels==1]= 0
                labels[labels==3]= 1
                
            elif(2 in labels):

                labels[labels==2]= 1

            
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            y_pred =  pred_labels
            y_true = labels
            y_pred_d = y_pred.cpu().detach().numpy()
            y_true_d = y_true.cpu().detach().numpy()
            rep = classification_report(y_pred_d, y_true_d, target_names=target_names)
            print("----------------------\n")
            print(rep)
            print("----------------------\n")
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss


def test_inference(args, model, test_dataset1, test_dataset2):
    """ Returns the test accuracy and loss.
    """

    model.eval()

    device = 'mps' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    acc1, acc_on_other1 = inf_test(model, test_dataset1, test_dataset2, device, criterion, "client1")

    model.eval()

    device = 'mps' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    acc2, acc_on_other2 = inf_test( model, test_dataset1, test_dataset2,device, criterion, "client2")

    print("Test accuracy on client 1", acc1)
    print("Test accuracy on client 2", acc2)
    
    accuracy = max(acc1, acc2)
    return acc1, acc_on_other1,acc2, acc_on_other2


def inf_test(model, test_dataset1, test_dataset2, device, criterion, clientId):
    if(clientId=="client1"):

        print("########################\n")

        print("Client 1 Test Statistics\n")

        # target_names = ["Negative Corona","Positive Corona"]
        # target_names = ["Normal", "Mild"]
        target_names = ["NORMAL", "DME"]

        print("==========================\n")
        print("For client 1 original classes : ", target_names)

        loss, total, correct = 0.0, 0.0, 0.0

        testloader = DataLoader(test_dataset1, batch_size=128,
                                shuffle=False)

        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            # Inference
            outputs = model(images)
            labels=labels.long()
            labels[labels==2]= 1
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            y_pred =  pred_labels
            y_true = labels
            y_pred_d = y_pred.cpu().detach().numpy()
            y_true_d = y_true.cpu().detach().numpy()
            rep = classification_report(y_pred_d, y_true_d, target_names=target_names)
            print("----------------------\n")
            print(rep)
            print("----------------------\n")
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
        accuracy_client1_original = correct/total




        # target_names = ["Negative Lung Op","Positive Lung Op"]
        # target_names = ["Normal", "Severe"]
        target_names = ["NORMAL", "DRUSEN"]

        print("==========================\n")
        print("Testing client 1 on client 2 original classes : ", target_names)
        
        loss, total, correct = 0.0, 0.0, 0.0

        testloader = DataLoader(test_dataset2, batch_size=128,
                                shuffle=False)

        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            # Inference
            outputs = model(images)
            labels=labels.long()
            labels[labels==1]= 0
            labels[labels==3]= 1           
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            y_pred =  pred_labels
            y_true = labels
            y_pred_d = y_pred.cpu().detach().numpy()
            y_true_d = y_true.cpu().detach().numpy()
            rep = classification_report(y_pred_d, y_true_d, target_names=target_names)
            print("----------------------\n")
            print(rep)
            print("----------------------\n")
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
        accuracy_client1_onDist2 = correct/total

        return accuracy_client1_original, accuracy_client1_onDist2






    else:

        print("########################\n")
        
        print("Client 2 Test Statistics\n")

        # target_names = ["Negative Lung Op","Positive Lung Op"]
        # target_names = ["Normal", "Severe"]
        target_names = ["NORMAL", "DRUSEN"]

        print("==========================\n")
        print("For client 2 original classes : ", target_names)
  

        loss, total, correct = 0.0, 0.0, 0.0

        testloader = DataLoader(test_dataset2, batch_size=128,
                                shuffle=False)

        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            # Inference
            outputs = model(images)
            labels=labels.long()
            labels[labels==1]= 0
            labels[labels==3]= 1
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            y_pred =  pred_labels
            y_true = labels
            y_pred_d = y_pred.cpu().detach().numpy()
            y_true_d = y_true.cpu().detach().numpy()
            rep = classification_report(y_pred_d, y_true_d, target_names=target_names)
            print("----------------------\n")
            print(rep)
            print("----------------------\n")
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
        accuracy_client2_original = correct/total
    
    
    
    
        # target_names = ["Negative Corona","Positive Corona"]
        # target_names = ["Normal", "Mild"]
        target_names = ["NORMAL", "DME"]

        print("==========================\n")
        print("Testing client 2 on client 1 original classes : ", target_names)
        
        loss, total, correct = 0.0, 0.0, 0.0

        testloader = DataLoader(test_dataset1, batch_size=128,
                                shuffle=False)

        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            # Inference
            outputs = model(images)
            labels=labels.long()
            labels[labels==2]= 1
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            y_pred =  pred_labels
            y_true = labels
            y_pred_d = y_pred.cpu().detach().numpy()
            y_true_d = y_true.cpu().detach().numpy()
            rep = classification_report(y_pred_d, y_true_d, target_names=target_names)
            print("----------------------\n")
            print(rep)
            print("----------------------\n")
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
        accuracy_client2_onDist1 = correct/total

        return accuracy_client2_original, accuracy_client2_onDist1


def inf_test_base2(model, test_dataset1, test_dataset2, device, criterion, clientId):
    if(clientId=="client1"):
        
        print("########################\n")

        print("Client 1 Test Statistics\n")

        # target_names = ["Negative Corona","Positive Corona"]
        # target_names = ["Normal", "Mild"]
        target_names = ["NORMAL", "DME"]

        print("==========================\n")
        print("For client 1 original classes : ", target_names)
        
        loss, total, correct = 0.0, 0.0, 0.0

        testloader = DataLoader(test_dataset1, batch_size=128,
                                shuffle=False)

        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            # Inference
            outputs = model(images)
            labels=labels.long()
            labels[labels==2]= 1
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            y_pred =  pred_labels
            y_true = labels
            y_pred_d = y_pred.detach().numpy()
            y_true_d = y_true.detach().numpy()
            rep = classification_report(y_pred_d, y_true_d, target_names=target_names)
            print("----------------------\n")
            print(rep)
            print("----------------------\n")
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
        accuracy_client1_original = correct/total


        
        
        


        # target_names = ["Negative Lung Op","Positive Lung Op"]
        # target_names = ["Normal", "Severe"]
        target_names = ["NORMAL", "DRUSEN"]

        print("==========================\n")
        print("Testing client 1 on client 2 original classes : ", target_names)
        
        loss, total, correct = 0.0, 0.0, 0.0

        testloader = DataLoader(test_dataset2, batch_size=128,
                                shuffle=False)

        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            # Inference
            outputs = model(images)
            labels=labels.long()
            labels[labels==1]= 0
            labels[labels==3]= 1           
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            y_pred =  pred_labels
            y_true = labels
            y_pred_d = y_pred.detach().numpy()
            y_true_d = y_true.detach().numpy()
            rep = classification_report(y_pred_d, y_true_d, target_names=target_names)
            print("----------------------\n")
            print(rep)
            print("----------------------\n")
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
        accuracy_client1_onDist2 = correct/total

        return accuracy_client1_original, accuracy_client1_onDist2


    else:
                
        print("########################\n")
        
        print("Client 2 Test Statistics\n")

        # target_names = ["Negative Lung Op","Positive Lung Op"]
        # target_names = ["Normal", "Severe"]
        target_names = ["NORMAL", "DRUSEN"]

        print("==========================\n")
        print("For client 2 original classes : ", target_names)
  

        loss, total, correct = 0.0, 0.0, 0.0

        testloader = DataLoader(test_dataset2, batch_size=128,
                                shuffle=False)

        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            # Inference
            outputs = model(images)
            labels=labels.long()
            labels[labels==1]= 0
            labels[labels==3]= 1
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            y_pred =  pred_labels
            y_true = labels
            y_pred_d = y_pred.detach().numpy()
            y_true_d = y_true.detach().numpy()
            rep = classification_report(y_pred_d, y_true_d, target_names=target_names)
            print("----------------------\n")
            print(rep)
            print("----------------------\n")
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
        accuracy_client2_original = correct/total
    
    
    
    
        # target_names = ["Negative Corona","Positive Corona"]
        # target_names = ["Normal", "Mild"]
        target_names = ["NORMAL", "DME"]

        print("==========================\n")
        print("Testing client 2 on client 1 original classes : ", target_names)
        
        loss, total, correct = 0.0, 0.0, 0.0

        testloader = DataLoader(test_dataset1, batch_size=128,
                                shuffle=False)

        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            # Inference
            outputs = model(images)
            labels=labels.long()
            labels[labels==2]= 1
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            y_pred =  pred_labels
            y_true = labels
            y_pred_d = y_pred.detach().numpy()
            y_true_d = y_true.detach().numpy()
            rep = classification_report(y_pred_d, y_true_d, target_names=target_names)
            print("----------------------\n")
            print(rep)
            print("----------------------\n")
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
        accuracy_client2_onDist1 = correct/total

        return accuracy_client2_original, accuracy_client2_onDist1
    
    
    
    
    
    
    
    


def test_inference_base2(args, model, test_dataset1, test_dataset2):
    """ Returns the test accuracy and loss.
    """

    model.eval()

    device = 'cpu' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    acc1, acc_on_other1 = inf_test_base2(model, test_dataset1, test_dataset2, device, criterion, "client1")

    device = 'cpu' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    acc2, acc_on_other2 = inf_test_base2( model,test_dataset1, test_dataset2,device, criterion, "client2")

    accuracy = max(acc1, acc2)
    return acc1, acc_on_other1,acc2, acc_on_other2