# Import libraries
import numpy as np
import os
import csv
import math
import torch.nn.parallel
import pickle
import argparse
from tqdm import tqdm
from noise_data_mnist import *
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
torch.autograd.set_detect_anomaly(True)
num_classes = 10
num_epochs = 20
CUDA = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


CE = nn.CrossEntropyLoss().to(device)

opt = parser.parse_args()

        
# Stable version of CE Loss
class CrossEntropyLossStable(nn.Module):
    def __init__(self, reduction='mean', eps=1e-5):
        super(CrossEntropyLossStable, self).__init__()
        self._name = "Stable Cross Entropy Loss"
        self._eps = eps
        self._softmax = nn.Softmax(dim=-1)
        self._nllloss = nn.NLLLoss(reduction=reduction)

    def forward(self, outputs, labels):
        return self._nllloss( torch.log( self._softmax(outputs) + self._eps ), labels )

        
criterion = CrossEntropyLossStable()
criterion.to(device)


# Training
def train(train_loader, peer_loader, model, optimizer, epoch, alpha):

    model.train()
    for i, (idx, input, target) in enumerate(train_loader):
        if idx.size(0) != batch_size:
            continue
        input = torch.autograd.Variable(input.to(device))
        target = torch.autograd.Variable(target.to(device))
        output = model(input)
        optimizer.zero_grad()
        
        # Prepare mixmatched images and labels for the Peer Term
        peer_iter = iter(peer_loader)
        input1 = peer_iter.next()[1]
        output1 = model(input1)
        target2 = peer_iter.next()[2]
        # Peer Loss with Cross-Entropy loss: L(f(x), y) - L(f(x1), y2)
        loss = criterion(output, target) - alpha[epoch] * criterion(output1, target2)
        loss.to(device)
        loss.backward()
        optimizer.step()


# Calculate accuracy
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    for i, (idx, input, target) in enumerate(test_loader):
        input = torch.Tensor(input).to(device)
        target = torch.autograd.Variable(target).to(device)

        total += target.size(0)
        output = model(input)
        _, predicted = torch.max(output.detach(), 1)
        correct += predicted.eq(target).sum().item()
    accuracy = 100. * correct / total

    return accuracy


def main(writer, a_list):
    model_PL = CNNModel().to(device)
    best_val_acc = 0
    train_acc_result = []
    val_acc_noisy_result = []
    test_acc_result = []
    # Dataloader for peer samples, which is used for the estimation of the marginal distribution
    peer_train = peer_data_train(batch_size=args.batchsize, img_size=(28, 28))
    peer_val = peer_data_val(batch_size=args.batchsize, img_size=(28, 28))
    
    # Below we provide two learning rate settings which are used for all experiments in MNIST
    for epoch in range(num_epochs):
        print("epoch=", epoch,'r=', args.r)
        # Setting 1
        learning_rate = 1e-4
        
        # We adopted the ADAM optimizer
        optimizer_PL = torch.optim.Adam(model_PL.parameters(), lr=learning_rate)
        train(train_loader=train_loader_noisy, peer_loader = peer_train, model=model_PL, optimizer=optimizer_PL, epoch=epoch, alpha = a_list)
        print("validating model_PL...")
        
        # Training acc is calculated via noisy training data
        train_acc = test(model=model_PL, test_loader=train_loader_noisy)
        train_acc_result.append(train_acc)
        print('train_acc=', train_acc)
        
        # Validation acc is calculated via noisy validation data
        valid_acc = test(model=model_PL, test_loader=valid_loader_noisy)
        val_acc_noisy_result.append(valid_acc)
        print('valid_acc_noise=', valid_acc)
        
        # Calculate test accuracy
        test_acc = test(model=model_PL, test_loader=test_loader_)
        test_acc_result.append(test_acc)
        print('test_acc=', test_acc)
        
       
        # Best model is selected by referring to the accuracy of validation noisy
        
        if best_val_acc <= valid_acc:
            best_val_acc = valid_acc
            torch.save(model_PL, './trained_models/' + str(args.r) + '_' + str(args.s))
            print("saved, the accuracy of validation noisy increases.")
        
        writer.writerow([epoch, train_acc, valid_acc, test_acc])


def evaluate(path):
    model = torch.load(path)
    test_acc = test(model=model, test_loader=test_loader_)
    print('test_acc=', test_acc)


if __name__ == '__main__':
    # Save statistics
    print("Begin:")
    writer1 = csv.writer(open(f'result_{r}.csv','w'))
    writer1.writerow(['Epoch', 'Training Acc', 'Val_Noisy_Acc', 'Test_ACC'])
    os.makedirs("./trained_models/", exist_ok=True)
    
    # alpha list for the peer term
    alpha_threshold = [0.0, 0.0, 1.0, 2.0, 5.0, 10.0, 20.0]
    milestone = [0, 20, 40, 50, 100, 200, 300]
    alpha_list = []
    for i in range(len(milestone) - 1):
        count = milestone[i]
        a_ratio = (alpha_threshold[i + 1] - alpha_threshold[i]) / (milestone[i + 1] - milestone[i])
        while count < milestone[i + 1]:
            a = alpha_threshold[i] + (count - milestone[i] + 1) * a_ratio
            alpha_list.append(a)
            count += 1
            
    main(writer1, alpha_list)
    evaluate('./trained_models/' + str(args.r) + '_' + str(args.s))
    print("Traning finished")
