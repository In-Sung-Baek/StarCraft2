# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 10:38:05 2019

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 13:16:50 2019

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 08:31:47 2019

@author: user
"""

import sys
import numpy as np
import pandas as pd

import torch

from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from numpy import load

import os

PROJECT_DIR = os.path.abspath('D:\\1.Project\\2019.04_Game AI\\Data_ConvLSTM\\TvsT_unit\\')

#---------------------------------------- 1.1 5-Dimensions data load ----------------------------------------#

# load npz files(repaly, labels)
replay = np.load(PROJECT_DIR + '\\replay_27.npz')
replay = replay['arr_0']
labels = np.load(PROJECT_DIR + '\\labels_27.npz')
labels = labels['arr_0']

# gpu device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# convert numpy to tensor(example size =  (26, 500, 8, 128, 128))
replay_tensor = torch.FloatTensor(replay).to(device)
labels_tensor = torch.LongTensor(labels).to(device)

#---------------------------------------- 2. CovLSTM Modeling----------------------------------------#
 
# Covolutional Neural Network
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=8, out_channels=12, kernel_size=2, stride=2, padding=0)
        self.relu1 = nn.ReLU()
        
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
     
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=2, stride=2, padding=0)
        self.relu2 = nn.ReLU()
        
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        # Convolution 3
        self.cnn3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=2, stride=2, padding=0)
        self.relu3 = nn.ReLU()
        
        # Max pool 3
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        
        # Convolution 4
        self.cnn4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.relu4 = nn.ReLU()
        
        # Fully connected 1 (shape = 1, 32)
        self.fc1 = nn.Linear(64, 32)
                     
    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x).requires_grad_()
        out = self.relu1(out)
        
        # Max pool 1
        out = self.maxpool1(out)
        
        # Convolution 2 
        out = self.cnn2(out).requires_grad_()
        out = self.relu2(out)
        
        # Max pool 2 
        out = self.maxpool2(out)
        # out = out.view(out.size(0), -1)

        # Convolution 3
        out = self.cnn3(out).requires_grad_()
        out = self.relu3(out)
        # out = out.view(out.size(0), -1)      
        
        # Max pool 3
        out = self.maxpool3(out)
        
        # Convolution 4
        out = self.cnn4(out).requires_grad_()
        out = self.relu4(out)
        out = out.view(out.size(0), -1)
        
        # Linear function (readout)
        out = self.fc1(out).detach()
        
        return out

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your LSTM
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        out = self.fc(out[:, -1, :]) 
        # out.size() --> 100, 10
        return out        

# ConvLSTM Model
class ConvLSTMModel(nn.Module):
    def __init__(self):
        super(ConvLSTMModel, self).__init__()
        
        self.CNN_Model = CNNModel()
        self.LSTM_Model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
        
    def forward(self, replay_tensor, batch_size):
        
        # Input Data = (games, frame, featuer, heigth, width)
        # example Input = (26, 500, 8, 128, 128)
        for i in range(batch_size):
        
            win_data_ = replay_tensor[i]
            # CNN Model Input = (8, 128, 128) → output = (1, 32) vector
            for j in range(len(win_data_)):
                win_data = win_data_[j]    
                win_data = win_data.unsqueeze(dim=0)
                win_cnn = self.CNN_Model(win_data)
                win_cnn = win_cnn.unsqueeze(dim=0)
                
                if j == 0:
                    all_frame = win_cnn
                else:
                    all_frame =  torch.cat([all_frame, win_cnn], dim=0)
            
            # all_frame size = (500, 1, 32)
            all_frame = all_frame.unsqueeze(dim=0)
        
            if i == 0:
                star_cnn = all_frame
            else:
                star_cnn = torch.cat([star_cnn, all_frame], dim=0)
        
        # star_cnn size = (26, 500, 32)
        star_cnn = star_cnn.squeeze(dim=2)
        # LSTM Model Input = (batch = 2, 500, 32)
        out = self.LSTM_Model(star_cnn)

        return out                    

#---------------------------------------- 3. HyperParameter fix ----------------------------------------#

# LSTM Hyper paramter
input_dim = 32
hidden_dim = 32
layer_dim = 1
output_dim = 2
iter = 0

# ConvLSTM Model
model = ConvLSTMModel()

# Loss function
criterion = nn.CrossEntropyLoss()

# learnig_rate
learning_rate = 0.0001

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# len(list(model.parameters()))

# Number of paramter step by step
for i in range(len(list(model.parameters()))):
    print(list(model.parameters())[i].size())
    
# batch, epoch size 
batch_size = 2
n_iters = replay_tensor.shape[0]
num_epochs = n_iters / (len(replay_tensor) / batch_size)
num_epochs = int(num_epochs)
num_batches = int(len(replay_tensor)/batch_size)


#---------------------------------------- 4. Training  ----------------------------------------#

# Traing code
for epoch in range(num_epochs):
    print('\n===> epoch %d' % epoch)
    
    model = model.to(device)
    running_loss = 0.0
    
    # Training
    for bch in range(num_batches):
        #obs=0
        inputs_batch = replay_tensor[bch*batch_size:(bch+1)*batch_size]
        labels_batch = labels_tensor[bch*batch_size:(bch+1)*batch_size]
        inputs_batch, labels_batch = inputs_batch.to(device), labels_batch.to(device)
        
        # inputs_batch = inputs_batch.squeeze(dim=2)
        labels_batch = labels_batch.squeeze(dim=1)        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs_batch, batch_size)
        loss = criterion(outputs, labels_batch)
        loss.backward(retain_graph = True)
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        
        print('[%d, %5d] loss: %.3f' % (epoch, bch*batch_size, running_loss))
        running_loss = 0.0


#---------------------------- 5. Testing (Not Completed editing )  ---------------------------------#
# Testing(now, this is the training set)
class_correct = list(0. for i in range(len(np.unique(labels_tensor))))
class_total = list(0. for i in range(len(np.unique(labels_tensor))))            

with torch.no_grad():
    for bch in range(num_batches):
        inputs_batch = replay_tensor[bch*batch_size:(bch+1)*batch_size]
        labels_batch = labels_tensor[bch*batch_size:(bch+1)*batch_size]
        inputs_batch, labels_batch = inputs_batch.to(device), labels_batch.to(device)
        #net = net.to(device)
        #net = net.cuda()
        
        inputs_batch = inputs_batch.squeeze(dim=2)
        # labels_batch = labels_batch.squeeze(dim=1)        
        
        outputs = model(inputs_batch, batch_size)
        
        _, predicted = torch.max(outputs, 1) # prediction
        c = (predicted == labels_batch).squeeze()
        
        for i in range(len(labels_batch)):
            label = labels_batch[i]
            class_correct[label] += c[i][i].item()
            class_total[label] += 1
            
classes = ('win', 'lose')

# performance evaluation
for i in range(len(np.unique(labels_tensor))):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))