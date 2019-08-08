# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 15:47:27 2019

@author: user
"""

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
from tqdm import tqdm

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
# cpu device
device = torch.device('cpu')

# conver numpy to tensor
replay_tensor = torch.FloatTensor(replay).to(device) # (26, 500, 8, 128, 128)
labels_tensor = torch.LongTensor(labels).to(device) # (26, 1)

#------------------------------------- 1.1 preprocessing(one-hot-channel[3d]) -------------------------------------#

def embedding_preprocessing(replay_cat, features, matchs, timesteps):
 
    one_hot = replay_cat[:,:,features,:,:]
    _, _, width, height = one_hot.shape
            
    # scale definition
    if features == 0:
        scale_ = 4    
    elif features == 2:
        scale_ = 5
    elif features == 4:
        scale_ = 1914  
    else:
        scale_ = 2
        
    # if  one_hot_value > scale_, change value '0'
    one_hot = one_hot.flatten()
        
    over_idx = np.where(one_hot > scale_)[0]
    one_hot[over_idx] = 0
    
    one_hot = one_hot.reshape(replay_cat.shape[0], replay_cat.shape[1], replay_cat.shape[3], replay_cat.shape[4])
        
    # One_hot_Encoding Categorical feature
    one_hot_ = one_hot[matchs][timesteps]
    one_hot_ = one_hot_.flatten()
    nonzero_indices = np.where(one_hot_ != 0)[0]
    nonzero_values = one_hot_[nonzero_indices] 
                
    def get_yx(index):
        return [int(nonzero_values[index]), nonzero_indices[index]]
                    
    yx_indices = [get_yx(i) for i in range(nonzero_indices.shape[0])]
     
    result = np.zeros((height*width, scale_))
                   
    for (y, x) in yx_indices:
        result[x][y] = 1.
                
    result = np.transpose(result)
        
    return result

#------------------------------ 2. Embedding(categorical), CNN(Continous)-------------------------------------#

# using Embedding layer for 'categorical value'
class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embed(x)
    
# Continous features Convolution
class Continuous_conv(nn.Module):
    def __init__(self, emb_dim):
        super(Continuous_conv, self).__init__()
        
        self.emb_dim = emb_dim
       # Continuous Convolution 1
        self.cnn_con1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=0)
        self.relu_con1 = nn.ReLU()
        # Max pool 1
        self.maxpool_con1 = nn.MaxPool2d(kernel_size=2)
        
        # Continuous Convolution 2
        self.cnn_con2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=0)
        self.relu_con2 = nn.ReLU()
        
        # Max pool 2
        self.maxpool_con2 = nn.MaxPool2d(kernel_size=2)  
        #  
        self.fc1 = nn.Linear(961, self.emb_dim)        
                     
    def forward(self, x):

        # Conv 1
        out_con = self.cnn_con1(x).requires_grad_()
        out_con = self.relu_con1(out_con)
            
        # Max pool 1
        out_con = self.maxpool_con1(out_con)
        
        # Conv 2
        out_con = self.cnn_con2(x).requires_grad_()
        out_con = self.relu_con2(out_con)
            
        # Max pool 2
        out_con = self.maxpool_con2(out_con)
        out_con = out_con.view(out_con.size(0), -1)
        
        # Linear function (readout)
        out_con = self.fc1(out_con).detach()            
        
        return out_con
#------------------------------------ 3. CovLSTM Modeling -----------------------------------------

# Convolutional Neural Network
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1928, out_channels=964, kernel_size=(1,2), stride=1, padding=0)
        self.relu1 = nn.ReLU()
        
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1,2))
     
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=964, out_channels=482, kernel_size=(1,2), stride=1, padding=0)
        self.relu2 = nn.ReLU()
        
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1,2))
        
        # Convolution 3
        self.cnn3 = nn.Conv2d(in_channels=482, out_channels=241, kernel_size=(1,2), stride=1, padding=0)
        self.relu3 = nn.ReLU()
        
        # Max pool 3
        self.maxpool3 = nn.MaxPool2d(kernel_size=(1,2))
        self.drop_out1 = nn.Dropout(0.5)
        
        # Convolution 4
        self.cnn4 = nn.Conv2d(in_channels=241, out_channels=100, kernel_size=(1,2), stride=1, padding=0)
        self.relu4 = nn.ReLU()
        self.drop_out2 = nn.Dropout(0.5)
        
        # Fully connected 1 (shape = 1, 100)
        self.fc1 = nn.Linear(2300, 100)
                     
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
        out = self.drop_out1(out)
        
        # Convolution 4
        out = self.cnn4(out).requires_grad_()
        out = self.relu4(out)
        out = self.drop_out2(out)
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
    def __init__(self, vocab_size, d_model, emb_dim):
        super(ConvLSTMModel, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.emb_dim = emb_dim
        self.Embedding = Embedder(self.vocab_size, self.d_model)
        
        # Continuous Convolution
        self.conv_conti = Continuous_conv(self.emb_dim)     
        #CNN Model
        self.CNN_Model = CNNModel()
        #LSTM Model
        self.LSTM_Model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
        
    def forward(self, replay_tensor, batch_size):
        
        replay = replay_tensor.numpy()
        
        replay = np.delete(replay, 2, 2) # delete 'creep' feature
        replay = np.delete(replay, 3, 2) # delte 'player_id' feature
        
        # split continuous, categorical
        replay_scalar = replay[:,:,0,:,:] # select 'visibility_map' feature
        replay_scalar = np.expand_dims(replay_scalar, 2)
        replay_cat = replay[:,:,1:6,:,:] # select categorical features(except for 'visibility_map')
        
        # embedding categorical features
        for i in range(batch_size):
            for j in range(replay_cat.shape[1]):
                for k in range(replay_cat.shape[2]):
                    replay_pre = embedding_preprocessing(replay_cat, k, i, j) 
        
                    replay_pre = torch.LongTensor(replay_pre) # (26, 500, scale_, 128*128)
                    embedding = self.Embedding(replay_pre).to(device)
                    embedding = Embedder(replay_pre.shape[0], emb_dim).to(device)
                    replay_emb = embedding(replay_pre)
                    for s in range(replay_emb.shape[0]):
                        replay_emb_ = replay_emb[s,0,:]
                        replay_emb_ = replay_emb_.unsqueeze(dim=0) # replay_emb size = (1, emb_dim) # embedding size
                        if s == 0:
                            replay_emb_feat = replay_emb_
                        else:
                            # replay_emb_feat size = (num of cat, 500)
                            replay_emb_feat = torch.cat([replay_emb_feat, replay_emb_], dim=0)
                                  
                    if k == 0:
                        replay_emb_time = replay_emb_feat 
                    else:
                        replay_emb_time = torch.cat([replay_emb_time, replay_emb_feat], dim=0)
                
                replay_emb_time = replay_emb_time.unsqueeze(dim=0)
                if j == 0:
                    replay_emb_match = replay_emb_time
                else:
                    replay_emb_match = torch.cat([replay_emb_match, replay_emb_time], dim=0)
                print('=' * 80)
                print('replay_emedding frame by frame_' + str(j+1) + '_' + str(i+1) + '_th steps')
                
            replay_emb_match = replay_emb_match.unsqueeze(dim=0)
            if i == 0:
                replay_emb_full = replay_emb_match
            else:
                replay_emb_full = torch.cat([replay_emb_full, replay_emb_match], dim=0)
            print('=' * 80)
            print('replay_emedding match by match_' + str(i+1) + '_th steps')       
        
        print('=============================Completed Embeeding=============================') 
        # Continuous Convolution
        for i in range(batch_size):
            for j in range(replay_scalar.shape[1]):
                replay_sca_con = replay_scalar[i,j,:,:,:]
                replay_sca_con = torch.FloatTensor(replay_sca_con) # (feature, 128, 128)
                replay_sca_con = replay_sca_con.unsqueeze(dim=0)
                #conv_conti = Continuous_conv(emb_dim)
                replay_sca = self.conv_conti(replay_sca_con)
                replay_sca = replay_sca.unsqueeze(dim=0)
                
                if j == 0:
                    replay_sca_time = replay_sca
                else:
                   replay_sca_time = torch.cat([replay_sca_time, replay_sca], dim=0) 
            
            replay_sca_time = replay_sca_time.unsqueeze(dim=0)
            if i==0:
                replay_sca_full = replay_sca_time
            else:
                replay_sca_full = torch.cat([replay_sca_full, replay_sca_time], dim=0)
        
        # replay data after preprocessing(Categorical & Continous)   
        replay_all = torch.cat([replay_sca_full, replay_emb_full], dim=2)
        replay_all = replay_all.unsqueeze(dim=3)
        
        print('=' * 80)
        print('Completed Preprocessing & Ready to ConvLSTM Modeling')
        # Input Data = (games, frame, no of cat, 1, embed_size)
        # example Input = (26, 500, 1928, 1, embed_size)
        for i in range(batch_size):
        
            replay_data_ = replay_all[i]
            # CNN Model Input = (1928, 1, embed_size) → output = (1, 100) vector
            for j in range(len(replay_data_)):
                replay_data = replay_data_[j]    
                replay_data = replay_data.unsqueeze(dim=0)
                replay_cnn = self.CNN_Model(replay_data).to(device)
                replay_cnn = replay_cnn.unsqueeze(dim=0)
                
                if j == 0:
                    all_frame = replay_cnn
                else:
                    all_frame =  torch.cat([all_frame, replay_cnn], dim=0)
            
            # all_frame size = (500, 1, 100)
            all_frame = all_frame.unsqueeze(dim=0)
        
            if i == 0:
                star_cnn = all_frame
            else:
                star_cnn = torch.cat([star_cnn, all_frame], dim=0)
        
        # star_cnn size = (26, 500, 100)
        star_cnn = star_cnn.squeeze(dim=2)
        # LSTM Model Input = (batch = 2, 500, 100)
        out = self.LSTM_Model(star_cnn).to(device)

        return out

#---------------------------------------- 3. HyperParameter fix ----------------------------------------#

# LSTM Hyper paramter
input_dim = 100
hidden_dim = 100
layer_dim = 1
output_dim = 2
iter = 0

# Embedding hyper paramter
vocab_size = replay_tensor.shape[3] * replay_tensor.shape[4] 
d_model = 200
emb_dim = 200

# ConvLSTM Model
model = ConvLSTMModel(vocab_size, d_model, emb_dim)

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
        # c = (predicted == labels_batch).squeeze()
        c = (predicted == labels_batch).sum().item()
        
        for i in range(len(labels_batch)):
            label = labels_batch[i]
            class_correct[label] += c[i][i].item()
            class_total[label] += 1
            
classes = ('win', 'lose')

# performance evaluation
for i in range(len(np.unique(labels_tensor))):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))