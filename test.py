import torch
from torch import nn
from utils import run, cross_validation
from dlc_practical_prologue import generate_pair_sets
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
from model_cnn import FC, CNN, ResNet

# set seed
torch.manual_seed(88)

# generate 1,000 pairs for training and test
train_input, train_target, train_class, test_input, test_target, test_class = generate_pair_sets(1000)

# loss function
cross_entropy = nn.CrossEntropyLoss()

epochs = 25 # number of epochs
rounds = 15 # number of rounds

# test the performance improvement of using weight sharing or auxiliary loss or both
if_weight_sharing = [False, True, False, True]
if_auxiliary_loss = [False, False, True, True]

######################################################################################################################

### FC
AL_weight = 1   # auxiliary loss weighting
model = FC

# Step 1: Find the best parameters using cross validation
if_cross_validation = False

if if_cross_validation:
    k_fold = 5
    lr_set = [0.0001, 0.001, 0.01, 0.1]  # learning rate
    reg_set = [0, 0.1, 0.2, 0.3]  # weight decay factor
    batchsize_set = [20,50,100,200]
    for i in range(4):
        best_lr, best_reg, best_batchsize = cross_validation(k_fold, lr_set, reg_set, batchsize_set, model, cross_entropy, AL_weight, epochs,
                         weight_sharing = if_weight_sharing[i], auxiliary_loss = if_auxiliary_loss[i])
        print(f"For weight_sharing = {if_weight_sharing[i]} and auxiliary_loss = {if_auxiliary_loss[i]}")
        print(f"best_lr = {best_lr}, best_reg = {best_reg}, best_batchsize = {best_batchsize}")

# Step 2: train and test

record_mean_acc = []
record_std_acc = []


# hyperparameters
lr = [0.001, 0.001, 0.001, 0.001]  # learning rate
reg = [0.1, 0.1, 0.1, 0]  # weight decay factor (for l2 regularization)
batch_size = [100, 50, 20, 20]


for i in range(4):
    
    record_acc = []

    for j in range(rounds):

        train_loader = DataLoader(list(zip(train_input, train_target, train_class)), batch_size[i])
        test_loader = DataLoader(list(zip(test_input, test_target, test_class)), batch_size[i])
        model = FC(if_weight_sharing[i], if_auxiliary_loss[i])
        optimizer=optim.Adam(model.parameters(), lr=lr[i], weight_decay=reg[i])
        
        test_acc = run(train_loader, test_loader,
                                model, optimizer,
                                cross_entropy, AL_weight,
                                epochs,
                                if_weight_sharing[i],
                                if_auxiliary_loss[i])
        record_acc.append(test_acc)
        
    mean_acc = np.mean(record_acc)
    std_acc  = np.std(record_acc)
    
    record_mean_acc.append(mean_acc)
    record_std_acc.append(std_acc)


# print the test results
print("-- Result for Fully Connect Network--\n")
for i in range(4):
    print(f"Case {i+1}: \n Auxiliary loss:  {if_auxiliary_loss[i]}, Weight sharing: {if_weight_sharing[i]}\n Test Acc: {record_mean_acc[i]}, Std = {record_std_acc[i]}")


######################################################################################################################

### CNN
AL_weight = 1.0   # auxiliary loss weighting
model = CNN

# Step 1: Find the best parameters using cross validation
if_cross_validation = False

if if_cross_validation:
    
    k_fold = 5
    lr_set = [0.0001, 0.001, 0.01, 0.1]  # learning rate
    reg_set = [0, 0.1, 0.2, 0.3]  # weight decay factor
    batchsize_set = [20,50,100,200]
    
    for i in range(4):
        best_lr, best_reg, best_batchsize = cross_validation(k_fold, lr_set, reg_set, batchsize_set, model, cross_entropy, AL_weight, epochs,
                          weight_sharing = if_weight_sharing[i], auxiliary_loss = if_auxiliary_loss[i])
        print(f"For weight_sharing = {if_weight_sharing[i]} and auxiliary_loss = {if_auxiliary_loss[i]}")
        print(f"best_lr = {best_lr}, best_reg = {best_reg}, best_batchsize = {best_batchsize}")

# Step 2: train and test

record_mean_acc = []
record_std_acc = []


# hyperparameters
lr = [0.01, 0.001, 0.001, 0.01]  # learning rate
reg = [0, 0, 0, 0]  # weight decay factor
batch_size = [100, 20, 20, 50]

for i in range(4):
    
    record_acc = []

    for j in range(rounds):
        print(i,j)
        model = CNN(if_weight_sharing[i], if_auxiliary_loss[i])
        optimizer=optim.Adam(model.parameters(), lr=lr[i], weight_decay=reg[i])
        
        train_loader = DataLoader(list(zip(train_input, train_target, train_class)), batch_size[i])
        test_loader = DataLoader(list(zip(test_input, test_target, test_class)), batch_size[i])
        
        test_acc = run(train_loader, test_loader,
                                model, optimizer,
                                cross_entropy, AL_weight,
                                epochs,
                                if_weight_sharing[i],
                                if_auxiliary_loss[i])
        record_acc.append(test_acc)
        
    mean_acc = np.mean(record_acc)
    std_acc  = np.std(record_acc)
    
    record_mean_acc.append(mean_acc)
    record_std_acc.append(std_acc)


# print the test results
print("-- Result for Convolution Network--\n")
for i in range(4):
    print(f"Case {i+1}: \n Auxiliary loss:  {if_auxiliary_loss[i]}, Weight sharing: {if_weight_sharing[i]}\n Test Acc: {record_mean_acc[i]}, Std = {record_std_acc[i]}")
    
###########################################################################################################

### Resnet
AL_weight = 1.0   # auxiliary loss weighting
model = ResNet

# Step 1: Find the best parameters using cross validation
if_cross_validation = False

if if_cross_validation:
    
    k_fold = 5
    lr_set = [0.0001, 0.001, 0.01, 0.1]  # learning rate
    reg_set = [0, 0.1, 0.2, 0.3]  # weight decay factor
    batchsize_set = [20,50,100,200]
    
    for j in range(3):
        i = j + 1
        print(f"For weight_sharing = {if_weight_sharing[i]} and auxiliary_loss = {if_auxiliary_loss[i]}")
        best_lr, best_reg, best_batchsize = cross_validation(k_fold, lr_set, reg_set, batchsize_set, model, cross_entropy, AL_weight, epochs,
                          weight_sharing = if_weight_sharing[i], auxiliary_loss = if_auxiliary_loss[i])
        print(f"For weight_sharing = {if_weight_sharing[i]} and auxiliary_loss = {if_auxiliary_loss[i]}")
        print(f"best_lr = {best_lr}, best_reg = {best_reg}, best_batchsize = {best_batchsize}")

# Step 2: train and test

record_mean_acc = []
record_std_acc = []


# hyperparameters
lr = [0.0001, 0.0001, 0.001, 0.01]  # learning rate
reg = [0, 0, 0, 0]  # weight decay factor
batch_size = [100, 20, 20, 100]

for i in range(4):
    
    record_acc = []

    for j in range(rounds):
        print(i,j)
        model = ResNet(if_weight_sharing[i], if_auxiliary_loss[i])
        optimizer=optim.Adam(model.parameters(), lr=lr[i], weight_decay=reg[i])
        
        train_loader = DataLoader(list(zip(train_input, train_target, train_class)), batch_size[i])
        test_loader = DataLoader(list(zip(test_input, test_target, test_class)), batch_size[i])
        
        test_acc = run(train_loader, test_loader,
                                model, optimizer,
                                cross_entropy, AL_weight,
                                epochs,
                                if_weight_sharing[i],
                                if_auxiliary_loss[i])
        record_acc.append(test_acc)
        
    mean_acc = np.mean(record_acc)
    std_acc  = np.std(record_acc)
    
    record_mean_acc.append(mean_acc)
    record_std_acc.append(std_acc)


# print the test results
print("-- Result for Convolution Network--\n")
for i in range(4):
    print(f"Case {i+1}: \n Auxiliary loss:  {if_auxiliary_loss[i]}, Weight sharing: {if_weight_sharing[i]}\n Test Acc: {record_mean_acc[i]}, Std = {record_std_acc[i]}")
