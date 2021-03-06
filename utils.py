import torch
from torch import optim
from dlc_practical_prologue import generate_pair_sets
from torch.utils.data import DataLoader
import numpy as np



def run(train_data_loader, test_data_loader,
          model, optimizer, criterion, AL_weight=0.5,
          epochs = 25,  weight_sharing=False, auxiliary_loss=False):
    """
    Train model with given dataset and parameters, and test the model after training
    Args:
        train_data_loader: data loader for training set
        test_data_loader: data load for test set
        model: the model to be trained
        optimizer: optimizer for training
        criterion: loss function
        AL_weight: weight of auxiliary loss
        epochs: number of training epochs
        weight_sharing: if applying weight sharing
        auxiliary_loss: if applying auxiliary loss
    Return:
        acc_test: accuracy on test set
    """

    for epoch in range(epochs):
        for (image, target, classes) in train_data_loader:
            model.train()
            optimizer.zero_grad()
            
            if auxiliary_loss:
                digit1, digit2, output = model(image)
                output_loss = criterion(output, target)
                aux_loss = criterion(digit1, classes[:, 0]) + criterion(digit2, classes[:, 1])
                loss = output_loss + AL_weight * aux_loss
                
            else:
                output = model(image)
                loss = criterion(output, target)
                
            loss.backward()
            optimizer.step()
                
    # evaluate model at the end of last epoch
    model.eval()
    with torch.no_grad():
            acc_train, loss_tr= evaluate(model, train_data_loader, auxiliary_loss, criterion)
            acc_test, loss_te = evaluate(model, test_data_loader, auxiliary_loss, criterion)

    return acc_test



def evaluate(model, data_loader, auxiliary_loss, criterion):
    """
    Evaluatinmg given network model with given dataset and criterion
    Args:
        model: the model to be evaluated
        data_loader: dataset used to evaluate the model
        auxiliary_loss: if the model using auxiliary loss 
        (the model using auxiliary loss retruns 3 values, but we only need the final output)
        criterion: loss function
    Returns:
        testing accuracy, testing loss
    """
    correct = 0
    total = 0
    loss = 0
    
    for (image, target, digit_target) in data_loader:
        total += len(target)
        
        if auxiliary_loss:
            _, _, output = model(image)
            
        else:
            output = model(image)
            
        loss += criterion(output, target)
        _, pred = torch.max(output, 1)
        correct += (pred == target).sum().item()
            
    return correct / total, loss



def cross_validation(k_fold, lr_set, reg_set, batchsize_set, model, criterion, AL_weight, epochs,
                     weight_sharing = False, auxiliary_loss = False):
    """
    K-fold cross validation to optimize hyperprameters learning rate (lr), regularization coefficoent (reg), and batchsize on the given model
    Args:
        k_fold: number of cross validation folds
        lr_set: set of learning rate (lr)
        reg_set: set of regularization coefficoent (reg)
        batchsize_set: set of batchsize
        model: network model
        criterion: loss function
        AL_weight: Weight of auxiliary loss
        epochs: number of epochs
        weight_sharing: if applying weight sharing
        auxiliary_loss: if applying auxiliary loss
    Returns:
        best_lr: best learning rate
        best_reg: best regularization coefficoent
        best_batchsize: best batch size
    """

    data_input, data_target, data_class, _, _, _ = generate_pair_sets(1000) # generate data set

    size_validation_set = int(data_input.shape[0]/ k_fold) # calculate data set size for each fold
    indices = torch.randperm(data_input.shape[0]) # shuffle data indicies
    
    # traverse all combination of lr, and reg to find the best hyperparameters
    max_test_accuracy = 0
    best_lr = 0
    best_reg = 0
    best_batchsize = 0
    
    for lr in lr_set:
        for reg in reg_set:
            for batch_size in batchsize_set:
            
                acc_accuracy_test = 0
                
                print('lr = ', lr, 'reg = ', reg, 'batch_size = ', batch_size)
                for k in range(k_fold):
                    # initialize model (set the number of input channels and output channels to 2)
                    network = model(weight_sharing, auxiliary_loss)
                    optimizer = optim.Adam(network.parameters(), lr=lr, weight_decay=reg)
                    
                    # divide data into k-fold and prepare train and validation sets
                    train_indices = torch.cat((indices[0:k*size_validation_set],indices[(k+1)*size_validation_set:]),0)
                    
                    train_input = data_input[train_indices]
                    train_target = data_target[train_indices]
                    train_class = data_class[train_indices]
                    
                    val_indices = indices[ k*size_validation_set : (k+1)*size_validation_set ]
                    
                    val_input = data_input[val_indices]
                    val_target = data_target[val_indices]
                    val_class = data_class[val_indices]
    
                    # Data loaders
                    train_set = DataLoader(list(zip(train_input, train_target, train_class)), batch_size, shuffle=True)
                    test_set = DataLoader(list(zip(val_input , val_target, val_class)), batch_size, shuffle=True)
                    accuracy_test = run(train_set, test_set,
                                                    network,
                                                    optimizer,
                                                    criterion, AL_weight,
                                                    epochs, 
                                                    weight_sharing,
                                                    auxiliary_loss)
    
                    acc_accuracy_test += accuracy_test
                
                tmp_accuracy_test = acc_accuracy_test / k_fold
    
                if tmp_accuracy_test > max_test_accuracy:
                    max_test_accuracy = tmp_accuracy_test
                    best_lr = lr
                    best_reg = reg
                    best_batchsize = batch_size
                    
    return best_lr, best_reg, best_batchsize
