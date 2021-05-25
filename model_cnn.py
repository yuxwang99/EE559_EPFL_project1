import torch
from torch import nn
import torch.nn.functional as F

#################################################
class FC(nn.Module):
    def __init__(self, weight_sharing, auxiliary_loss):
        
        super().__init__()
        self.weight_sharing = weight_sharing
        self.auxiliary_loss = auxiliary_loss
        self.fc1 = nn.Sequential(nn.Linear(14*14, 128),
                                  nn.ReLU(),
                                  nn.Linear(128,64),
                                  nn.ReLU(),
                                  nn.Linear(64,10),
                                  nn.Softmax(-1))
            
        self.fc2 = nn.Sequential(nn.Linear(14*14, 128),
                                  nn.ReLU(),
                                  nn.Linear(128,64),
                                  nn.ReLU(),
                                  nn.Linear(64,10),
                                  nn.Softmax(-1))
        self.fc3 = nn.Linear(20, 2)

    def forward(self, x):
        _x1 = torch.reshape(x[:, 0, :, :], (-1, 1, 14, 14))
        _x1 = torch.reshape(_x1, (_x1.shape[0], -1))
        _x2 = torch.reshape(x[:, 1, :, :], (-1, 1, 14, 14))
        _x2 = torch.reshape(_x2, (_x2.shape[0], -1))

        if self.weight_sharing == True:
            y1 = self.fc1(_x1)
            y2 = self.fc1(_x2)
        else:
            y1 = self.fc1(_x1)
            y2 = self.fc2(_x2)

        y = torch.cat((y1, y2), 1)
        y = self.fc3(y)
        if self.auxiliary_loss == True:
            return y1, y2, y
        else:
            return y
        
        
####################################################
class CNN(nn.Module):
    def __init__(self, weight_sharing, auxiliary_loss):
        super().__init__()
        self.weight_sharing = weight_sharing
        self.auxiliary_loss = auxiliary_loss
        self.conv1 = nn.Sequential(nn.Conv2d(1, 64, 3),  nn.Tanh(), nn.BatchNorm2d(64), nn.MaxPool2d(2,2), nn.Dropout(0.5), nn.Conv2d(64, 64, 3),  nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2,2),nn.Dropout(0.5) )
        self.conv2 = nn.Sequential(nn.Conv2d(1, 64, 3), nn.Tanh(), nn.BatchNorm2d(64), nn.MaxPool2d(2,2),  nn.Dropout(0.5), nn.Conv2d(64, 64, 3),  nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2,2),nn.Dropout(0.5) )
        self.fc1 = nn.Sequential(nn.Linear(64 * 2 * 2, 256), nn.Tanh(), nn.Linear(256, 10), nn.Softmax(-1))
        self.fc2 = nn.Sequential(nn.Linear(64 * 2 * 2, 256), nn.Tanh(), nn.Linear(256, 10), nn.Softmax(-1))
        self.fc3 = nn.Sequential(nn.Linear(20, 2), nn.Softmax(-1))

    def forward(self, x):
        if self.weight_sharing:
            x1 = self.conv1(torch.unsqueeze(x[:,0],dim=1))
            x2 = self.conv1(torch.unsqueeze(x[:,1],dim=1))
            
            x1 = x1.view(-1, 64 *2 *2)
            x2 = x2.view(-1, 64 *2 *2)
            
            x1 = self.fc1(x1)
            x2 = self.fc1(x2)
        else:
            x1 = self.conv1(torch.unsqueeze(x[:,0],dim=1))
            x2 = self.conv2(torch.unsqueeze(x[:,1],dim=1))
            
            x1 = x1.view(-1, 64 *2 *2)
            x2 = x2.view(-1, 64 *2 *2)
            
            x1 = self.fc1(x1)
            x2 = self.fc2(x2)

        y = torch.cat((x1, x2), dim = 1)
        y = y.view(-1, 20)
        y = self.fc3(y)

        if self.auxiliary_loss == True:
            return x1, x2, y
        else:
            return y
        
##########################################     
#Resnet

class ResBlock(nn.Module):
    def __init__(self, kernel_size):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(32, 32, kernel_size,padding=(kernel_size - 1) // 2), nn.ReLU(), nn.BatchNorm2d(32))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, kernel_size,padding=(kernel_size - 1) // 2), nn.ReLU())
        self.bn = nn.BatchNorm2d(32)
    def forward(self, x):
        
        y = self.conv1(x)
        y = self.conv2(y)
        y += x
        y = self.bn(y)
        
        return y


class ResNet(nn.Module):
    def __init__(self, weight_sharing, auxiliary_loss):
        super(ResNet, self).__init__()
        self.weight_sharing = weight_sharing
        self.auxiliary_loss = auxiliary_loss
        self.conv0 = nn.Conv2d(1, 32, kernel_size=1)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=1)
        self.resblocks1 = nn.Sequential(*(ResBlock(3) for _ in range(3)))
        self.resblocks2 = nn.Sequential(*(ResBlock(3) for _ in range(3)))
        self.avg1 = nn.AvgPool2d(kernel_size=2)
        self.avg2 = nn.AvgPool2d(kernel_size=2)
        self.dropout = nn.Dropout(p=0.3)
        self.fc1=nn.Sequential(nn.Linear(32*7*7,128),nn.ReLU(),nn.Linear(128,10), nn.Softmax(-1))
        self.fc2=nn.Sequential(nn.Linear(32*7*7,128),nn.ReLU(),nn.Linear(128,10), nn.Softmax(-1))
        self.fc3= nn.Sequential(nn.Linear(20,2), nn.Softmax(-1))
    
    def forward(self, x):
        if self.weight_sharing==True:
            x1 = torch.reshape(x[:, 0, :, :], (-1, 1, 14, 14))
            x2 = torch.reshape(x[:, 1, :, :], (-1, 1, 14, 14))
            x1 = F.relu(self.conv0(x1))
            x2 = F.relu(self.conv0(x2))
            y1 = self.resblocks1(x1)
            y2 = self.resblocks1(x2)
            
            y1 = self.avg1(y1)
            y2 = self.avg1(y2)
            y1 = self.dropout(y1)
            y2 = self.dropout(y2)
            
            y_1 = y1.view(-1, 32 * 7 * 7)
            y_2 = y2.view(-1, 32 * 7 * 7)
            y_1 = self.fc1(y_1)
            y_2 = self.fc1(y_2)
        else:
            x1 = torch.reshape(x[:, 0, :, :], (-1, 1, 14, 14))
            x2 = torch.reshape(x[:, 1, :, :], (-1, 1, 14, 14))
            x1 = F.relu(self.conv0(x1))
            x2 = F.relu(self.conv1(x2))
            y1 = self.resblocks1(x1)
            y2 = self.resblocks2(x2)
            
            y1 = self.avg1(y1)
            y2 = self.avg2(y2)
            y1 = self.dropout(y1)
            y2 = self.dropout(y2)
            
            y_1 = y1.view(-1, 32 * 7 * 7)
            y_2 = y2.view(-1, 32 * 7 * 7)
            y_1 = self.fc1(y_1)
            y_2 = self.fc2(y_2)
        
        y=torch.cat((y_1, y_2), 1)
        y = y.view(-1, 20)
        y = self.fc3(y)
        
        if self.auxiliary_loss == True:
            return y_1, y_2, y
        else:
            return y
