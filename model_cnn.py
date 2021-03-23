import torch
from torch import nn
from torch.nn import functional as F
import pdb

class SiameseCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # convolutional network
        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, 3, stride=1, padding=0, dilation=1),
                                   nn.Tanh(),
                                   nn.MaxPool2d(2, stride=1),
                                   nn.Dropout(0.3),

                                   nn.Conv2d(16, 64, 3),
                                   nn.Tanh(),
                                   nn.MaxPool2d(2, stride=1),
                                   nn.Dropout(0.3),
                                   )

        # fully connected layer
        self.fc1 = nn.Sequential(nn.Linear(64 * 8 * 8, 256),
                                 nn.Tanh(),
                                 nn.Linear(256, 10),
                                 nn.Softmax(-1)
                                 )

        self.out = nn.Sequential(nn.Linear(20, 2), 
                               nn.Tanh(),
                               nn.Softmax(-1))
 
        self.compare = nn.Sequential(nn.Linear(20, 2),  nn.Softmax(-1))

    def forward(self, inputdata, aux_loss=True):
        n_epoch, dim, h, w = inputdata.size()
        output = []
        for i in range(dim):
            data = inputdata[:, i, :, :]
            data = data.view(n_epoch,1,h,w)
            output1 = self.conv1(data)
            output1 = output1.view(n_epoch, -1)
            output1 = self.fc1(output1)
            output.append(output1)

        if aux_loss:
            classes = torch.cat(output, 1)
            bool_out = self.compare(classes)
            return output, bool_out

        if not (aux_loss):
            classes = torch.cat(output, 1)
            bool_out = self.compare(classes)
            return bool_out