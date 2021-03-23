import torch
from torch import nn

def to_one_hot(classes):
    size = classes.size()
    if len(size) == 2:
        n, k = size[0], size[1]
        one_hot_classes = torch.zeros(n, k, 10)
        for i in range(k):
            one_hot_classes[range(0, n), i, classes[:, i]] = 1
        return one_hot_classes

    if len(size) == 1:
        one_hot_classes = torch.zeros(size[0], 2)
        one_hot_classes[range(0, size[0]), classes] = 1
        return one_hot_classes

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

class DatasetLoader():
    def __init__(self, data, target, classes, batch_size=1,shuffle = False):
        self.n = 0
        self.batch_size = batch_size
        self.target = target
        self.classes = classes
        self.nsample, _, _, _ = data.size()
        self.idx = 0
        self.n_times = self.nsample/self.batch_size
        self.shuffle = shuffle
        self.data = data


    def __iter__(self):
        if (self.shuffle == True) & (self.n==0):
            self.perm = torch.randperm(self.nsample)
            self.data = self.data[self.perm]
            self.target = self.target[self.perm]
            self.classes = self.classes[self.perm]

        self.idx = self.idx+self.batch_size
        return self

    def __next__(self):
        if self.n <= self.n_times :
            x = self.n
            self.n = self.n + 1
            batch_target = self.target[self.idx:self.idx+self.batch_size, :].type(torch.float).reshape(self.batch_size, -1)
            batch_data = self.data[self.idx:self.idx+self.batch_size, :, :, :]
            batch_classes = []
            batch_classes.append(self.classes[self.idx:self.idx+self.batch_size, 0, :])
            batch_classes.append(self.classes[self.idx:self.idx+self.batch_size, 1, :])
        else:
            self.n = 0
            self.idx = 0
            raise StopIteration

        return batch_data, batch_target, batch_classes

    def __len__(self):
        return self.n_times