import os
import numpy as np
from torchvision import datasets, utils, transforms




class DataWrapper(object):

    def __init__(self,task,batch_size,validation=0.05,seed=1):
        self.clock = 0
        self.batch_size = batch_size
        if task == 'mnist':
            data, targets = get_mnist()
        if task == 'shuffled-mnist':
            data, targets = get_mnist(True,seed)
        else:
            raise NotImplementedError()
        
        self.valid_data = data[:int(data.shape[0]*validation)]
        self.valid_targets = targets[:int(data.shape[0]*validation)]
        self.train_data = data[int(data.shape[0]*validation):]
        self.train_targets = targets[int(data.shape[0]*validation):]

        
        self.train_size = self.train_data.shape[0]
        self.valid_size = self.valid_data.shape[0]
    
    def next_batch(self):
        x = self.train_data[self.clock*self.batch_size: self.clock*self.batch_size+self.batch_size]
        y = self.train_targets[self.clock*self.batch_size: self.clock*self.batch_size+self.batch_size]
        self.clock = (self.clock + 1) % self.train_size
        return x, y 
    
    def validate(self):
        return self.valid_data, self.valid_targets
    


def get_mnist(shuffle=False,seed=None,dire='mnist'):
    if not os.path.isdir(dire):
        os.mkdir('mnist')
    try:
        train_set = datasets.MNIST(root=dire,train=True,download=False,transform=transforms.ToTensor())
    except:
        train_set = datasets.MNIST(root=dire,train=True,download=True,transform=transforms.ToTensor())
    out = []
    labels = []
    for i in range(len(train_set)):
        im, lab = train_set[i]
        im = np.array(im)
        labels.append(np.array(lab))
        out.append(im)
    
    data = np.array(out).reshape(-1,28*28)
    labels = np.array(labels)
    if shuffle:
        rng = np.random.RandomState(seed)
        idx = np.arange(data.shape[1])
        rng.shuffle(idx)
        for i in range(data.shape[0]):
            data[i,:] = data[i,idx]
    
    return data, labels




