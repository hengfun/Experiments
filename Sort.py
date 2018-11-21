import numpy as np 




class SortingDataWrapper(object):
    def __init__(self,batch_size,min_T,max_T,seed=1):
        self.rng = np.random.RandomState(seed)
        self.batch_size = batch_size
        self.clock = self.min_T = min_T
        self.max_T = max_T

    def next_batch(self):
        data = []
        targets = []
        masks = []
        for i in range(self.batch_size):
            x,y,m = self.generate(self.clock)
            data.append(x)
            targets.append(y)
            masks.append(m)
        self.clock += 1
        if self.clock > self.max_T:
            self.clock = self.min_T
        return np.array(data), np.array(targets), np.array(masks)

    def validate(self,T):
        data = []
        targets = []
        masks = []
        for i in range(self.batch_size):
            x,y,m = self.generate(T)
            data.append(x)
            targets.append(y)
            masks.append(m)
        self.clock += 1
        if self.clock > self.max_T:
            self.clock = self.min_T
        return np.array(data), np.array(targets), np.array(masks)

    def generate(self,T):
        X = np.zeros((T*2, 2))
        X[T-1,0] = 1.0
        x = self.rng.uniform(-1,1,(T))
        X[:T,1] = x
        y = np.argsort(x)
        Y = np.zeros(T*2)
        Y[T:] = y
        M = np.zeros((T*2))
        M[T:] = 1.0
        return X, Y, M 
 
class __SortingDataWrapper(object):
    def __init__(self,batch_size,num_elements,min_T,max_T,seed=1):
        self.rng = np.random.RandomState(seed)
        self.batch_size = batch_size
        self.num_elements = num_elements
        self.clock = self.min_T = min_T
        self.max_T = max_T

    def next_batch(self):
        data = []
        targets = []
        for i in range(self.batch_size):
            x,y = self.generate(self.num_elements,self.clock)
            data.append(x)
            targets.append(y)
        self.clock += 1
        if self.clock > self.max_T:
            self.clock = self.min_T
        return np.array(data), np.array(targets)

    def generate(self,num_elem,T):
        idx = np.arange(T)
        x = self.rng.uniform(0,1,(T,num_elem))
        self.rng.shuffle(idx)
        x[:,0] = idx 
        y = x[:,1:]
        #y = y[idx]
        tmp = x[:,1:]
        x[:,1:] = tmp[:,idx]
        return x, y
    

