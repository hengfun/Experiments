import numpy as np
import itertools




class AddingDataWrapper(object):
    def __init__(self,batch_size, min_T, max_T,seed=1):
        self.min_T = self.clock = min_T
        self.max_T = max_T
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)

    def next_batch(self):
        X,Y,T = generate_add_data(self.batch_size,self.clock,self.rng)
        self.clock += 1
        if self.clock > self.max_T:
            self.clock = self.min_T
        return X,Y,T






def generate_add_data(n,max_t,rng):
    #n: number of samples
    #max_t: maximum length of sequence
    max_l = max_t*2
    noise = rng.uniform(0,1,(n,max_l))
    T = rng.randint(1,max_t,(n))
    max_roll = max_l-T-1

    mask = np.zeros((n,max_l))

    for i,t in enumerate(T):
        mask[i,0] = 1
        mask[i,t] = 1

    rolls = []
    for roll in max_roll:
        rolls.append(rng.randint(0,roll,1))
    rolls = np.vstack(rolls)

    for i,r in enumerate(rolls):
        mask[i] = np.roll(mask[i],r)
    y = (mask*noise).sum(axis=1).reshape(-1,1)
    mask = mask.reshape(-1,max_l,1)
    noise = noise.reshape(-1,max_l,1)
    x = np.concatenate([mask,noise],2)
    y = (mask*noise).sum(axis=1).reshape(-1,1)
    return x,y,T