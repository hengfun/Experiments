import numpy as np
import itertools

np.random.seed(0)

def generate_add_data(n,max_t):
    #n: number of samples
    #max_t: maximum length of sequence
    max_l = max_t*2
    noise = np.random.uniform(0,1,(n,max_l))
    T = np.random.randint(1,max_t,(n))
    max_roll = max_l-T-1

    mask = np.zeros((n,max_l))

    for i,t in enumerate(T):
        mask[i,0] = 1
        mask[i,t] = 1

    rolls = []
    for roll in max_roll:
        rolls.append(np.random.randint(0,roll,1))
    rolls = np.vstack(rolls)

    for i,r in enumerate(rolls):
        mask[i] = np.roll(mask[i],r)
    y = (mask*noise).sum(axis=1).reshape(-1,1)
    mask = mask.reshape(-1,max_l,1)
    noise = noise.reshape(-1,max_l,1)
    x = np.concatenate([mask,noise],2)
    y = (mask*noise).sum(axis=1).reshape(-1,1)
    print('{:2.2f}% of n are max T'.format((T==max_t-1).sum()/n*100))
    return x,y,T