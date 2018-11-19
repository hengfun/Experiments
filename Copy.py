import numpy as np

"""https://arxiv.org/pdf/1710.02224.pdf
Copy memory problem
"""

class CopyTask(object):
    def __init__(self, n, t_start, t_end):
        self.t_start = t_start
        self.t_end = t_end
        self.n = n
        self.generate_data(n,t_start,t_end)
        self.counter = t_start - 1

    def single_copy(self, t):
        # generate a single copy
        head = np.random.randint(0, 8, 10)  # The first ten values are randomly generated from integers 0 to 7
        middle = np.array([8] * (t - 1))  # next T-1 values are all 8
        tail = np.array([9] * 11)  # the last 11 values are all 9
        x = np.concatenate([head, middle, tail], axis=0)
        y = np.zeros_like(x)
        y[-10:] = head  # the last ten values are the inputs
        return x, y

    def generate_batch(self, n, t):
        # generates n copies to form a batch
        x = []
        y = []
        for i in range(0, n):
            x_b, y_b = self.single_copy(t)
            x.append(x_b)
            y.append(y_b)
        x = np.array(x)
        y = np.array(y)
        return x, y

    def generate_data(self, n, t_start, t_end):
        #generates a range from t_start to t_end
        # n: number of samples per batch
        # t_start: where to begin
        # t_start: where to end
        #saves as dict
        x = {}
        y = {}
        for t in range(t_start, t_end + 1):
            x_b, x_y = self.generate_batch(self.n, t)
            x[t] = x_b
            y[t] = x_b
        self.x = x
        self.y = y

    def next_batch(self):
        #each call generates new random data
        #iterates to end and starts over
        self.counter += 1
        self.counter = max(self.counter % (self.t_end + 1), self.t_start)
        t = self.counter % (self.t_end + 1)
        # print(self.counter)
        return self.generate_batch(self.n, t)