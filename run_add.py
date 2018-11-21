import tensorflow as tf
import argparse
from Add import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='lstm',
                 help='gru, gers, or lstm')
parser.add_argument('--epochs', type=int, default=9000,
                 help='# of Epochs, default 100k')
parser.add_argument('--seed', type=int, default=0,
                 help='default seed 0')
parser.add_argument('--num_seeds', type=int, default=10,
                 help='trains total number of seeds')
parser.add_argument('--learning_rate', type=float, default=10e-5 ,
                 help='learning rate, default 10e-5')
parser.add_argument('--optimizer', type=str, default='adam',
                 help='adam, or momentum')
parser.add_argument('--data', type=str, default='nmsd',
                 help='nmsd, or msd')
parser.add_argument('--interval', type=int, default=999,
                 help='train specific batch')
parser.add_argument('--no_peep', action="store_false", default=True,
                    help='train with peephole')
parser.add_argument('--gpu', action="store_true", default=False,
                    help='use gpu over cpu')
parser.add_argument('--dtype', type=int, default=32,
                 help='specify dtype')
parser.add_argument('--cycles', type=int, default=4,
                 help='num of cycles for MSD task')
parser.add_argument('--layers', type=int, default=1,
                 help='layers')
parser.add_argument('--hidden', type=int, default=100,
                 help='hidden')
parser.add_argument('--batch_size', type=int, default=50,
                 help='batch_size')

args = parser.parse_args()

batch_size = args.batch_size


class Model(object):
    def __init__(self, args):
        self.rate = args.learning_rate
        self.seed = args.seed
        self.optim = args.optimizer
        self.epochs = args.epochs
        self.hidden = args.hidden
        self.layers = args.layers
        self.num_classes = 1
        self.batch_size = args.batch_size
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        if args.dtype == 32:
            self.dtype = tf.float32
        else:
            self.dtype = tf.float64
        print('Using dtype tf.float{}'.format(self.dtype))

        if self.layers ==1:
            if args.model == 'lstm':
                self.cell = tf.contrib.rnn.LSTMCell(num_units=self.hidden, dtype=self.dtype)
            else:
                self.cell = tf.contrib.rnn.GRUCell(num_units=self.hidden, dtype=self.dtype)

        else:
            if args.model == 'lstm':
                self.cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(num_units=self.hidden, dtype=self.dtype)
                                                         for _ in range(self.layers)])
            else:
                self.cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(num_units=self.hidden, dtype=self.dtype)
                                                         for _ in range(self.layers)])
        self.x = tf.placeholder(dtype=self.dtype, shape=[None, None,2])
        batch_size = tf.shape(self.x)[0]
        max_steps = tf.shape(self.x)[1]

        self.y = tf.placeholder(dtype=self.dtype, shape=[None,self.num_classes])
        self.lr = tf.placeholder(dtype=self.dtype, name="learning_rate")

        self.initial_state = self.cell.zero_state(batch_size=batch_size, dtype=self.dtype)

        output, last_state = tf.nn.dynamic_rnn(inputs=self.x, cell=self.cell, dtype=self.dtype)

        last_output = tf.slice(output,[0,max_steps-1,0],[-1,-1,-1])
        #flatten output for dense
        output_flatten = tf.reshape(last_output,[-1,self.hidden])
        self.predict = tf.layers.dense(output_flatten,units=self.num_classes)
        self.loss = tf.losses.mean_squared_error(self.y,self.predict)

        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.grads_vars = self.optimizer.compute_gradients(self.loss)
        self.train_op = self.optimizer.apply_gradients(self.grads_vars,
                                                           global_step=self.global_step)
        self.t_error=tf.summary.scalar(name='Train_Error',tensor=self.loss)

        self.v_error = tf.summary.scalar(name='Validation_Error', tensor=self.loss)

    def step(self,sess,feed_dict):
        return sess.run([self.train_op,self.loss,self.global_step],feed_dict)

    def check(self,sess,feed_dict):
        return sess.run([self.y,self.v_error],feed_dict)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=.1)
lr = args.learning_rate
seeds = 10
t_start = 10
t_end = 20

a = AddingDataWrapper(batch_size,t_start,t_end)
xb,yb,t = a.next_batch()

for seed in range(seeds):
    # set seed
    tf.reset_default_graph()
    tf.set_random_seed(seed)

    model = Model(args)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    init = tf.global_variables_initializer()
    sess.run(init)
    print('Setting seed....')
    for e in range(args.epochs):
        e_loss = 0
        for b in range(0,t_end-t_start+1):
            # print(b)
            feed_dict = {model.x: xb, model.y: yb, model.lr:lr}
            _, b_loss,step=model.step(sess,feed_dict)
            e_loss+=b_loss/(t_end-t_start)
            #next batch
            xb,yb,t = a.next_batch()
        if e%50==0:
            print('Epoch:{} Loss:{:1.3f}'.format(e,e_loss))


    sess.close()