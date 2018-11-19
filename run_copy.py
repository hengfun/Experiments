import tensorflow as tf
import argparse
from Copy import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='lstm',
                 help='gru, gers, or lstm')
parser.add_argument('--epochs', type=int, default=100000,
                 help='# of Epochs, default 100k')
parser.add_argument('--seed', type=int, default=0,
                 help='default seed 0')
parser.add_argument('--num_seeds', type=int, default=10,
                 help='trains total number of seeds')
parser.add_argument('--learning_rate', type=float, default=10e-4 ,
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
parser.add_argument('--hidden', type=int, default=50,
                 help='hidden')
parser.add_argument('--batch_size', type=int, default=1000,
                 help='batch_size')

args = parser.parse_args()

batch_size = args.batch_size
c = CopyTask(batch_size,10,20)
xb,yb= c.next_batch()

import tensorflow as tf


class Model(object):
    def __init__(self, args):
        self.rate = args.learning_rate
        self.seed = args.seed
        self.optim = args.optimizer
        self.epochs = args.epochs
        self.hidden = args.hidden
        self.layers = args.layers
        self.num_classes = 10
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
        self.x = tf.placeholder(dtype=tf.int32, shape=[None, None])
        batch_size = tf.shape(self.x)[0]
        max_steps = tf.shape(self.x)[1]
        self.xo = tf.one_hot(self.x, self.num_classes)
        self.y = tf.placeholder(dtype=tf.int32, shape=[None, None])
        self.yo = tf.one_hot(self.y, self.num_classes)
        self.lr = tf.placeholder(dtype=self.dtype, name="learning_rate")
        #mask
        self.mask = tf.cast(tf.tile(tf.reshape(tf.math.logical_not(tf.sequence_mask([max_steps-10],max_steps)),
                                               [1,max_steps]),[batch_size,1]),dtype=self.dtype)

        self.initial_state = self.cell.zero_state(batch_size=batch_size, dtype=self.dtype)

        output, last_state = tf.nn.dynamic_rnn(inputs=self.xo, cell=self.cell, dtype=self.dtype)

        output_flatten = tf.reshape(output,[-1,self.hidden])
        predict = tf.layers.dense(output_flatten,units=self.num_classes)
        self.logits = tf.reshape(predict,[batch_size,max_steps,self.num_classes])
        self.softmax_output = tf.nn.softmax(self.logits,axis=2)
        self.predict = tf.argmax(self.softmax_output,axis=2)

        self.logits_predict = tf.slice(self.logits,[0,max_steps - 10,0],[-1,-1,-1])

        self.predictions = tf.argmax(self.logits_predict,axis=1)

        self.labels = tf.slice(self.yo, [0, max_steps - 10, 0], [-1, -1, -1])
        self.labels_predict = tf.argmax(self.labels,axis=1)

        self.all_acc = tf.reduce_mean(tf.cast(tf.math.equal(self.labels_predict,self.predictions),tf.float32),axis=1)
        self.batch_acc = tf.reduce_mean(self.all_acc)

        self.cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels,logits=self.logits_predict)
        self.loss = tf.reduce_mean(tf.reduce_mean(self.cross_entropy_loss,axis=1),axis=0)
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.grads_vars = self.optimizer.compute_gradients(self.loss)
        self.train_op = self.optimizer.apply_gradients(self.grads_vars,
                                                           global_step=self.global_step)
    def step(self,sess,feed_dict):
        return sess.run([self.train_op,self.loss,self.global_step,self.batch_acc],feed_dict)




gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=.2)
tf.ConfigProto(gpu_options=gpu_options)

# device = '/cpu:0'

lr = args.learning_rate
seeds = 10
t_start = 10
t_end = 20
c = CopyTask(batch_size,t_start,t_end)
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
        e_acc = 0
        for b in range(0,t_end-t_start+1):
            # print(b)
            feed_dict = {model.x: xb, model.y: yb, model.lr:lr}
            _, b_loss,step,b_acc=model.step(sess,feed_dict)
            e_loss+=b_loss
            e_acc+=b_acc/(t_end-t_start)
            #next batch
            xb, yb = c.next_batch()
        if e%50==0:
            print('Epoch:{} Loss:{:1.3f}, Acc:{:1.3f}'.format(e,e_loss,e_acc))


    sess.close()