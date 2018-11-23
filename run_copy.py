import tensorflow as tf
import argparse
import os
from Copy import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='lstm',
                 help='gru, or lstm')
parser.add_argument('--epochs', type=int, default=100000,
                 help='# of Epochs, default 100k')
parser.add_argument('--seeds', type=int, default=5,
                 help='trains total number of seeds')
parser.add_argument('--lr', type=float, default=1e-3 ,
                 help='learning rate, default 10e-5')
parser.add_argument('--optimizer', type=str, default='adam',
                 help='adam, or rmsprop')
parser.add_argument('--validate_freq', type=int, default=200,
                    help='validation freq')
parser.add_argument('--dtype', type=int, default=32,
                 help='specify dtype')
parser.add_argument('--layers', type=int, default=1,
                 help='layers')
parser.add_argument('--hidden', type=int, default=100,
                 help='hidden')
parser.add_argument('--batch_size', type=int, default=400,
                 help='batch_size')
parser.add_argument('--gpu', type=float, default=.04,
                 help='gpu memory')
parser.add_argument('--t_start', type=int, default=100,
                 help='t start')
parser.add_argument('--t_end', type=int, default=100,
                 help='t end')


args = parser.parse_args()
batch_size = args.batch_size


class Model(object):
    def __init__(self, args):
        self.rate = args.lr
        self.optim = args.optimizer
        self.epochs = args.epochs
        self.hidden = args.hidden
        self.layers = args.layers
        self.num_classes = 10
        self.batch_size = args.batch_size
        self.clip_grad_norm =True
        self.max_norm_gradient = 1.0
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        if args.dtype == 32:
            self.dtype = tf.float32
        else:
            self.dtype = tf.float64
        print('Using dtype tf.float{}'.format(self.dtype))

        if self.layers ==1:
            if args.model == 'lstm':
                self.cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units=self.hidden)
                # self.cell = tf.contrib.rnn.LSTMCell(num_units=self.hidden, dtype=self.dtype)
            else:
                self.cell = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(num_units=self.hidden)
                # self.cell = tf.contrib.rnn.GRUCell(num_units=self.hidden, dtype=self.dtype)
        else:
            if args.model == 'lstm':
                self.cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units=self.hidden)
                                                         for _ in range(self.layers)])
            else:
                self.cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(num_units=self.hidden)
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

        self.predictions = tf.cast(tf.argmax(self.logits_predict,axis=2),tf.int32)

        self.labels = tf.slice(self.y, [0, max_steps - 10], [-1, -1])
        self.labels_predict = tf.slice(self.yo,[0,max_steps - 10,0],[-1,-1,-1])

        self.all_acc = tf.reduce_mean(tf.cast(tf.math.equal(self.labels,self.predictions),tf.float32),axis=1)
        self.batch_acc = tf.reduce_mean(self.all_acc)

        self.cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels_predict,logits=self.logits_predict)
        self.loss = tf.reduce_mean(tf.reduce_mean(self.cross_entropy_loss,axis=1),axis=0)
        self.optim = tf.train.AdamOptimizer(self.rate)

        self.grads_vars = self.optim.compute_gradients(self.loss)
        if self.clip_grad_norm:
            grads, variables = zip(*self.grads_vars)
            grads_clipped, _ = tf.clip_by_global_norm(grads, clip_norm=self.max_norm_gradient)
            self.train_op = self.optim.apply_gradients(zip(grads_clipped, variables),global_step=self.global_step)
        else:
            self.train_op = self.optim.apply_gradients(self.grads_vars,global_step=self.global_step)


        # self.grads_vars = self.optimizer.compute_gradients(self.loss)
        # self.train_op = self.optimizer.apply_gradients(self.grads_vars,global_step=self.global_step)
                 
        ## summaries
        self.t_error=tf.summary.scalar(name='Train_Error',tensor=self.loss)
        self.t_acc = tf.summary.scalar(name='Train_Accuracy',tensor=self.batch_acc)

        self.v_error = tf.summary.scalar(name='Validation_Error', tensor=self.loss)
        # self.v_acc = tf.summary.scalar(name='Validation_Accuracy', tensor=self.batch_acc)
        self.valid_accuracy_summaries = []
        for i in range(args.t_start, args.t_end+1):
            self.valid_accuracy_summaries.append(tf.summary.scalar(name='Validation_Accuracy_t={0}'.format(i), tensor=self.batch_acc))
        self.valid_loss_summaries = []
        # for i in range(args.t_start, args.t_end+1):
        #     self.valid_loss_summaries.append(tf.summary.scalar(name='Validation_Accuracy_t={0}'.format(i), tensor=self.loss))


    def step(self,sess,feed_dict):
        return sess.run([self.train_op,self.loss,self.global_step,self.batch_acc,self.t_error,self.t_acc],feed_dict)

    def check(self,sess,feed_dict):
        return sess.run([self.loss,self.batch_acc,self.labels,self.predictions],feed_dict)


c = CopyTask(batch_size,args.t_start,args.t_end,seed=0)

for seed in range(args.seeds):
    # set seed
    tf.reset_default_graph()
    tf.set_random_seed(seed)

    model = Model(args)
    logdir = './logs/copy_replicate/{0}_{1}_{2}_ts{3}_te{4}'

    sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu)))
    init = tf.global_variables_initializer()
    writer = tf.summary.FileWriter(logdir.format(args.model, args.lr, seed, args.t_start, args.t_end),
                                   sess.graph)
    sess.run(init)
    # saver = tf.train.Saver()
    print('Setting seed....')

    Solved =False
    step = 0

    hist_loss = []
    patience = 100
    min_delta = 0.0001
    patience_cnt = 0
    early_stop=False

    while not Solved and step<=args.epochs and not early_stop:
        xb, yb = c.next_batch()
        feed_dict = {model.x: xb, model.y: yb}
        _, e_loss,step,e_acc,esumm,asumm=model.step(sess,feed_dict)
        writer.add_summary(esumm, step)
        writer.add_summary(asumm, step)

        #next batch
        if step%args.validate_freq==0:
            acc = 0
            t_loss = 0
            for t in range(args.t_start,args.t_end+1):
                xb, yb = c.next_batch()
                feed_dict = {model.x: xb, model.y: yb}
                summ = sess.run(model.valid_accuracy_summaries[t - args.t_start],
                                     {model.x: xb, model.y: yb})
                writer.add_summary(summ, step)

                e_loss,e_acc,y_act,y_pred = model.check(sess,feed_dict)

                # directory = modeldir.format(args.model,args.learning_rate,seed,t_start,t_end)
                # if not os.path.isdir(directory):
                #     os.makedirs(directory)
                # saver.save(sess, directory, global_step=step)
                acc+=e_acc/(args.t_end-args.t_start+1)
                t_loss+=e_loss/(args.t_end-args.t_start+1)

                if t==args.t_end:
                    print('act:{}'.format(y_act[0]))
                    print('pred:{}'.format(y_pred[0]))
                    print('Epoch:{} Loss:{:1.7f}, Acc:{:1.7f}, ACC{:1.7f}'.format(step,e_loss,e_acc,acc))
            print(round(float(acc),3),float(1))
            if round(float(acc),3)==float(1):
                # saver.save(sess, directory, global_step=step)
                print('Solved: Epoch:{} Loss:{:1.5f}, Acc:{:1.7f}'.format(step, e_loss, e_acc))
                Solved=True
            hist_loss.append(acc/(args.t_end-args.t_start+1))
            if step>50000:
                if hist_loss[-2] - hist_loss[-1] > min_delta:
                    patience_cnt = 0
                else:
                    patience_cnt +=1
                if patience_cnt>patience:
                    print('Early stopping')
                    early_stop=True



        step+=1
    # folder = logdir.format(args.model, args.lr, seed, args.t_start, args.t_end)
    # filename = 'Final_test.txt'
    # file = open(os.path.join(folder,filename))
    # for t in range(args.t_start, args.t_end + 1):
    #     xb, yb = c.next_batch()
    #     feed_dict = {model.x: xb, model.y: yb}
    #     summ = sess.run(model.valid_accuracy_summaries[t - args.t_start],
    #                     {model.x: xb, model.y: yb})
    #     writer.add_summary(summ, step)
    #
    #     e_loss, e_acc, y_act, y_pred = model.check(sess, feed_dict)


    sess.close()