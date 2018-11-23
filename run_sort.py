
import numpy as np
import tensorflow as tf

from Sort import SortingDataWrapper


n_steps = 10000


logdir = './logs/sort/{0}_{1}_{2}'


class Config(object):
    def __init__(self):
        self.learning_rate = 2e-3
        self.seed = 1
        self.optimizer = 'adam'
        self.hidden_units = 100
        self.dtype = 'float32'
        self.model = 'lstm'
        self.validate_freq = 100
        self.batch_size = 20 #100
        self.min_t = 2
        self.max_t = 15


# gpu = args.gpu
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=.2)





class SortModel(object):
    def __init__(self,args):
        self.seed = args.seed
        self.optim = args.optimizer
        self.min_T = args.min_t
        self.max_T = args.max_t

        self.hidden_units = args.hidden_units

        self.num_classes = 10
        if args.dtype == tf.float64:
            self.dtype = tf.float64
        else:
            self.dtype = tf.float32

        if args.model == 'lstm':
            self.cell = tf.contrib.rnn.LSTMCell(num_units=self.hidden_units)
            # self.cell = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=1,num_units=self.hidden_units)
        elif args.model == 'gru':
            self.cell = tf.contrib.rnn.GRUCell(num_units=self.hidden_units)

        self.X = tf.placeholder(dtype=self.dtype,shape=[None,None,2])
        self.batch_size = tf.shape(self.X)[0]
        self.sequence_length = tf.shape(self.X)[1]
        self.mask_output = tf.placeholder(dtype=self.dtype,shape=[None,None])

        self.cur_T = tf.placeholder(dtype=tf.int32)

        self.Y = tf.placeholder(dtype=tf.int32,shape=[None,None])
        self.Y_onehot = tf.one_hot(self.Y,self.max_T)

        self.initial_state = self.cell.zero_state(batch_size=self.batch_size,dtype=self.dtype)

        self.output, last_state = tf.nn.dynamic_rnn(inputs=self.X,cell=self.cell,dtype=self.dtype)

        self.output_flat = tf.reshape(self.output, [-1,self.hidden_units])

        self.logits = tf.layers.dense(self.output_flat,self.max_T,name='final_dense')

        #self.prediction = tf.reshape(tf.cast(tf.argmax(self.logits,axis=2),tf.int32),[self.batch_size,-1])
        #self.accuracy = tf.reduce_sum(tf.cast(tf.equal(self.Y,self.prediction),tf.float32)) / self.batch_size
        #self.accuracy = tf.reduce_sum(tf.cast(tf.equal(self.Y,self.prediction),tf.int32)) / (self.batch_size*self.sequence_length)
        self.prediction = tf.cast(tf.argmax(tf.reshape(self.logits,[self.batch_size,-1,self.max_T]),axis=2),tf.int32)
        self.accuracy = tf.reduce_mean(tf.cast(tf.cast(tf.equal(self.Y[:,self.sequence_length//2:],self.prediction[:,self.sequence_length//2:]),tf.int32),tf.float32))

        #self.Y_oh_flat = tf.reshape()
        self.loss_ = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.reshape(self.Y_onehot,[-1,self.max_T]),logits=self.logits)
        self.loss = self.loss_ * tf.reshape(self.mask_output,[-1])

        self.loss_batch = tf.reduce_mean(self.loss)

        if self.optim=='adam':
            self.optim = tf.train.AdamOptimizer(args.learning_rate)
        else:
            self.optim = tf.train.MomentumOptimizer(args.learning_rate,momentum=.99)
        self.train_step = self.optim.minimize(self.loss)



print('building model..')


args = Config()
tf.set_random_seed(args.seed)
model = SortModel(args)


print('getting data..')

data = SortingDataWrapper(args.batch_size,args.min_t,args.max_t)




print('begin training for {0} steps'.format(n_steps))

sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.initializers.global_variables())

writer = tf.summary.FileWriter(logdir.format(args.model,args.learning_rate,args.seed), sess.graph)
train_error_summary = tf.summary.scalar(name='Train_Error',tensor=model.loss_batch)
#valid_accuracy_summary = tf.summary.scalar(name='Accuracy',tensor=model.accuracy)

valid_accuracy_summaries = []
for i in range(args.min_t,args.max_t+1):
    valid_accuracy_summaries.append(tf.summary.scalar(name='Accuracy_t={0}'.format(i),tensor=model.accuracy))


for step in range(n_steps):
    tx,ty,m = data.next_batch()
    _, e, summ = sess.run([model.train_step,model.loss_batch,train_error_summary],{model.X:tx,model.Y:ty,model.mask_output:m})
    writer.add_summary(summ,step)

    if step % args.validate_freq == 0:
        for t in range(args.min_t,args.max_t+1):
            vx,vy,m = data.validate(t)
            acc,summ = sess.run([model.accuracy,valid_accuracy_summaries[t-args.min_t]],{model.X:vx,model.Y:vy})
            writer.add_summary(summ,step)
        print('step {0} :: {1}, acc {2}'.format(step,round(e,4),round(acc,4),acc))














