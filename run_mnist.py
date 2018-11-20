
import numpy as np 
import tensorflow as tf 

from Mnist import MnistWrapper



n_steps = 10000


logdir = './logs/mnist/{0}_{1}_{2}_clip{3}'


class Config(object):
    def __init__(self):
        self.learning_rate = 1e-3
        self.seed = 1
        self.optimizer = 'adam'
        self.hidden_units = 100
        self.dtype = 'float32'
        self.model = 'lstm'
        self.validate_freq = 200
        self.batch_size = 32
        self.clip_grad_norm = True
        if self.clip_grad_norm:
            self.max_norm_grad = 5.0
        else:
            self.max_norm_grad = None


# gpu = args.gpu
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=.2)





class MNISTModel(object):
    def __init__(self,args):
        self.rate = args.learning_rate
        self.seed = args.seed
        self.optim = args.optimizer
        self.max_norm_gradient = args.max_norm_grad
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
        
        self.X = tf.placeholder(dtype=self.dtype,shape=[None,28*28,1])
        self.batch_size = tf.shape(self.X)[0]
        self.mask = self.x_flat = tf.reshape(self.X,[-1])

        self.Y = tf.placeholder(dtype=tf.int32,shape=[None])
        self.Y_onehot = tf.one_hot(self.Y,self.num_classes)

        self.sequence_length = tf.placeholder(dtype=self.dtype,name='sequence_lengths')

        self.lr = tf.placeholder(dtype=self.dtype,name='learning_rate')

        self.initial_state = self.cell.zero_state(batch_size=self.batch_size,dtype=self.dtype)
        
        output, last_state = tf.nn.dynamic_rnn(self.cell,self.X,dtype=self.dtype)
        self.last_hidden = output[:,-1,:]

        self.logits = tf.layers.dense(self.last_hidden,self.num_classes,name='final_dense')

        self.prediction = tf.cast(tf.argmax(self.logits,axis=1),tf.int32)
        #self.accuracy = tf.reduce_sum(tf.cast(tf.equal(self.Y,self.prediction),tf.float32)) / self.batch_size
        self.accuracy = tf.reduce_sum(tf.cast(tf.equal(self.Y,self.prediction),tf.int32)) / self.batch_size


        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Y_onehot,logits=self.logits)
        self.loss_batch = tf.reduce_mean(self.loss)

        if self.optim=='adam':
            self.optim = tf.train.AdamOptimizer(args.learning_rate)
        else:
           self.optim = tf.train.MomentumOptimizer(args.learning_rate,momentum=.99)
        self.grads_vars =self.optim.compute_gradients(self.loss)
        if args.clip_grad_norm:
            grads, variables = zip(*self.grads_vars)
            grads_clipped, _ = tf.clip_by_global_norm(grads, clip_norm=self.max_norm_gradient)
            self.train_op = self.optim.apply_gradients(zip(grads_clipped, variables))
        else:
            self.train_op = self.optimizer.apply_gradients(self.grads_vars)
        self.train_step = self.optim.minimize(self.loss)



print('building model..')


args = Config()
tf.set_random_seed(args.seed)
model = MNISTModel(args)


print('getting data..')

data = MnistWrapper(batch_size=args.batch_size)




print('begin training for {0} steps'.format(n_steps))

sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.initializers.global_variables())

writer = tf.summary.FileWriter(logdir.format(args.model,args.learning_rate,args.seed,args.max_norm_grad), sess.graph)
train_error_summary = tf.summary.scalar(name='Train_Error',tensor=model.loss_batch)
train_accuracy_summary = tf.summary.scalar(name='train_Accuracy',tensor=model.accuracy)
valid_error_summary = tf.summary.scalar(name='Validation_Error',tensor=model.loss_batch)
valid_accuracy_summary = tf.summary.scalar(name='Validation_Accuracy',tensor=model.accuracy)


for step in range(n_steps):
    tx,ty = data.next_batch()
    _, e,a, summ,asumm = sess.run([model.train_step,model.loss_batch,model.accuracy,train_error_summary,train_accuracy_summary],{model.X:tx,model.Y:ty})
    writer.add_summary(summ,step)
    writer.add_summary(asumm,step)

    if step % args.validate_freq == 0:
        vx, vy = data.validate()
        ve, vacc, esumm,asumm = sess.run([model.loss_batch,model.accuracy,valid_error_summary,valid_accuracy_summary],{model.X:vx,model.Y:vy})
        writer.add_summary(esumm,step)
        writer.add_summary(asumm,step)
        print('step {0} | e {1} a {2} | e {3} a {4}'.format(step,round(e,4),round(a,3),round(ve,4),round(vacc,3)))





