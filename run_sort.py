
import numpy as np
import tensorflow as tf

from Sort import SortingDataWrapper




logdir = './logs/sort/{0}_{1}_seed{2}_clip{3}_hid{4}'


class Config(object):
    def __init__(self):
        self.learning_rate = 1e-3
        self.seeds = 5
        self.gpu = .1
        self.n_steps = 100
        self.optimizer = 'adam'
        self.hidden_units = 100
        self.dtype = 'float32'
        self.model = 'lstm'
        self.validate_freq = 50
        self.batch_size = 100
        self.min_t = 10
        self.max_t = 100
        self.clip_grad_norm = False
        if self.clip_grad_norm:
            self.max_norm_grad = 1.0
        else:
            self.max_norm_grad = None







class SortModel(object):
    def __init__(self,args):
        #self.seed = args.seed
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
        #self.train_step = self.optim.minimize(self.loss)
        self.grads_vars =self.optim.compute_gradients(self.loss)
        if args.clip_grad_norm:
            grads, variables = zip(*self.grads_vars)
            grads_clipped, _ = tf.clip_by_global_norm(grads, clip_norm=self.max_norm_gradient)
            self.train_step = self.optim.apply_gradients(zip(grads_clipped, variables))
        else:
            self.train_step = self.optim.apply_gradients(self.grads_vars)
        self.train_step = self.optim.minimize(self.loss)



print('building model..')


args = Config()
#model = SortModel(args)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu)

print('getting data..')

data = SortingDataWrapper(args.batch_size,args.min_t,args.max_t)



import datetime
tstart = datetime.datetime.now()


for seed in range(args.seeds):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    model = SortModel(args)
    print('begin training for {0} steps'.format(args.n_steps))

    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.initializers.global_variables())

    writer = tf.summary.FileWriter(logdir.format(args.model,args.learning_rate,seed,args.max_norm_grad,args.hidden_units), sess.graph)
    train_error_summary = tf.summary.scalar(name='Train_Error',tensor=model.loss_batch)
    #valid_accuracy_summary = tf.summary.scalar(name='Accuracy',tensor=model.accuracy)

    valid_accuracy_summaries = []
    for i in range(args.min_t,args.max_t+1):
        valid_accuracy_summaries.append(tf.summary.scalar(name='Accuracy_t_{0}'.format(i),tensor=model.accuracy))

    hist_loss = []
    patience = 100
    min_delta = 0.01
    patience_cnt = 0

    for step in range(args.n_steps):
        tx,ty,m = data.next_batch()
        _, e, summ = sess.run([model.train_step,model.loss_batch,train_error_summary],{model.X:tx,model.Y:ty,model.mask_output:m})
        writer.add_summary(summ,step)
        hist_loss.append(e)
        if step % args.validate_freq == 0:
            for t in range(args.min_t,args.max_t+1):
                vx,vy,m = data.validate(t)
                acc,summ = sess.run([model.accuracy,valid_accuracy_summaries[t-args.min_t]],{model.X:vx,model.Y:vy})
                writer.add_summary(summ,step)
            print('step {0} :: {1}, acc {2}'.format(step,round(e,4),round(acc,4),acc))
        if step>500:
            if hist_loss[-2] - hist_loss[-1] > min_delta:
                patience_cnt = 0
            else:
                patience_cnt += 1
            if patience_cnt > patience:
                print('early stopping')
                break

    tend = datetime.datetime.now() 
    print('finished in ',(tend-tstart))
    sess.close()














