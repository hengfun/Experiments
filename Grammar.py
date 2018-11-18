import numpy as np
import random
import tensorflow as tf

class Grammar(object):

    def __init__(self, config, gr, phase, pred_len=3):
        self.data = []
        self.gr = gr
        self.num_classes = config.num_classes
        if phase == 'train':
            self.max_length = config.max_n_train
            self.max_length_seq = config.max_length_seq_train
            self.num_str = config.num_str_train
        elif phase == 'test':
            self.max_length = config.max_n_test
            self.max_length_seq = config.max_length_seq_test
            self.num_str = config.num_str_test
        # elif phase == 'predict':
        #     self.max_length = config.max_n_test
        #     self.max_length_seq = config.max_length_seq_test
        #     self.num_str = 1
        #     self.data.append('S' + 'a' *
        #                      pred_len + 'b' * pred_len)

        random.seed(config.seed)
        if phase == 'train' or 'test':
            self.generate_grammar(phase)

        self.input, self.target = self.encode()

        # if phase == 'predict':
        #     del(self.data[0])

        self.data_len = len(self.data)

    def generate_grammar(self, phase):
        if self.gr == 'ab':
            for idx in range(self.num_str):
                length_a = np.mod(idx, self.max_length + 1)
                self.data.append('S' + 'a' *
                                 length_a + 'b' * length_a)

        elif self.gr == 'abba':
            length_b = 0
            for idx in range(self.num_str):
                length_a = np.mod(idx, self.max_length) + 1
                if length_a == 1:
                    length_b = np.mod(length_b, self.max_length) + 1
                    if length_b == 1:
                        self.data.append('S')
                self.data.append('S' + 'a' * length_a + 'b' *
                                 length_b + 'B' * length_b + 'A' *
                                 length_a)
                # Generate just num_str data
                self.data = self.data[0:self.num_str]

        elif self.gr == 'abc':
            for idx in range(self.num_str):
                length_a = np.mod(idx, self.max_length + 1)
                self.data.append('S' + 'a' * length_a + 'b' *
                                 length_a + 'c' * length_a)

            if phase == 'train':
                random.shuffle(self.data)
        return


    def encode(self):
        input = []
        target = []

        if self.gr == 'ab':
            for string in self.data:
                input_str = []
                target_str = []
                for elem in string:
                    if elem == 'S':
                        input_str.append([1, 0, 0])
                        target_str.append([1, -1, 1])
                    elif elem == 'a':
                        input_str.append([0, 1, 0])
                        target_str.append([1, 1, -1])
                    elif elem == 'b':
                        input_str.append([0, 0, 1])
                        target_str.append([-1, 1, -1])
                    else:
                        print("Invalid string")
                if len(string) != 1:
                    target_str[-1] = [-1, -1, 1]
                input.append(input_str)
                target.append(target_str)

        elif self.gr == 'abba':
            for string in self.data:
                input_str = []
                target_str = []
                cont = 0
                for elem in string:
                    if elem == 'S':
                        input_str.append([1, 0, 0, 0, 0])
                        target_str.append([1, -1, -1, -1, 1])
                    elif elem == 'a':
                        input_str.append([0, 1, 0, 0, 0])
                        target_str.append([1, 1, -1, -1, -1])
                        cont = cont + 1
                    elif elem == 'b':
                        input_str.append([0, 0, 1, 0, 0])
                        target_str.append([-1, 1, 1, -1, -1])
                    elif elem == 'B':
                        input_str.append([0, 0, 0, 1, 0])
                        target_str.append([-1, -1, 1, -1, -1])
                    elif elem == 'A':
                        input_str.append([0, 0, 0, 0, 1])
                        target_str.append([-1, -1, -1, 1, -1])
                    else:
                        print("Invalid string")
                if len(string) != 1:
                    target_str[-1] = [-1, -1, -1, -1, 1]
                    target_str[-(cont+1)] = [-1, -1, -1, 1, -1]
                input.append(input_str)
                target.append(target_str)

        elif self.gr == 'abc':
            for string in self.data:
                input_str = []
                target_str = []
                cont = 0
                for elem in string:
                    if elem == 'S':
                        input_str.append([1, 0, 0, 0])
                        target_str.append([1, -1, -1, 1])
                    elif elem == 'a':
                        input_str.append([0, 1, 0, 0])
                        target_str.append([1, 1, -1, -1])
                        cont = cont + 1
                    elif elem == 'b':
                        input_str.append([0, 0, 1, 0])
                        target_str.append([-1, 1, -1, -1])
                    elif elem == 'c':
                        input_str.append([0, 0, 0, 1])
                        target_str.append([-1, -1, 1, -1])
                    else:
                        print("Invalid string")
                if len(string) != 1:
                    target_str[-1] = [-1, -1, -1, 1]
                    target_str[-(cont+1)] = [-1, -1, 1, -1]
                input.append(input_str)
                target.append(target_str)


        return input, target

    def length(self, batch):
        return list(map(lambda t: len(t), batch))


    def next_batch(self, start_idx, batch_size):
        end_idx = start_idx + batch_size
        rest = end_idx - self.data_len
        rest = batch_size if rest > batch_size else rest

        if rest <= 0:
            batch_x = self.input[start_idx:end_idx]
            batch_y = self.target[start_idx:end_idx]
            start_idx = end_idx
        else:
            batch_x = self.input[start_idx:start_idx+rest]
            batch_y = self.target[start_idx:start_idx+rest]
            start_idx = start_idx + rest
        return self.pad(batch_x, batch_size),\
               self.pad(batch_y, batch_size), start_idx

    def pad(self, batch, batch_size):
        max_len = max(self.length(batch))
        padded = np.zeros((batch_size, max_len, self.num_classes),
                          dtype=int)
        for string in range(batch_size):
            for elem in range(len(batch[string])):
                padded[string, elem, :] = batch[string][elem]


        return padded

class Config_ab(object):

    def __init__(self, args):
        self.num_str_train        = args.num_str_train
        self.max_n_train          = args.max_n_train
        self.num_str_test         = args.num_str_test
        self.max_n_test           = args.num_str_test
        self.max_length_seq_train = 2 * self.max_n_train + 1
        self.max_length_seq_test  = 2 * self.max_n_test + 1
        self.num_classes          = 3
        self.batch_size           = 20
        self.hidden               = 1
        self.momentum             = args.momentum
        self.init_scale           = args.init_scale
        self.input_bias           = args.input_bias
        self.output_bias          = args.output_bias
        self.forget_bias          = args.forget_bias
        self.max_max_epoch        = args.max_max_epoch
        self.learning_rate        = args.learning_rate
        self.optimizer            = args.optimizer
        self.peephole             = args.peephole
        self.VERBOSE              = args.VERBOSE
        self.save_graph           = 'true'
        self.seed                 = args.seed
        self.name                 = args.model


class Config_abba(object):

    def __init__(self, args):
        self.num_str_train        = args.num_str_train
        self.max_n_train          = args.max_n_train
        self.num_str_test         = args.num_str_test
        self.max_n_test           = args.num_str_test
        self.max_length_seq_train = 2 * self.max_n_train + 1
        self.max_length_seq_test  = 2 * self.max_n_test + 1
        self.num_classes          = 5
        self.batch_size           = args.batch_size
        self.hidden               = args.hidden
        self.momentum             = args.momentum
        self.init_scale           = args.init_scale
        self.input_bias           = args.input_bias
        self.output_bias          = args.output_bias
        self.forget_bias          = args.forget_bias
        self.max_max_epoch        = args.max_max_epoch
        self.learning_rate        = args.learning_rate
        self.optimizer            = args.optimizer
        self.peephole             = args.peephole
        self.VERBOSE              = args.VERBOSE
        self.save_graph           = 'true'
        self.seed                 = args.seed
        self.name                 = args.model


class Config_abc(object):

    def __init__(self, args):
        self.num_str_train        = args.num_str_train
        self.max_n_train          = args.max_n_train
        self.num_str_test         = args.num_str_test
        self.max_n_test           = args.num_str_test
        self.max_length_seq_train = 3 * self.max_n_train
        self.max_length_seq_test  = 3 * self.max_n_test
        self.num_classes          = 4
        self.batch_size           = args.batch_size
        self.hidden               = args.hidden
        self.momentum             = args.momentum
        self.init_scale           = args.init_scale
        self.input_bias           = args.input_bias
        self.output_bias          = args.output_bias
        self.forget_bias          = args.forget_bias
        self.max_max_epoch        = args.max_max_epoch
        self.learning_rate        = args.learning_rate
        self.optimizer            = args.optimizer
        self.peephole             = args.peephole
        self.VERBOSE              = args.VERBOSE
        self.save_graph           = 'true'
        self.seed                 = args.seed
        self.name                 = args.model


import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--path',          help='save here',        type=str,   default='temp')
parser.add_argument('--task',          help='class of problem', type=str,   default='abc')
parser.add_argument('--model',         help='model to use',     type=str,   default='lstm')
parser.add_argument('--seed',          help='RNG seed',         type=int,   default=7193)
parser.add_argument('--mode',          help='seq/parallel',     type=str,   default='sequential')
parser.add_argument('--save',          help='save log',         type=str,   default='true')
parser.add_argument('--num_str_train', help='num_str_train',    type=int,   default=1000)
parser.add_argument('--max_n_train',   help='max_n_train',      type=int,   default=10)
parser.add_argument('--num_str_test',  help='num_str_test',     type=int,   default=1001)
parser.add_argument('--batch_size',    help='batch_size',       type=int,   default=50)
parser.add_argument('--hidden',        help='number of cells',  type=int,   default=4)
parser.add_argument('--momentum',      help='momentum',         type=float, default=.99)
parser.add_argument('--init_scale',    help='init_scale',       type=float, default=.1)
parser.add_argument('--input_bias',    help='input_bias',       type=float, default=-1.0)
parser.add_argument('--output_bias',   help='output_bias',      type=float, default=-2.0)
parser.add_argument('--forget_bias',   help='forget_bias',      type=float, default=2.0)
parser.add_argument('--max_max_epoch', help='number of epochs', type=int,   default=30)
parser.add_argument('--learning_rate', help='learning_rate',    type=float, default=8e-3)
parser.add_argument('--optimizer',     help='optimizer to use', type=str,   default='adam')
parser.add_argument('--peephole',      help='use peephole',     type=bool,  default=True)
parser.add_argument('--VERBOSE',       help='print output',     type=bool,  default=True)

args = parser.parse_args()

# Set random seed

config     = Config_ab(args)
train_data = Grammar(config, gr='ab', phase='train')
