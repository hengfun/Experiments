from Grammar import Grammar
import tensorflow as tf

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

if __name__ == '__main__':
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
    tf.set_random_seed(args.seed)

config     = Config_ab(args)
train_data = Grammar(config, gr='ab', phase='train')