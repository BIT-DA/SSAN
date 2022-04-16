import torch
import os
import pickle
import numpy as np
import random
from matplotlib import pyplot as plt
from collections import defaultdict


# seed for everything
def seed_everything(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)


# Write Log Record
def write_log_record(args, configuration, best_acc):
    with open(args.log_path, 'a') as fp:
        fp.write('PN_HDA: '
                 + '| seed = ' + str(args.seed).ljust(4)
                 + '| src = ' + args.source.ljust(4)
                 + '| tar = ' + args.target.ljust(4)
                 + '| best tar acc = ' + str('%.4f' % best_acc).ljust(4)
                 + '| nepoch = ' + str(args.nepoch).ljust(4)
                 + '| layer =' + str(args.layer).ljust(4)
                 + '| d_common =' + str(args.d_common).ljust(4)
                 + '| optimizer =' + str(args.optimizer).ljust(4)
                 + '| lr = ' + str(args.lr).ljust(4)
                 + '| temperature =' + str(args.temperature).ljust(4)
                 + '| alpha =' + str(args.alpha).ljust(4)
                 + '| beta = ' + str(args.beta).ljust(4)
                 + '| gamma = ' + str(args.gamma).ljust(4)
                 + '| time = ' + args.time_string
                 + '| checkpoint_path = ' + str(args.checkpoint_path)
                 + '\n')
    fp.close()


# Command Line Argument Bool Helper
def bool_string(input_string):
    if input_string.lower() not in ['true', 'false']:
        raise ValueError('Bool String Input Invalid! ')
    return input_string.lower() == 'true'


#  make dirs for model_path, result_path, log_path, diagram_path
def make_dirs(args):
    save_name = '_'.join([args.source.lower(), args.target.lower()])
    log_path = os.path.join(args.checkpoint_path, 'logs')
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(log_path)
        print('Makedir: ' + str(log_path))
    args.log_path = os.path.join(log_path, save_name + '.txt')
    args.avg_path = os.path.join(log_path, save_name + '_avg.txt')
