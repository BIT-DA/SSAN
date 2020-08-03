import torch
import os.path as osp
import scipy.io as sio
import numpy as np
from sklearn import preprocessing

IMAGE2IMAGE_PATH = 'datasets/ImageToImageObjectRecognition/'

# image datasets
DATASETS = {
    'amazon_surf': osp.join(IMAGE2IMAGE_PATH, 'amazon_surf.mat'),
    'amazon_decaf': osp.join(IMAGE2IMAGE_PATH, 'amazon_decaf.mat'),
    'amazon_resnet': osp.join(IMAGE2IMAGE_PATH, 'amazon_resnet.mat'),
    'dslr_surf': osp.join(IMAGE2IMAGE_PATH, 'dslr_surf.mat'),
    'dslr_decaf': osp.join(IMAGE2IMAGE_PATH, 'dslr_decaf.mat'),
    'dslr_resnet': osp.join(IMAGE2IMAGE_PATH, 'dslr_resnet.mat'),
    'caltech_surf': osp.join(IMAGE2IMAGE_PATH, 'caltech_surf.mat'),
    'caltech_decaf': osp.join(IMAGE2IMAGE_PATH, 'caltech_decaf.mat'),
    'caltech_resnet': osp.join(IMAGE2IMAGE_PATH, 'caltech_resnet.mat'),
    'webcam_surf': osp.join(IMAGE2IMAGE_PATH, 'webcam_surf.mat'),
    'webcam_decaf': osp.join(IMAGE2IMAGE_PATH, 'webcam_decaf.mat'),
    'webcam_resnet': osp.join(IMAGE2IMAGE_PATH, 'webcam_resnet.mat'),
}


def get_configuration(args):
    SOURCE_PATH = DATASETS[args.source.lower()]
    TARGET_PATH = DATASETS[args.target.lower()]

    # source and target domain infos
    print('========= Source & Target Info =========')
    print('Source Domain: ' + SOURCE_PATH)
    print('Target Domain: ' + TARGET_PATH)
    print('========= Loading Data =========')
    source = sio.loadmat(SOURCE_PATH)
    target = sio.loadmat(TARGET_PATH)
    print('========= Loading Data Completed =========')
    print()
    print('========= Data Information =========')

    # Amount of labeled target instances for each class
    if args.target.lower() == 'spanish20':
        labeled_amount = 20
    elif args.target.lower() == 'spanish15':
        labeled_amount = 15
    elif args.target.lower() == 'spanish10':
        labeled_amount = 10
    elif args.target.lower() == 'spanish5':
        labeled_amount = 5
    else:
        labeled_amount = 3

    xs = source['features']
    xs = preprocessing.normalize(xs, norm='l2')
    xs_label = source['labels'] - 1  # Label range: 0 - 9 both inclusive
    print('xs.shape = ', xs.shape)
    print('xs_label.shape = ', xs_label.shape)

    entire_t = target['features']
    entire_t = preprocessing.normalize(entire_t, norm='l2')
    entire_t_label = target['labels'] - 1

    print('xt.shape = ', entire_t.shape)
    print('xt_label.shape = ', entire_t_label.shape)
    print('xt_label.len = ', len(entire_t_label))

    assert len(np.unique(xs_label)) == len(np.unique(entire_t_label))
    class_number = len(np.unique(xs_label))  # number of classes

    xl = []
    xl_label = []

    for cls in range(class_number):
        amount = labeled_amount
        while amount > 0:
            random_index = np.random.randint(0, entire_t.shape[0])
            if entire_t_label[random_index] == cls:
                xl.append(entire_t[random_index])
                xl_label.append(entire_t_label[random_index])
                amount -= 1
                entire_t = np.delete(entire_t, random_index, 0)
                entire_t_label = np.delete(entire_t_label, random_index, 0)
    xl = np.array(xl)
    xl_label = np.array(xl_label)
    xu = entire_t
    xu_label = entire_t_label

    ns, ds = xs.shape  # ns = number of source instances, ds = dimension of source instances
    nl, dt = xl.shape  # nl = number of labeled target instances, ds = dimension of all target instances
    nu, _ = xu.shape
    nt = nl + nu  # total amount of target instances
    print('ns = ', ns)
    print('nl = ', nl)
    print('nu = ', nu)
    print('ds = ', ds)
    print('dt = ', dt)
    print('Class_number: ', class_number)
    print()

    # Generate dataset objects
    source_data = [torch.from_numpy(xs), torch.from_numpy(xs_label)]
    labeled_target_data = [torch.from_numpy(xl), torch.from_numpy(xl_label)]
    unlabeled_target_data = [torch.from_numpy(xu), torch.from_numpy(xu_label)]

    # Data Allocation In Each Batch
    print('Number of Source Instances: ' + str(ns))
    print('Number of Labeled Target Instances: ' + str(nl))
    print('Number of Unlabeled Target Instances: ' + str(nu))
    print()

    # data configurations
    configuration = {'ns': ns, 'nl': nl, 'nu': nu, 'nt': nt, 'class_number': class_number,
                     'labeled_amount': labeled_amount, 'd_source': ds, 'd_target': dt,
                     'source_data': source_data, 'labeled_target_data': labeled_target_data,
                     'unlabeled_target_data': unlabeled_target_data}

    print('========= Loading Done =========')
    print()
    print('========= Training Started =========')
    return configuration
