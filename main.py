import os
import time
import random
import datetime
import argparse
import warnings
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import data_loader
import numpy as np
import torch.nn as nn
from collections import defaultdict
from models import Prototypical, Discriminator
from loss import classification_loss_func, explicit_semantic_alignment_loss_func, knowledge_distillation_loss_func, \
    get_prototype_label
from utils import write_log_record, seed_everything, make_dirs

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(
    description='Simultaneous Semantic Alignment Network for Heterogeneous Domain Adaptation')
parser.add_argument('--source', type=str, default='amazon_surf', help='Source domain',
                    choices=['amazon_surf', 'amazon_decaf', 'amazon_resnet',
                             'webcam_surf', 'webcam_decaf', 'webcam_resnet',
                             'caltech_surf', 'caltech_decaf', 'caltech_resnet'])
parser.add_argument('--target', type=str, default='amazon_decaf', help='Target domain',
                    choices=['amazon_surf', 'amazon_decaf', 'amazon_resnet',
                             'webcam_surf', 'webcam_decaf', 'webcam_resnet',
                             'caltech_surf', 'caltech_decaf', 'caltech_resnet',
                             'dslr_decaf', 'dslr_resnet'])
parser.add_argument('--cuda', type=str, default='0', help='Cuda index number')
parser.add_argument('--nepoch', type=int, default=3000, help='Epoch amount')
parser.add_argument('--partition', type=int, default=20, help='Number of partition')
parser.add_argument('--prototype', type=str, default='three', choices=['two', 'three'],
                    help='how many prototypes used for domain and general alignment loss')
parser.add_argument('--layer', type=str, default='double', choices=['single', 'double'],
                    help='Structure of the projector network, single layer or double layers projector')
parser.add_argument('--d_common', type=int, default=256, help='Dimension of the common representation')
parser.add_argument('--optimizer', type=str, default='mSGD', choices=['SGD', 'mSGD', 'Adam'], help='optimizer options')
parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
parser.add_argument('--temperature', type=float, default=5.0, help='source softmax temperature')
parser.add_argument('--alpha', type=float, default=0.1,
                    help='Trade-off parameter in front of L_soft, set to 0.0 to turn it off'
                         'Weight the (1 - alpha) * hard CE loss and alpha * soft CE loss')
parser.add_argument('--beta', type=float, default=0.004, help='Trade-off parameter of L_ESA, set to 0 to turn off')
parser.add_argument('--gamma', type=float, default=0.1, help='Trade-off parameter of L_D, set to 0 to turn off')
parser.add_argument('--combine_pred', type=str, default='Cosine',
                    choices=['Euclidean', 'Cosine', 'Euclidean_threshold', 'Cosine_threshold', 'None'],
                    help='the way of prototype predictions Euclidean, Cosine, None(not use)')
parser.add_argument('--checkpoint_path', type=str, default='checkpoint', help='All records save path')
parser.add_argument('--seed', type=int, default=2020, help='seed for everything')

args = parser.parse_args()
args.time_string = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H-%M-%S')

if torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    if len(args.cuda) == 1:
        torch.cuda.set_device(int(args.cuda))

# seed for everything
seed_everything(args)
# make dirs
make_dirs(args)
print(str(args))


def test(model, configuration, srctar):
    model.eval()
    if srctar == 'source':
        loader = configuration['source_data']
        N = configuration['ns']
    elif srctar == 'labeled_target':
        loader = configuration['labeled_target_data']
        N = configuration['nl']
    elif srctar == 'unlabeled_target':
        loader = configuration['unlabeled_target_data']
        N = configuration['nu']
    else:
        raise Exception('Parameter srctar invalid! ')

    with torch.no_grad():
        feature, label = loader[0].float(), loader[1].reshape(-1, ).long()
        if torch.cuda.is_available():
            feature, label = feature.cuda(), label.cuda()
        classifier_output, _ = model(input_feature=feature)
        _, pred = torch.max(classifier_output.data, 1)
        n_correct = (pred == label).sum().item()
        acc = float(n_correct) / N * 100.

    return acc


def train(model, model_d, optimizer, optimizer_d, configuration):
    best_acc = -float('inf')

    # training
    for epoch in range(args.nepoch):

        start_time = time.time()
        model.train()
        model_d.train()
        optimizer.zero_grad()
        optimizer_d.zero_grad()

        # prepare data
        source_data = configuration['source_data']
        l_target_data = configuration['labeled_target_data']
        u_target_data = configuration['unlabeled_target_data']
        source_feature, source_label = source_data[0].float(), source_data[1].reshape(-1, ).long()
        l_target_feature, l_target_label = l_target_data[0].float(), l_target_data[1].reshape(-1, ).long()
        u_target_feature = u_target_data[0].float()
        if torch.cuda.is_available():
            source_feature, source_label = source_feature.cuda(), source_label.cuda()
            l_target_feature, l_target_label = l_target_feature.cuda(), l_target_label.cuda()
            u_target_feature = u_target_feature.cuda()

        # forward propagation
        source_output, source_learned_feature = model(input_feature=source_feature)
        l_target_output, l_target_learned_feature = model(input_feature=l_target_feature)
        u_target_output, u_target_learned_feature = model(input_feature=u_target_feature)
        _, u_target_pseudo_label = torch.max(u_target_output, 1)
        if args.combine_pred == 'None':
            u_target_selected_feature = u_target_learned_feature
            u_target_selected_label = u_target_pseudo_label
            if epoch % 10 == 0:
                n_correct = (u_target_pseudo_label.cpu() == u_target_data[1].reshape(-1, ).long()).sum().item()
                acc_nn = float(n_correct) / configuration['nu'] * 100.
                print('Pesudo acc: (NN)', acc_nn)
        elif args.combine_pred.find('Euclidean') != -1 or args.combine_pred.find('Cosine') != -1:
            # get unlabeled data label via prototype prediction & network prediction
            u_target_prototype_label = get_prototype_label(source_learned_features=source_learned_feature,
                                                           l_target_learned_features=l_target_learned_feature,
                                                           u_target_learned_features=u_target_learned_feature,
                                                           source_labels=source_label,
                                                           l_target_labels=l_target_label,
                                                           configuration=configuration,
                                                           combine_pred=args.combine_pred,
                                                           epoch=epoch)
            # select consistent examples
            u_target_selected_feature = u_target_learned_feature.index_select(dim=0, index=(
                    u_target_pseudo_label == u_target_prototype_label).nonzero().reshape(-1, ))
            u_target_selected_label = u_target_pseudo_label.index_select(dim=0, index=(
                    u_target_pseudo_label == u_target_prototype_label).nonzero().reshape(-1, ))

            if epoch % 10 == 0:
                print('shared predictions:', len(u_target_selected_label), '/', len(u_target_pseudo_label))
                n_correct = (u_target_prototype_label.cpu() == u_target_data[1].reshape(-1, ).long()).sum().item()
                acc_pro = float(n_correct) / configuration['nu'] * 100.
                print('Prototype acc: (pro)', acc_pro)

        # ========================source data loss============================
        # labeled source data
        # CrossEntropy loss
        error_overall = classification_loss_func(source_output, source_label)
        if epoch % 10 == 0:
            print('Use source CE loss: ', error_overall)

        # ========================alignment loss============================
        # Calculate implicit semantic alignment loss
        isa_loss, hard_loss, soft_loss = knowledge_distillation_loss_func(source_output, source_label,
                                                                          l_target_output, l_target_label, args)
        error_overall += isa_loss
        if epoch % 10 == 0:
            print('Use ISA loss: ', isa_loss, 'hard CE loss: ', hard_loss, 'soft CE loss: ', soft_loss)

        # Calculate global adversarial alignment loss
        if args.gamma:
            transfer_criterion = nn.BCELoss()
            alpha = 2. / (1. + np.exp(-10 * float(epoch / args.nepoch))) - 1
            domain_labels = torch.from_numpy(
                np.array([[1]] * configuration['ns'] + [[0]] * configuration['nt'])).float()
            if torch.cuda.is_available():
                domain_labels = domain_labels.cuda()
            discriminator_out = model_d(
                torch.cat((source_learned_feature, l_target_learned_feature, u_target_learned_feature), dim=0), alpha)
            domain_adv_alignment_loss = transfer_criterion(discriminator_out, domain_labels)
            error_overall += args.gamma * domain_adv_alignment_loss
            if epoch % 10 == 0:
                print('Use domain adversarial loss: ', args.gamma * domain_adv_alignment_loss)

        # Calculate explicit semantic alignment loss
        if args.beta:
            u_target_selected_label = u_target_selected_label.reshape(-1, )

            general_alignment_loss = explicit_semantic_alignment_loss_func(
                source_learned_features=source_learned_feature,
                l_target_learned_features=l_target_learned_feature,
                u_target_learned_features=u_target_selected_feature,
                source_labels=source_label,
                l_target_labels=l_target_label,
                u_target_pseudo_labels=u_target_selected_label,
                configuration=configuration,
                prototype=args.prototype)
            error_overall += args.beta * general_alignment_loss
            # general_align_list[epoch].append(general_alignment_loss.item())
            if epoch % 10 == 0:
                print('Use ESA loss:', args.beta * general_alignment_loss)

        # backward propagation
        error_overall.backward()
        optimizer.step()
        optimizer_d.step()

        # Testing Phase
        acc_src = test(model, configuration, 'source')
        acc_labeled_tar = test(model, configuration, 'labeled_target')
        acc_unlabeled_tar = test(model, configuration, 'unlabeled_target')
        end_time = time.time()
        print('ACC -> ', end='')
        print('Epoch: [{}/{}], {:.1f}s, Src acc: {:.4f}%, LTar acc: {:.4f}%, UTar acc: {:.4f}%'.format(
            epoch, args.nepoch, end_time - start_time, acc_src, acc_labeled_tar, acc_unlabeled_tar))

        if best_acc < acc_unlabeled_tar:
            best_acc = acc_unlabeled_tar
            best_text = args.source.ljust(10) + '-> ' + args.target.ljust(10) \
                        + ' The proposed model for HDA achieves current best accuracy. '
            print(best_text)
            if epoch >= 1000:
                print('need more epoch training')

    # end for max_epoch
    print('Best Test Accuracy: {:.4f}%'.format(best_acc))
    write_log_record(args, configuration, best_acc)
    return best_acc


if __name__ == '__main__':
    result = 0.
    for i in range(args.partition):
        configuration = data_loader.get_configuration(args)
        model = Prototypical(configuration['d_source'], configuration['d_target'], args.d_common,
                             configuration['class_number'], args.layer)
        model_D = Discriminator(args.d_common)
        if torch.cuda.is_available():
            model = model.cuda()
            model_D = model_D.cuda()
        if args.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=args.lr)
            optimizer_d = optim.SGD(model_D.parameters(), lr=args.lr)
        elif args.optimizer == 'mSGD':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                  weight_decay=0.001, nesterov=True)
            optimizer_d = optim.SGD(model_D.parameters(), lr=args.lr, momentum=0.9,
                                    weight_decay=0.001, nesterov=True)
        elif args.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
            optimizer_d = optim.Adam(model_D.parameters(), lr=args.lr, betas=(0.9, 0.99))

        result += train(model, model_D, optimizer, optimizer_d, configuration)

    with open(args.log_path, 'a') as fp:
        fp.write('PN_HDA: '
                 + '| src = ' + args.source.ljust(4)
                 + '| tar = ' + args.target.ljust(4)
                 + '| avg acc = ' + str('%.4f' % (result / args.partition)).ljust(4)
                 + '\n'
                 + str(args)
                 + '\n')
    # write to another avg txt
    with open(args.avg_path, 'a') as fp:
        fp.write('PN_HDA: '
                 + '| src = ' + args.source.ljust(4)
                 + '| tar = ' + args.target.ljust(4)
                 + '| avg acc = ' + str('%.4f' % (result / args.partition)).ljust(4)
                 + '\n'
                 + str(args)
                 + '\n')
    fp.close()
    print('Avg acc:', str('%.4f' % (result / args.partition)).ljust(4))
