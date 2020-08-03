import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict


def classification_loss_func(prediction, true_labels, ce_temperature=1.0):
    celoss_criterion = nn.CrossEntropyLoss()
    return celoss_criterion(prediction / ce_temperature, true_labels)


def explicit_semantic_alignment_loss_func(source_learned_features, l_target_learned_features,
                                          u_target_learned_features, source_labels, l_target_labels,
                                          u_target_pseudo_labels, configuration, prototype):
    """
    class-level feature alignment: k-th class features of source, target, source-target,
    and calculate MSELOss between each pair
    :param prototype: how many prototypes used for general loss
    :param source_learned_features: source feature
    :param l_target_learned_features:  labeled target feature
    :param u_target_learned_features: unlabeled target feature
    :param source_labels: source groundtruth
    :param l_target_labels: label target groundtruth
    :param u_target_pseudo_labels: unlabeled target pseudo label
    :param configuration:
    :return:
    """
    class_number = configuration['class_number']
    mu_s = OrderedDict()
    mu_t = OrderedDict()

    if prototype == 'two':
        for i in range(class_number):
            mu_s[i] = []
            mu_t[i] = []

        assert source_learned_features.shape[0] == len(source_labels)
        for i in range(source_learned_features.shape[0]):
            mu_s[int(source_labels[i])].append(source_learned_features[i])

        assert l_target_learned_features.shape[0] == len(l_target_labels)
        for i in range(l_target_learned_features.shape[0]):
            mu_t[int(l_target_labels[i])].append(l_target_learned_features[i])

        assert u_target_learned_features.shape[0] == len(u_target_pseudo_labels)
        for i in range(u_target_learned_features.shape[0]):
            mu_t[int(u_target_pseudo_labels[i])].append(u_target_learned_features[i])

        error_general = 0
        mseloss_critein = nn.MSELoss(size_average=False)

        for i in range(class_number):
            mu_s[i] = torch.mean(torch.stack(mu_s[i], 0).float(), 0).float()

            mu_t[i] = torch.mean(torch.stack(mu_t[i], 0).float(), 0).float()

            error_general += mseloss_critein(mu_s[i], mu_t[i])

        return error_general

    elif prototype == 'three':
        mu_st = OrderedDict()

        for i in range(class_number):
            mu_s[i] = []
            mu_t[i] = []
            mu_st[i] = [[], []]

        assert source_learned_features.shape[0] == len(source_labels)
        for i in range(source_learned_features.shape[0]):
            mu_s[int(source_labels[i])].append(source_learned_features[i])
            mu_st[int(source_labels[i])][0].append(source_learned_features[i])

        assert l_target_learned_features.shape[0] == len(l_target_labels)
        for i in range(l_target_learned_features.shape[0]):
            mu_t[int(l_target_labels[i])].append(l_target_learned_features[i])
            mu_st[int(l_target_labels[i])][1].append(l_target_learned_features[i])

        assert u_target_learned_features.shape[0] == len(u_target_pseudo_labels)
        for i in range(u_target_learned_features.shape[0]):
            mu_t[int(u_target_pseudo_labels[i])].append(u_target_learned_features[i])
            mu_st[int(u_target_pseudo_labels[i])][1].append(u_target_learned_features[i])

        error_general = 0
        mseloss_critein = nn.MSELoss(size_average=False)

        for i in range(class_number):
            source_mean = torch.mean(torch.stack(mu_s[i], 0).float(), 0).float()

            target_mean = torch.mean(torch.stack(mu_t[i], 0).float(), 0).float()

            mu_st_numerator = 0
            mu_st_numerator += torch.sum(torch.stack(mu_st[i][0], 0).float(), 0).float()
            mu_st_numerator += torch.sum(torch.stack(mu_st[i][1], 0).float(), 0).float()
            source_target_mean = torch.div(mu_st_numerator, len(mu_st[i][0]) + len(mu_st[i][1]))

            error_general += mseloss_critein(source_mean, target_mean)
            error_general += mseloss_critein(source_mean, source_target_mean)
            error_general += mseloss_critein(target_mean, source_target_mean)

        return error_general


def knowledge_distillation_loss_func(source_predic, source_label, l_target_predic, l_target_label, args):
    """
        semantic-level alignment: source prediction, target prediction, source label, target label
        q: soft label for class k is the average over the softmax of all activations of source example in class k
        p: each labeled target smaple softmax output with temperature (T>1)
        :param args: temperature parameter
        :param source_predic: source output
        :param source_label:
        :param l_target_predic: labeled target output
        :param l_target_label: labeled target label
        :return: implicit semantic-level alignment loss
        """
    if args.alpha == 1.0:
        return classification_loss_func(l_target_predic, l_target_label), \
               torch.Tensor([0.])[0], torch.Tensor([0.])[0]

    assert source_predic.shape[1] == l_target_predic.shape[1]
    class_num = source_predic.shape[1]
    k_categories = torch.zeros((class_num, class_num))
    source_softmax = F.softmax(source_predic / args.temperature)
    l_target_softmax = F.softmax(l_target_predic)
    soft_loss = 0

    for k in range(class_num):
        k_source_softmax = source_softmax.index_select(dim=0, index=(source_label == k).nonzero().reshape(-1, ))
        k_categories[k] = torch.mean(k_source_softmax, dim=0)

    if torch.cuda.is_available():
        k_categories = k_categories.cuda()

    for k in range(class_num):
        k_l_target_softmax = l_target_softmax.index_select(dim=0, index=(l_target_label == k).nonzero().reshape(-1, ))
        soft_loss -= torch.mean(torch.sum(k_categories[k] * torch.log(k_l_target_softmax + 1e-5), 1))

    hard_loss = classification_loss_func(l_target_predic, l_target_label)
    loss = (1 - args.alpha) * hard_loss + args.alpha * soft_loss
    return loss, (1 - args.alpha) * hard_loss, args.alpha * soft_loss


def get_prototype_label(source_learned_features, l_target_learned_features, u_target_learned_features, source_labels,
                        l_target_labels, configuration, combine_pred, epoch):
    """
    get unlabeled target prototype label
    :param epoch: training epoch
    :param combine_pred: Euclidean, Cosine
    :param configuration: dataset configuration
    :param source_learned_features: source feature
    :param l_target_learned_features:  labeled target feature
    :param u_target_learned_features:  unlabeled target feature
    :param source_labels: source labels
    :param l_target_labels: labeled target labels
    :return: unlabeled target prototype label
    """
    def prototype_softmax(features, feature_centers):
        assert features.shape[1] == feature_centers.shape[1]
        n_samples = features.shape[0]
        C, dim = feature_centers.shape
        pred = torch.FloatTensor()
        for i in range(n_samples):
            if combine_pred.find('Euclidean') != -1:
                dis = -torch.sum(torch.pow(features[i].expand(C, dim) - feature_centers, 2), dim=1)
            elif combine_pred.find('Cosine') != -1:
                dis = torch.cosine_similarity(features[i].expand(C, dim), feature_centers)
            if not i:
                pred = dis.reshape(1, -1)
            else:
                pred = torch.cat((pred, dis.reshape(1, -1)), dim=0)
        return pred

    assert source_learned_features.shape[1] == u_target_learned_features.shape[1]
    class_num = configuration['class_number']
    feature_dim = source_learned_features.shape[1]
    feature_centers = torch.zeros((class_num, feature_dim))
    for k in range(class_num):
        # calculate feature center of each class for source and target
        k_source_feature = source_learned_features.index_select(dim=0,
                                                                index=(source_labels == k).nonzero().reshape(-1, ))
        k_l_target_feature = l_target_learned_features.index_select(dim=0, index=(
                l_target_labels == k).nonzero().reshape(-1, ))
        feature_centers[k] = torch.mean(torch.cat((k_source_feature, k_l_target_feature), dim=0), dim=0)

    if torch.cuda.is_available():
        feature_centers = feature_centers.cuda()

    # assign 'pseudo label' by Euclidean distance or Cosine similarity between feature and prototype,
    # select the most confident samples in each pseudo class, not confident label=-1
    prototype_pred = prototype_softmax(u_target_learned_features, feature_centers)
    prototype_value, prototype_label = torch.max(prototype_pred.data, 1)

    # add threshold
    if combine_pred.find('threshold') != -1:
        if combine_pred == 'Euclidean_threshold':
            # threshold for Euclidean distance
            select_threshold = 0.2
        elif combine_pred == 'Cosine_threshold':
            # Ref: Progressive Feature Alignment for Unsupervised Domain Adaptation CVPR2019
            select_threshold = 1. / (1 + np.exp(-0.8 * (epoch + 1))) - 0.01
            # select_threshold = 0.1
        prototype_label[(prototype_value < select_threshold).nonzero()] = -1

    return prototype_label
