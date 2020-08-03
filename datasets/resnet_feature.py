from torchvision import models, transforms, datasets
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import scipy.io as sio


class ResBase(nn.Module):
    r"""Constructs a feature extractor based on ResNet-50 model.
    remove the last layer
    """

    def __init__(self):
        super(ResBase, self).__init__()
        model_res50 = torchvision.models.resnet50(pretrained=True)
        self.conv1 = model_res50.conv1
        self.bn1 = model_res50.bn1
        self.relu = model_res50.relu
        self.maxpool = model_res50.maxpool
        self.layer1 = model_res50.layer1
        self.layer2 = model_res50.layer2
        self.layer3 = model_res50.layer3
        self.layer4 = model_res50.layer4
        self.avgpool = model_res50.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool,
                                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)
        self.__in_features = 2048

    def forward(self, x):
        """
        :param x: the input Tensor as [bs, 3, 224, 224]
        :return: 2048-dim feature
        """
        feature = self.feature_layers(x)
        feature = feature.view(feature.size(0), -1)
        return feature

    def output_num(self):
        return self.__in_features


# Resnet-50 model
resnet50 = ResBase()
if torch.cuda.is_available():
    resnet50 = resnet50.cuda()
IMAGE_PATH = '/data1/TL/data/office_caltech_10/'
DOMAINS = ['amazon', 'caltech', 'dslr', 'webcam']
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

for d in DOMAINS:
    print('start:', d)
    data_set = datasets.ImageFolder(os.path.join(IMAGE_PATH, d), data_transforms)
    dataset_size = len(data_set)
    data_loader = DataLoader(data_set, batch_size=128, shuffle=False, num_workers=4)
    flag = True
    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            print(i)
            inputs, labels = data
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()
            labels = labels + 1
            features = resnet50(inputs)
            if flag:
                all_features = features
                all_labels = labels
                flag = False
            else:
                all_features = torch.cat((all_features, features), 0)
                all_labels = torch.cat((all_labels, labels), 0)
    save_name = str(d) + '_resnet.mat'
    sio.savemat(save_name, {'features': all_features.cpu().numpy(), 'labels': all_labels.long().cpu().numpy()})

    print('finished', d)
