import torch.nn as nn
from torch.autograd import Function


# HDA Feature Projector
class Projector(nn.Module):
    def __init__(self, d_input, d_common, layer):
        super(Projector, self).__init__()
        if layer.lower() == "single":
            layer = nn.Linear(d_input, d_common)
            leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

            # init weight and bias
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.normal_(layer.bias, std=0.01)

            projector = nn.Sequential(layer, leaky_relu)
        elif layer.lower() == "double":
            d_intermediate = int((d_input + d_common) / 2)
            layer1 = nn.Linear(d_input, d_intermediate)
            leaky_relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
            layer2 = nn.Linear(d_intermediate, d_common)
            leaky_relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

            # init weight and bias
            nn.init.normal_(layer1.weight, std=0.01)
            nn.init.normal_(layer1.bias, std=0.01)
            nn.init.normal_(layer2.weight, std=0.01)
            nn.init.normal_(layer2.bias, std=0.01)

            projector = nn.Sequential(layer1, leaky_relu1, layer2, leaky_relu2)
        else:
            raise Exception("Input layer invalid! ")
        self.projector = projector

    def forward(self, x):
        return nn.functional.normalize(self.projector(x), dim=1, p=2)


# Label Classifier
class Classifier(nn.Module):
    def __init__(self, d_common, class_number):
        super(Classifier, self).__init__()
        layer = nn.Linear(d_common, class_number)
        leakey_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # init weight and bias
        nn.init.normal_(layer.weight, std=0.01)
        nn.init.normal_(layer.bias, std=0.01)

        self.class_classifier = nn.Sequential(layer, leakey_relu)

    def forward(self, x):
        return self.class_classifier(x)


class ReverseLayerF(Function):
    r"""Gradient Reverse Layer(Unsupervised Domain Adaptation by Backpropagation)
    Definition: During the forward propagation, GRL acts as an identity transform. During the back propagation though,
    GRL takes the gradient from the subsequent level, multiplies it by -alpha  and pass it to the preceding layer.
    Args:
        x (Tensor): the input tensor
        alpha (float): \alpha =  \frac{2}{1+\exp^{-\gamma \cdot p}}-1 (\gamma =10)
        out (Tensor): the same output tensor as x
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


# Domain Discriminator
class Discriminator(nn.Module):
    def __init__(self, d_common):
        super(Discriminator, self).__init__()
        layer = nn.Linear(d_common, 1)
        sigmod = nn.Sigmoid()

        # init weight and bias
        nn.init.normal_(layer.weight, std=0.01)
        nn.init.normal_(layer.bias, std=0.01)

        self.discriminator = nn.Sequential(layer, sigmod)

    def forward(self, x, alpha):
        x = ReverseLayerF.apply(x, alpha)
        x = self.discriminator(x)

        return x


# Prototypical Network
class Prototypical(nn.Module):
    def __init__(self, d_source, d_target, d_common, class_number, layer):
        super(Prototypical, self).__init__()
        self.d_common = d_common
        self.d_source = d_source
        self.d_target = d_target
        self.class_number = class_number
        self.layer = layer
        self.projector_source = Projector(self.d_source, self.d_common, self.layer)
        self.projector_target = Projector(self.d_target, self.d_common, self.layer)
        self.classifier = Classifier(self.d_common, self.class_number)

    def forward(self, input_feature):
        if input_feature.shape[1] == self.d_source:
            feature = self.projector_source(input_feature)
        elif input_feature.shape[1] == self.d_target:
            feature = self.projector_target(input_feature)
        else:
            raise Exception("Input data wrong dimension! ")
        feature = feature.view(-1, self.d_common)
        classifier_output = self.classifier(feature)
        return classifier_output, feature
