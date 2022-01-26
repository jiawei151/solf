from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
# from torch.nn import init
# from torch.autograd import Variable
# from torchvision.models import resnet50, resnet34
import math
import os
import numpy as np
from .MetaModules import *
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = MetaConv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = MetaBatchNorm2d(planes)
        self.conv2 = MetaConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = MetaBatchNorm2d(planes)
        self.conv3 = MetaConv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = MetaBatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, remove_last_relu=False):
        super(BasicBlock, self).__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = MetaConv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = MetaBatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = MetaConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = MetaBatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.remove_last_relu = remove_last_relu
        
        self.act = OrderedDict()
        self.count = 0

    def forward(self, x):
        self.count = self.count % 2
        self.act['conv_{}'.format(self.count)] = x
        self.count += 1

        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        self.count = self.count % 2 
        self.act['conv_{}'.format(self.count)] = out
        self.count +=1

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        if not self.remove_last_relu:
            out = self.relu(out)

        return out

class MetaResNetBase(MetaModule):
    def __init__(self, block, layers, nf=64, zero_init_residual=True, dataset='cifar', start_class=0, remove_last_relu=False):
        super(MetaResNetBase, self).__init__()
        self.remove_last_relu = remove_last_relu
        self.inplanes = nf
        self.conv1 = MetaConv2d(3, nf, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = MetaBatchNorm2d(nf)
        self.relu = nn.ReLU(inplace=True)

        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 1*nf, layers[0])
        self.layer2 = self._make_layer(block, 2*nf, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 4*nf, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 8*nf, layers[3], stride=2, remove_last_relu=self.remove_last_relu)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = 8 * nf * block.expansion

        self.act = OrderedDict()

        for m in self.modules():

            if isinstance(m, MetaConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, MetaBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    # print(m.bn2.weight)
                    nn.init.constant_(m.bn2.weight, 0)
                    # print(m.bn2.weight)


    def _make_layer(self, block, planes, blocks, remove_last_relu=False, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                MetaConv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                MetaBatchNorm2d(planes * block.expansion),
            )

        layers = [
            block(self.inplanes, planes, stride, downsample)
        ]

        self.inplanes = planes * block.expansion
        if remove_last_relu:
            for i in range(1, blocks-1):
                layers.append(block(self.inplanes, planes))
            layers.append(block(self.inplanes, planes, remove_last_relu=True))
        else:    
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))
        
        return nn.Sequential(*layers)

    def reset_bn(self):
        for m in self.modules():
            if isinstance(m, MetaBatchNorm2d):
                m.reset_running_stats()

    def forward(self, x):

        self.act['conv_in'] = x

        x = self.conv1(x)
        x = self.bn1(x)
        # x = self.maxpool(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        self.act['linear'] = x
        return x

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    """
    model = MetaResNetBase(BasicBlock, [2, 2, 2, 2], **kwargs)

    return model

# class MetaResNet(MetaModule):
#     def __init_with_imagenet(self, baseModel):
#         model = resnet50(pretrained=True)
#         del model.fc
#         baseModel.copyWeight(model.state_dict())

#     def getBase(self):
#         baseModel = MetaResNetBase([3, 4, 6, 3])
#         self.__init_with_imagenet(baseModel)
#         return baseModel

#     def __init__(self, num_features=0, dropout=0, cut_at_pooling=False, norm=True, num_classes=[0,0,0], BNNeck=False):
#         super(MetaResNet, self).__init__()
#         self.num_features = num_features
#         self.dropout = dropout
#         self.cut_at_pooling = cut_at_pooling
#         self.num_classes1 = num_classes[0]
#         self.num_classes2 = num_classes[1]
#         self.num_classes3 = num_classes[2]
#         self.has_embedding = num_features > 0
#         self.norm = norm
#         self.BNNeck = BNNeck
#         if self.dropout > 0:
#             self.drop = nn.Dropout(self.dropout)
#         # Construct base (pretrained) resnet
#         self.base = self.getBase()
#         self.base.layer4[0].conv2.stride = (1, 1)
#         self.base.layer4[0].downsample[0].stride = (1, 1)
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         out_planes = 2048
#         if self.has_embedding:
#             self.feat = MetaLinear(out_planes, self.num_features)
#             init.kaiming_normal_(self.feat.weight, mode='fan_out')
#             init.constant_(self.feat.bias, 0)
#         else:
#             # Change the num_features to CNN output channels
#             self.num_features = out_planes

#         self.feat_bn = MixUpBatchNorm1d(self.num_features)
#         init.constant_(self.feat_bn.weight, 1)
#         init.constant_(self.feat_bn.bias, 0)

#     def forward(self, x, MTE='', save_index=0):
#         x= self.base(x)
#         x = self.gap(x)
#         x = x.view(x.size(0), -1)

#         if self.cut_at_pooling:
#             return x

#         if self.has_embedding:
#             bn_x = self.feat_bn(self.feat(x))
#         else:
#             bn_x = self.feat_bn(x, MTE, save_index)
#         tri_features = x

#         if self.training is False:
#             bn_x = F.normalize(bn_x)
#             return bn_x

#         if isinstance(bn_x, list):
#             output = []
#             for bnfeature in bn_x:
#                 if self.norm:
#                     bnfeature = F.normalize(bnfeature)
#                 output.append(bnfeature)
#             if self.BNNeck:
#                 return output, tri_features
#             else:
#                 return output

#         if self.norm:
#             bn_x = F.normalize(bn_x)
#         elif self.has_embedding:
#             bn_x = F.relu(bn_x)

#         if self.dropout > 0:
#             bn_x = self.drop(bn_x)

#         if self.BNNeck:
#             return bn_x, tri_features
#         else:
#             return bn_x


