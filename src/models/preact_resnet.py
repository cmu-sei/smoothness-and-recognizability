"""
ON THE HUMAN-RECOGNIZABILITY PHENOMENON OF ADVERSARIALLY TRAINED DEEP IMAGE CLASSIFIERS

Copyright 2020 Carnegie Mellon University.

NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE 
MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO 
WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, 
BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, 
EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON 
UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM 
PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.

Released under a MIT (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.

[DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  
Please see Copyright notice for non-US Government use and distribution.

Carnegie Mellon® is registered in the U.S. Patent and Trademark Office by Carnegie Mellon University.
This Software includes and/or makes use of the following Third-Party Software subject to its own license:

1. Python (https://docs.python.org/3/license.html#psf-license-agreement-for-python-release) Copyright 2001-2020 
Python Software Foundation 2001-2020.

2. PyTorch (https://github.com/pytorch/pytorch/blob/master/LICENSE#L3-L11) Copyright 2016 Facebook Inc.

3. Torchvision (https://github.com/pytorch/vision/blob/master/LICENSE) Copyright 2016 Soumith Chintala.

4. NumPy (https://github.com/numpy/numpy/blob/master/LICENSE.txt) Copyright 2005-2020 NumPy Developers.

5. tqdm (https://github.com/tqdm/tqdm/blob/master/LICENCE) Copyright noamraph 2013.

6. Jupyter (https://github.com/jupyter/notebook/blob/master/LICENSE) Copyright IPython Development Team 
2001-2015, Jupyter Development Team 2015-2020 IPython Development Team 2001-2015, Jupyter Development 
Team 2015-2020.

DM20-1153
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActBlock(nn.Module):
    """Pre-activation version of the BasicBlock
    Adapted from: https://github.com/locuslab/fast_adversarial
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.activation = nn.ReLU()

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        # out = F.relu(self.bn1(x))
        out = self.activation(self.bn1(x))
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        # out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv2(self.activation(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    """Pre-activation version of the original Bottleneck 
    Adapted from: https://github.com/locuslab/fast_adversarial
    """
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.activation = nn.ReLU()

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        # out = F.relu(self.bn1(x))
        out = self.activation(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        # out = self.conv2(F.relu(self.bn2(out)))
        # out = self.conv3(F.relu(self.bn3(out)))
        out = self.conv2(self.activation(self.bn2(out)))
        out = self.conv3(self.activation(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    """Adapted from: https://github.com/locuslab/fast_adversarial
    Modifications:
    - nn.AdaptiveAvgPool2d layer instead of nn.AvgPool2d to allow dynamic image shapes
    """
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # allows adaptive image sizes`
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.activation = nn.ReLU()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=None):
        out = self.conv1(x)
        out = self.layer1(out)
        if layer == 1: return out

        out = self.layer2(out)
        if layer == 2: return out

        out = self.layer3(out)
        if layer == 3: return out

        out = self.layer4(out)
        if layer == 4: return out

        out = self.activation(self.bn(out))
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def PreActResNet18():
    return PreActResNet(PreActBlock, [2,2,2,2])


def PreActResNet34():
    return PreActResNet(PreActBlock, [3,4,6,3])


def PreActResNet50():
    return PreActResNet(PreActBottleneck, [3,4,6,3])


def PreActResNet101():
    return PreActResNet(PreActBottleneck, [3,4,23,3])


def PreActResNet152():
    return PreActResNet(PreActBottleneck, [3,8,36,3])
