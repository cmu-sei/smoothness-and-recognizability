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
import copy
import contextlib

import torch
import numpy as np

# types
import typing as T
from typing import Union
from torch.nn import Module
from torch import Tensor

from constants import CIFAR10_SHAPE


def load_img(
    path          :str, 
    target_shape  :tuple  =CIFAR10_SHAPE
) -> Tensor:
    """Load an image from file and resize to ImageNet shape.
    Use this to seed a walk with non-random x0.
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError('Please install `Pillow`')

    assert len(target_shape) == 3
    assert target_shape[1] == target_shape[2], \
        'Only square target shapes are supported'
    img = Image.open(path)
    w, h = img.size
    w_ = min(w, h)  
    # center crop and resize image
    img = img.crop(( 
        (w - w_) // 2,
        (h - w_) // 2,
        (w + w_) // 2,
        (h + w_) // 2 )).resize(target_shape[1:])
    img = np.asarray(img)[..., :target_shape[0]]
    img = Tensor(img).permute(2,0,1)
    return img.contiguous().unsqueeze(0)


def normalize_img(x  :Tensor) -> Tensor: 
    """0-1 normalize an image
    """
    max_val = x.view(-1).max()
    min_val = x.view(-1).min()
    return (x - min_val) / (max_val - min_val)


def clamp(
    X            :Tensor, 
    lower_limit  :Tensor, 
    upper_limit  :Tensor
) -> Tensor:
    """multi-channel version of torch.clamp
    source: https://github.com/locuslab/fast_adversarial
    """
    return torch.max(torch.min(X, upper_limit), lower_limit)


def project(
    x         :Tensor, 
    epsilon   :Tensor, 
    geometry  :str            ='linf'
) -> Tensor:
    """lp ball projection operator
    """
    if geometry == 'linf':
        x = clamp(x, -epsilon, epsilon)
    elif geometry == 'l2':
        norm = x.flatten(1).norm(p=2, dim=1)
        mask = (norm > epsilon.flatten().mean())
        x[mask] /= norm.view(-1, *[1 for _ in x.shape[1:]])
        x[mask] *= epsilon
    else:
        raise NotImplementedError(f'Unknown adversary constraint geometry `{geometry}`')
    return x


@contextlib.contextmanager
def disable_batchnorm_tracking(model  :Module) -> None:
    """adapted from: https://github.com/lyakaap/VAT-pytorch/blob/master/vat.py
    """
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True
            
    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


@contextlib.contextmanager
def toggle_requires_grad(x  :Union[Tensor, Module], on=False) -> None:
    """Within context, set requires_grad attribute to `on` value, returning to original state out of context.
    """
    if isinstance(x, Tensor):
        requires_grad = x.requires_grad
    else:
        assert isinstance(x, Module)
        requires_grad = next(x.parameters()).requires_grad

    x.requires_grad_(on)
    yield
    x.requires_grad_(requires_grad) 


@contextlib.contextmanager
def toggle_eval(model  :Module) -> None:
    training = model.training

    model.eval()
    yield
    if training:
        model.train()


def zero_grad(m  :Module) -> None:
    """More efficient than optimizer.zero_grad().
    """
    for p in m.parameters():
        p.grad = None
