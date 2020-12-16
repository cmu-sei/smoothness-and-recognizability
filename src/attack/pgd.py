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
from typing import Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import clamp
from constants import STD


def attack_pgd(
    model         :torch.nn.Module, 
    X             :torch.Tensor, 
    y             :torch.Tensor, 
    epsilon       :torch.Tensor, 
    alpha         :float, 
    lower_limit   :torch.Tensor,
    upper_limit   :torch.Tensor,
    attack_iters  :int, 
    restarts      :int, 
    opt           :torch.optim.Optimizer  =None
) -> torch.Tensor:
    """Adapted from: https://github.com/locuslab/fast_adversarial
    """
    device = next(model.parameters()).device
    model.eval()

    lower_limit = lower_limit.to(device)
    upper_limit = upper_limit.to(device)
    max_loss = torch.zeros(y.shape[0]).to(device)
    max_delta = torch.zeros_like(X).to(device)

    for zz in range(restarts):
        delta = torch.zeros_like(X)
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True

        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = F.cross_entropy(output, y)

            if opt is not None:
                loss.backward()
            else:
                loss.backward()
            grad = delta.grad.detach()

            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()

        all_loss = F.cross_entropy(model(X+delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)

    return max_delta


def evaluate_pgd(
    test_loader   :torch.utils.data.DataLoader, 
    model         :torch.nn.Module, 
    lower_limit   :torch.Tensor,
    upper_limit   :torch.Tensor,
    attack_iters  :int, 
    restarts      :int,
    verbose       :bool,
    epsilon       :int  =8
) -> Tuple[float, float]:
    """Adapted from: https://github.com/locuslab/fast_adversarial
    """
    device = next(model.parameters()).device
    model.eval()

    epsilon = (epsilon / 255.) / STD.to(device)
    alpha = (2 / 255.) / STD.to(device)
    pgd_loss = 0
    pgd_acc = 0
    n = 0

    iterator = enumerate(test_loader)
    if verbose:
        iterator = tqdm(iterator, total=len(test_loader), desc='PGD evaluate')
    for i, (X, y) in iterator:
        X, y = X.to(device), y.to(device)
        pgd_delta = attack_pgd(model, X, y, 
            epsilon, alpha, 
            lower_limit, upper_limit, 
            attack_iters, restarts)
        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)

    return pgd_loss/n, pgd_acc/n
