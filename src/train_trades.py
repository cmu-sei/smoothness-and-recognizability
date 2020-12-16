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
import os
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import UPPER_LIMIT, LOWER_LIMIT, STD
from experiment import base_args, init_experiment, fit, evaluate
from utils import toggle_eval, toggle_requires_grad, disable_batchnorm_tracking, zero_grad, clamp, project


@base_args
def get_args(parser):
    parser.add_argument('--lr-max',        default=0.2,     type=float)
    parser.add_argument('--epsilon',       default=8,       type=int)
    parser.add_argument('--attack-iters',  default=7,       type=int,    help='Attack iterations')
    parser.add_argument('--alpha',         default=2,       type=int,    help='Attack step size')
    parser.add_argument('--reg-weight',    default=6.0,     type=float,  help='TRADES regularization weight')
    parser.add_argument('--geometry',      default='linf',  type=str,    help='Adversary lp constraint.')


def train_step(
    X, y, model, optimizer, scheduler,                                                       # generic 
    reg_weight, geometry, epsilon, alpha, lower_limit, upper_limit, attack_iters, criterion  # experiment-specific
):
    """TRADES algorithm [https://arxiv.org/abs/1901.08573].
    """
    ### adversarial perturbation
    # init perturbation
    delta = torch.zeros_like(X)
    for i in range(len(epsilon)):
        delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
    delta.data = clamp(delta, lower_limit - X, upper_limit - X)
    
    # find approx optimal perturbation
    criterion_kl = nn.KLDivLoss(reduction='sum')
    with toggle_eval(model):  # turn off batchnorm stat tracking
        for _ in range(attack_iters):
            with toggle_requires_grad(delta, True):
                grad = torch.autograd.grad(
                    outputs=criterion_kl(
                        F.log_softmax(model(X + delta), dim=1), 
                        F.softmax(model(X), dim=1)),
                    inputs     =delta,
                    only_inputs=True)[0]
            delta = project(delta + alpha * grad.sign(), epsilon, geometry=geometry)
            delta = clamp(delta, lower_limit - X, upper_limit - X)

    ### adversarial loss
    model.apply(zero_grad)
    # disable batchnorm stat tracking to avoid distribution shift from adversarial examples
    with disable_batchnorm_tracking(model):
        loss = criterion_kl(
            F.log_softmax(model(X + delta), dim=1),
            F.softmax(model(X), dim=1))
        loss *= (reg_weight / X.shape[0])
    loss.backward()

    ### clean loss
    # batchnorm stat tracking is fine here
    output = model(X)
    loss = criterion(output, y)

    ### update model
    loss.backward()
    optimizer.step()
    scheduler.step()

    return loss, output


def main():
    name = 'trades'

    # set up the experiment
    args, logger, device, \
    (train_loader, test_loader), \
    (model_factory, optimizer_factory, scheduler_factory) \
        = init_experiment(args_factory=get_args, name=name)

    model     = model_factory()
    opt       = optimizer_factory(model)
    scheduler = scheduler_factory(opt)

    ### training adversary config
    assert args.geometry == 'linf', \
        'l2 adversary not supported yet'
    std = STD.to(device)
    upper_limit = UPPER_LIMIT.to(device)
    lower_limit = LOWER_LIMIT.to(device)
    epsilon = (args.epsilon / 255.) / std
    # parameters for PGD training
    pgd_train_kwargs = dict(
        reg_weight  =args.reg_weight,
        geometry    =args.geometry,
        epsilon     =epsilon,
        alpha       =(args.alpha / 255.) / std,
        lower_limit =lower_limit,
        upper_limit =upper_limit,
        attack_iters=args.attack_iters,
        criterion   =nn.CrossEntropyLoss())
    # parameters for PGD early stopping
    pgd_kwargs = dict(
        epsilon     =epsilon,
        alpha       =(2 / 255.) / std,
        lower_limit =lower_limit,
        upper_limit =upper_limit,
        attack_iters=5,
        restarts    =1)

    # training
    model, best_state_dict = fit(
        step       =functools.partial(train_step, **pgd_train_kwargs),
        epochs     =args.epochs,
        model      =model,
        optimizer  =opt,
        scheduler  =scheduler,
        data_loader=train_loader,
        model_path =os.path.join(args.out_dir, f'model_preact_resnet18_{name}.pt'),
        logger     =logger,
        early_stop =args.early_stop,
        pgd_kwargs =pgd_kwargs)

    # eval
    if not args.no_eval:
        model_test = model_factory()
        model_test.load_state_dict(best_state_dict)
        evaluate(
            model      =model_test,
            test_loader=test_loader,
            upper_limit=upper_limit,
            lower_limit=lower_limit,
            verbose    =args.no_verbose)


if __name__ == "__main__":
    main()
