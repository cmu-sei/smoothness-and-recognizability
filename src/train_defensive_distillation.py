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
import subprocess
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import UPPER_LIMIT, LOWER_LIMIT 
from experiment import base_args, init_experiment, fit, evaluate
from utils import zero_grad

# types
from types import FunctionType
from torch.nn import Module
from torch import Tensor


@base_args
def get_args(parser):
    parser.add_argument('--lr-max',              default=0.4,  type=float)
    parser.add_argument('--softmax-temperature', default=100., type=float)
    parser.add_argument('--pretrained-model',    default=None, help='Path to a set of pretrained model weights.')


def load_model(
    path           :str, 
    model_factory  :FunctionType, 
    device         :str                  ='cpu'
) -> Module:
    """Load a model for eval. Assume that the model has the same architecture as 
    the model to be trained.

    :param path          : Location of model weights.
    :param model_factory : Function that returns a nn.Module instance. Must have a `device` kwarg.
    :param device        : Options are 'cuda:ID' or 'cpu'.
    """
    assert os.path.isfile(path)
    pdict = torch.load(path, map_location='cpu')
    model = model_factory(device='cpu')
    model.load_state_dict(pdict)
    model.requires_grad_(False)
    model.eval()
    return model.to(device)


def softmax(
    x            :Tensor, 
    temperature  :float        =1., 
    dim          :int          =0,
    dtype        :torch.dtype  =None
) -> Tensor:
    """Temperature-scaled softmax.
    """
    return F.softmax(x / temperature, dim=dim)


def cross_entropy_loss(
    logits  :Tensor, 
    y       :Tensor
) -> Tensor:
    """Cross-entropy w/ full label vector instead of singleton label.
    """
    return -(y * logits.log_softmax(dim=1)).sum(dim=1).mean()


def train_step(
    X, y, model, optimizer, scheduler,     # generic
    pretrained_model, softmax_temperature  # experiment-specific
) -> tuple:  
    model.train()

    ### temperature scaled predictions i.e. softened pseudo-labels
    with torch.no_grad():
        y_kd = softmax(pretrained_model(X), softmax_temperature, dim=1)

    ### compute loss using softened pseudo-labels
    logits = model(X)
    loss = cross_entropy_loss(logits, y_kd)

    ### update model
    model.apply(zero_grad)
    loss.backward()
    optimizer.step()
    scheduler.step()

    return loss, logits


def main():
    name = 'distilled'

    # set up the experiment
    args, logger, device, \
    (train_loader, test_loader), \
    (model_factory, optimizer_factory, scheduler_factory) \
        = init_experiment(args_factory=get_args, name=name)

    assert not args.early_stop, \
        'Model distillation is non-robust, do not early stop wrt adversarial accuracy.'

    if args.pretrained_model is None:
        logger.info('No pretrained model specified... pretraining model via `train_standard.py`.')
        process = subprocess.run([
            'python', 'train_standard.py', 
            '--epochs', str(args.epochs),
            '--no-eval', 
            '--temperature', str(args.softmax_temperature)])
        pretrained_model_path = os.path.join(args.out_dir, 'model_preact_resnet18_standard.pt')
    else:
        logger.info(f'Pretrained model specified as `{args.pretrained_model}`... skipping pretraining.')
        pretrained_model_path = args.pretrained_model

    pretrained_model = load_model(
        path         =pretrained_model_path, 
        model_factory=model_factory, 
        device       =device)

    model     = model_factory()
    opt       = optimizer_factory(model)
    scheduler = scheduler_factory(opt)

    ### train model
    model, best_state_dict = fit(
        step       =functools.partial(train_step, 
            pretrained_model   =pretrained_model, 
            softmax_temperature=args.softmax_temperature),
        epochs     =args.epochs,
        model      =model,
        optimizer  =opt,
        scheduler  =scheduler,
        data_loader=train_loader,
        model_path =os.path.join(args.out_dir, f'model_preact_resnet18_{name}.pt'),
        logger     =logger,
        early_stop =args.early_stop)

    ### evaluate model
    if not args.no_eval:
        model_test = model_factory()
        model_test.load_state_dict(best_state_dict)
        evaluate(
            model      =model_test,
            test_loader=test_loader,
            upper_limit=UPPER_LIMIT,
            lower_limit=LOWER_LIMIT,
            verbose    =args.no_verbose)


if __name__ == "__main__":
    main()