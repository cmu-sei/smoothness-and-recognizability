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
import functools
import os

import torch
import torch.nn as nn

from constants import UPPER_LIMIT, LOWER_LIMIT 
from experiment import base_args, init_experiment, fit, evaluate
from utils import zero_grad


@base_args
def get_args(parser):
    parser.add_argument('--lr-max',    default=0.2, type=float)
    parser.add_argument('--smoothing', default=1e-1, type=float)


def cross_entropy_label_smoothing(logits, y, smoothing=1e-1):
    """Train on softened labels `y_smooth = (1 - smoothing) y + smoothing / K`
    reference: https://arxiv.org/pdf/1906.02629.pdf
    """
    confidence = 1. - smoothing 
    log_scores = logits.log_softmax(dim=1)
    nll = -log_scores.gather(dim=1, index=y.unsqueeze(1)).squeeze(1)
    loss_smoothed = -log_scores.mean(dim=1)
    loss = confidence * nll + smoothing * loss_smoothed
    return loss.mean()


def train_step(
    X, y, model, optimizer, scheduler,  # generic
    smoothing                           # experiment-specific
) -> tuple:
    model.train()

    ### compute loss
    logits = model(X)
    loss = cross_entropy_label_smoothing(logits, y, smoothing=smoothing)

    ### update model
    model.apply(zero_grad)
    loss.backward()
    optimizer.step()
    scheduler.step()

    return loss, logits


def main():
    name = 'label_smoothing'

    # set up the experiment
    args, logger, device, \
    (train_loader, test_loader), \
    (model_factory, optimizer_factory, scheduler_factory) \
        = init_experiment(args_factory=get_args, name=name)

    assert not args.early_stop, \
        'Model distillation is non-robust, do not early stop wrt adversarial accuracy.'

    model     = model_factory()
    opt       = optimizer_factory(model)
    scheduler = scheduler_factory(opt)

    model, best_state_dict = fit(
        step       =functools.partial(train_step,
            smoothing=args.smoothing),
        epochs     =args.epochs,
        model      =model,
        optimizer  =opt,
        scheduler  =scheduler,
        data_loader=train_loader,
        model_path =os.path.join(args.out_dir, f'model_preact_resnet18_{name}.pt'),
        logger     =logger,
        early_stop =args.early_stop)

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
