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
import sys
import os
import warnings
import logging
import copy
import time
import functools
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm 

# types
from torch import Tensor as TensorType
from torch.optim import Optimizer as OptimizerType
from torch.optim.lr_scheduler import _LRScheduler as SchedulerType
from torch.nn import Module as ModuleType
from typing import Tuple, Optional
from types import FunctionType

from models.preact_resnet import PreActResNet18, PreActResNet34, PreActResNet50, PreActResNet101, PreActResNet152
from constants import CIFAR10_MEAN, CIFAR10_STD, LOWER_LIMIT, UPPER_LIMIT
from attack.pgd import evaluate_pgd, attack_pgd
from utils import clamp


def base_args(func):
    """Allows convenient addition of experiment-specific args via the decorator mechanism.

    Example usage:
    ```python
    @base_args
    def func(parser):
        parser.add_argument(...)
    ```
    """
    parser = argparse.ArgumentParser()

    try:
        data_dir = os.environ['DATA_DIR']
    except KeyError:
        raise EnvironmentError('Create a `DATA_DIR` environment variable pointing to a directory in which the dataset lives (or in which to download the dataset via `--download-data`).')

    try:
        out_dir  = os.environ['OUT_DIR']
    except KeyError:
        raise EnvironmentError('Create a `OUT_DIR` envionrment variable pointing to the directory in which to save outputs.')

    ## universal config
    parser.add_argument('--data-dir',       default=data_dir,    type=str,    help='Path to directory in which to store data/models.')
    parser.add_argument('--out-dir',        default=out_dir,     type=str,    help='Output directory')
    parser.add_argument('--model',          default='resnet18',  type=str,    help='Model architecture.', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
    parser.add_argument('--batch-size',     default=128,         type=int,    help='Train and test batch sizes.')
    parser.add_argument('--epochs',         default=15,          type=int,    help='Number of training epochs.')
    parser.add_argument('--epoch-peak',     default=7,           type=int,    help='Epoch at which peak learning rate should be achieved.')
    parser.add_argument('--lr-min',         default=0.,          type=float,  help='Minimum learning rate for scheduler.')
    parser.add_argument('--weight-decay',   default=5e-4,        type=float,  help='Weight L2 regularization level.')
    parser.add_argument('--momentum',       default=0.9,         type=float,  help='SGD momentum.')
    parser.add_argument('--seed',           default=0,           type=int,    help='Random seed')
    parser.add_argument('--loss-scale',     default='1.0',       type=str,    help='If loss_scale is "dynamic", adaptively adjust the loss scale over time', choices=['1.0', 'dynamic'])
    parser.add_argument('--gpu-id',         default=0,           type=int,    help='ID of the GPU to train on.')
    # flags
    parser.add_argument('--lr-scheduler',   action='store_false', help='Linear warmup from `--lr-min` to `--lr-max` for `--epoch-peak` epochs, then linear decay back down to `--lr-min` for remaining `--epochs`.')
    parser.add_argument('--early-stop',     action='store_true',  help='Early stop if overfitting occurs')
    parser.add_argument('--master-weights', action='store_true',  help='Maintain FP32 master weights to accompany any FP16 model weights, not applicable for O1 opt level')
    parser.add_argument('--no-verbose',     action='store_false', help='Display logging to stdout.')
    parser.add_argument('--download-data',  action='store_true',  help='Flag to tell pytorch to download the dataset.')
    parser.add_argument('--no-eval',        action='store_true',  help='Skip final evaluation after training model.')
    parser.add_argument('--debug',          action='store_true',  help='Flag to help with debugging during training.')

    # add additional user-defined arguments
    @functools.wraps(func)
    def decorator():
        func(parser)
        return parser.parse_args()
    
    return decorator


def get_environment(
    args_factory  :FunctionType,
    name          :str
) -> Tuple[object, object, str]:
    """Set up:
    - I/O directories
    - logging
    - GPU and cuda benchmarking
    - set random seeds
    """
    args = args_factory()

    # create I/O environment
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.out_dir,  exist_ok=True)

    # suppress user warnings
    if not args.no_verbose:
        warnings.filterwarnings('ignore', category=UserWarning)

    ### set up local data dump
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    logfile = os.path.join(args.out_dir, f'{name}_{args.model}_output.log')
    if os.path.exists(logfile):
        os.remove(logfile)

    ### training logger
    logger = logging.getLogger(__name__)
    if args.no_verbose:
        logger.addHandler(logging.StreamHandler(sys.stdout))
    logging.basicConfig(
        format  ='[%(asctime)s] - %(message)s',
        datefmt ='%Y/%m/%d %H:%M:%S',
        level   =logging.INFO,
        filename=logfile)
    logger.info(args)

    ### device configuration
    torch.backends.cudnn.benchmark = True
    assert args.gpu_id < torch.cuda.device_count(), \
        f'Invalid GPU ID {args.gpu_id}, only found {torch.cuda.device_count()} devices'
    device = f'cuda:{args.gpu_id}'

    ### reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    return args, logger, device


def get_dataset(
    data_dir       :str, 
    batch_size     :int, 
    download_data  :bool
) -> Tuple[DataLoader, DataLoader]: 
    """Obtain dataloaders for CIFAR-10 train/test.
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])

    kwargs = dict(
        batch_size =batch_size,
        shuffle    =True,
        pin_memory =True,
        num_workers=4)

    train_loader = DataLoader(
        dataset=datasets.CIFAR10(data_dir, 
            train    =True, 
            transform=train_transform, 
            download =download_data), 
        **kwargs)

    test_loader = DataLoader(
        dataset=datasets.CIFAR10(data_dir, 
            train    =False, 
            transform=test_transform, 
            download =False), 
        **kwargs)

    return train_loader, test_loader


def init_experiment(
    args_factory  :FunctionType,
    name          :str
) -> Tuple[
    object, object, str, 
    Tuple[DataLoader, DataLoader], 
    Tuple[FunctionType, FunctionType, FunctionType]
]:
    """Boilerplate for CIFAR-10 experimental setup. Model/optimizer/scheduler are implemented as factory methods to allow for easy creation of
    multiple models with the same architecture.
    """
    args, logger, device = get_environment(
        args_factory=args_factory, 
        name        =name)

    train_loader, test_loader = get_dataset(
        data_dir     =args.data_dir, 
        batch_size   =args.batch_size,
        download_data=args.download_data)

    def model_factory(device  :str  =device) -> ModuleType:
        available_models = dict(
            resnet18       =PreActResNet18,
            resnet34       =PreActResNet34,
            resnet50       =PreActResNet50,
            resnet101      =PreActResNet101,
            PreActResNet152=PreActResNet152)
        try:
            model = available_models[args.model]().to(device)
            model.train()
        except KeyError:
            raise NotImplementedError(f'Unknown model architecture `{args.model}`')

        return model

    def optimizer_factory(model  :ModuleType) -> OptimizerType:
        return torch.optim.SGD(model.parameters(), 
            lr          =args.lr_min if args.lr_scheduler else args.lr_max, 
            momentum    =args.momentum, 
            weight_decay=args.weight_decay)

    def scheduler_factory(opt  :OptimizerType) -> SchedulerType:
        """Linear warmup followed by linear decay for remaining epochs i.e. triangular schedule.
        """
        step_size_up = args.epoch_peak * len(train_loader)
        step_size_down = (args.epochs - args.epoch_peak) * len(train_loader)
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, 
            base_lr       =args.lr_min, 
            max_lr        =args.lr_max,
            step_size_up  =step_size_up, 
            step_size_down=step_size_down)
        return scheduler

    return args, logger, device, \
        (train_loader, test_loader), \
        (model_factory, optimizer_factory, scheduler_factory)


def fit(
    step         :FunctionType,
    epochs       :int,
    model        :ModuleType,
    optimizer    :OptimizerType,
    scheduler    :SchedulerType,
    data_loader  :DataLoader,
    model_path   :str,
    logger       :object,
    early_stop   :bool            =False,
    pgd_kwargs   :Optional[dict]  =None,
    verbose      :bool            =False
) -> Tuple[ModuleType, dict]:
    """Standard pytorch boilerplate for training a model.
    Allows for early stopping wrt robust accuracy rather than the usual clean accuracy.
    """
    device = next(model.parameters()).device
    prev_robust_acc = 0.
    start_train_time = time.time()
    logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')

    for epoch in range(epochs):
        start_epoch_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0

        data_generator = enumerate(data_loader)
        if verbose:
            data_generator = tqdm(data_generator, total=len(data_loader), desc=f'Epoch {epoch + 1}')

        for i, (X, y) in data_generator:
            X, y = X.to(device), y.to(device)
            if i == 0:
                first_batch = (X, y)

            loss, logits = step(X, y, 
                model    =model, 
                optimizer=optimizer, 
                scheduler=scheduler)

            train_loss += loss.item() * y.size(0)
            train_acc += (logits.argmax(dim=1) == y).sum().item()
            train_n += y.size(0)

        if early_stop:
            assert pgd_kwargs is not None
            # Check current PGD robustness of model using random minibatch
            X, y = first_batch

            pgd_delta = attack_pgd(
                model=model, 
                X    =X, 
                y    =y, 
                opt  =optimizer,
                **pgd_kwargs)

            model.eval()
            with torch.no_grad():
                output = model(clamp(X + pgd_delta[:X.size(0)], pgd_kwargs['lower_limit'], pgd_kwargs['upper_limit']))
            robust_acc = (output.softmax(dim=1).argmax(dim=1) == y).sum().item() / y.size(0)
            if robust_acc - prev_robust_acc < -0.2:
                logger.info('EARLY STOPPING TRIGGERED')
                break
            prev_robust_acc = robust_acc
            best_state_dict = copy.deepcopy(model.state_dict())
            model.train()
        
        epoch_time = time.time()
        lr = scheduler.get_last_lr()[0]
        logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f',
            epoch, 
            epoch_time - start_epoch_time, 
            lr, 
            train_loss / train_n, 
            train_acc / train_n)

    train_time = time.time()
    if not early_stop:
        best_state_dict = model.state_dict()
    torch.save(best_state_dict, model_path)
    logger.info('Total train time: %.4f minutes', (train_time - start_train_time)/60)

    return model, best_state_dict


def evaluate_standard(
    test_loader  :DataLoader, 
    model        :ModuleType,
    verbose      :bool
) -> Tuple[float, float]:
    """source: https://github.com/locuslab/fast_adversarial
    """
    device = next(model.parameters()).device
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()
    iterator = enumerate(test_loader)
    if verbose:
        iterator = tqdm(iterator, total=len(test_loader), desc='Standard evaluate')
    with torch.no_grad():
        for i, (X, y) in iterator:
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return test_loss / n, test_acc / n


def evaluate(
    model        :ModuleType, 
    test_loader  :DataLoader, 
    upper_limit  :TensorType, 
    lower_limit  :TensorType, 
    verbose      :bool
) -> None:
    """Compute robust/clean accuracies.
    """
    model.float()
    model.eval()

    pgd_loss, pgd_acc = evaluate_pgd(
        test_loader =test_loader, 
        model       =model, 
        upper_limit =upper_limit, 
        lower_limit =lower_limit, 
        attack_iters=50, 
        restarts    =10,
        verbose     =verbose)

    test_loss, test_acc = evaluate_standard(
        test_loader=test_loader, 
        model      =model, 
        verbose    =verbose)

    logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
    logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc)
 