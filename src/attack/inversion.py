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
import torch.nn.functional as F

from utils import toggle_requires_grad

# types
from typing import Union, List, Optional
from torch import Tensor, LongTensor
from torch.nn import Module


def inversion(
    model         :Module,
    x0            :Tensor,
    *,
    stepsize      :float,
    category      :int,
    untargeted    :bool                    =False,
    max_iters     :int                     =10000,
    clamp         :Optional[torch.Tensor]  =None,
    keep_path     :bool                    =False,
    geometry      :str                     ='linf',
) -> Union[List[Tensor], Tensor]:
    """Runs adversarial perturbation -- both targeted and untargeted version are available.
    No constraints are assumed by default but can be added via the clamp argument.
    This means that only the projection operator is only defined for l_\infty balls. 

    :param model         : PyTorch module
    :param x0            : Initialization point
    :param stepsize      : The stepsize of each gradient ascent step
    :param category      : The target label for cross-entropy loss
    :param untargeted    : (False) - Targeted perturbation. 
                           (True)  - Untargeted perturbation.
    :param max_iters     : Number of iterations. Currently there is no convergence criterion.
    :param clamp         : A tensor specifying the range of values to restrict the pixel values to. 
                           A range needs to be specified for each color channel.
    :param keep_path     : Flag to retain the perturbation iterates. Useful for visualization.
    :param geometry      : ('l2' or 'linf') Under which geometry to normalize the gradient directions.
    """
    device = next(model.parameters()).device
    model.eval()

    batch_size = x0.shape[0]
    if isinstance(category, int):
        category = torch.LongTensor([category]).repeat(batch_size).to(device)
    else:
        category = category.to(device)

    x_inv = x0.clone().to(device)

    if clamp is not None:
        assert len(clamp) == x0.shape[1]
        assert len(clamp[0]) == 2

    if keep_path:
        path = []

    for n_iter in range(max_iters):   
        with toggle_requires_grad(model, False), toggle_requires_grad(x_inv, True):
            grad = torch.autograd.grad(
                outputs    =F.cross_entropy(model(x_inv), category), 
                inputs     =x_inv, 
                only_inputs=True)[0]

        if geometry == 'l2':
            norm = grad.flatten(1).norm(dim=1)[:, None, None, None]
            x_inv -= stepsize * grad / (norm + 1e-10)
        elif geometry == 'linf':
            x_inv -= stepsize * grad.sign()
        else:
            raise NotImplementedError
            
        if clamp is not None:
            # clamp each channel to proper range
            for i in range(len(clamp)):
                x_inv[:,i] = torch.clamp(x_inv[:,i], *clamp[i])

        if keep_path:
            path.append(x_inv.clone().cpu())

    if keep_path:
        x_inv = path
    return x_inv
