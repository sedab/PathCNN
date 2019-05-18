import time
import os
from glob import glob

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.autograd import Variable

import torchvision
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import torchvision.transforms as transforms
from torchvision import datasets

from itertools import accumulate
from functools import reduce

from utils.CLR import *

# New imports
from PIL import Image
from utils.dataloader import *
from utils.auc import *
from utils import new_transforms
import argparse
import random


# models_to_test = 'inception_v3'
# classes = [0,1,2]
# num_classes = len(classes)
# model_urls = {'inception_v3': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth' }
# model_names = model_urls.keys()
# input_sizes = {'inception' : (299,299)}
# last_params = ['AuxLogits.fc.weight', 'AuxLogits.fc.bias', 'fc.weight', 'fc.bias']


def diff_states(dict_canonical, dict_subset,params):
    names1, names2 = (list(dict_canonical.keys()), list(dict_subset.keys()))
    
    #Sanity check that param names overlap
    #Note that params are not necessarily in the same order
    #for every pretrained model
    not_in_1 = [n for n in names1 if n not in names2]
    not_in_2 = [n for n in names2 if n not in names1]
    assert len(not_in_1) == 0
    assert len(not_in_2) == 0
    for name, v1 in dict_canonical.items():

        if name in params:
            break

        v2 = dict_subset[name]
        assert hasattr(v2, 'size')
        if v1.size() != v2.size():
            print(name,v1.size(),v2.size())
            yield (name, v1)   


def load_model_merged(name, num_classes, model_urls, params):
    model = models.__dict__[name](num_classes=num_classes)
    model_dict = model.state_dict()
    pretrained_state = model_zoo.load_url(model_urls[name])

    #Diff
    diff = [s for s in diff_states(model.state_dict(), pretrained_state,params)]
    print("Replacing the following state from initialized", name, ":", [d[0] for d in diff])
    for name, value in diff:
        pretrained_state[name] = value
        assert len([s for s in diff_states(model.state_dict(), pretrained_state)]) == 0

    # Remove last layer weights because different number of classes
    for name in params:
        del pretrained_state[name]
    
    #Update weights from pretrained state
    model_dict.update(pretrained_state)

    #Merge
    model.load_state_dict(model_dict)
    return model, diff


def filtered_params(net, param_list=None):
    def in_param_list(s):
        for p in param_list:
            if s.endswith(p):
                return True
        return False    
    #Caution: DataParallel prefixes '.module' to every parameter name
    params = net.named_parameters() if param_list is None \
    else (p for p in net.named_parameters() if \
          in_param_list(p[0]) and p[1].requires_grad)
    return params






