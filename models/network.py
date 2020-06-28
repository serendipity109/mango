#!/usr/bin/env python
# coding: utf-8

# In[26]:


import torch  
import torch.nn as nn  
import yaml
import models.cls_hrnet as cls_hrnet
from torchvision import datasets, models, transforms
import numpy as np
        
def config(config_path):
    """ Loading config file. """
    with open(config_path, 'r') as f:
        return yaml.load(f, Loader = yaml.FullLoader)
    
def hr(num_class):
    model = eval('cls_hrnet.get_cls_net')(config('models/cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'))
    model.load_state_dict(torch.load("models/hrnetv2_w18_imagenet_pretrained.pth"))
    model.classifier = nn.Linear(2048, num_class)
    return model

def vgg(num_class):
    model = models.vgg16(pretrained=True)
    model.classifier._modules['6'] = nn.Linear(4096, num_class)
    return model

def res(num_class):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, num_class)
    return model

def mob(num_class):
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(1280, num_class)
    return model

def alex(num_class):
    model = models.alexnet(pretrained=True)
    model.classifier._modules['6'] = nn.Linear(4096, num_class)
    return model

