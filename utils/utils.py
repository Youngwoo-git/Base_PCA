import torch
from thop import profile


def calculate_op(x, model):
    
    macs, params = profile(model, inputs=(x, ), verbose = False)
    
    return macs, params