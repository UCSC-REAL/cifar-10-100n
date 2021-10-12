import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
import math
# Loss functions


def loss_cross_entropy(epoch, y, t,class_list, ind, noise_or_not,loss_all,loss_div_all):
    ## Record loss and loss_div for further analysis
    loss = F.cross_entropy(y, t, reduce = False)
    loss_numpy = loss.data.cpu().numpy()
    num_batch = len(loss_numpy)
    loss_all[ind,epoch] = loss_numpy
    return torch.sum(loss)/num_batch


