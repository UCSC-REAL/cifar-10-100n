import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
import math
# Loss functions

def det_acc(save_dir,best_ratio,args,loss_matrix, noise_or_not, epoch, sum_epoch = False):
    if sum_epoch is False:
        loss = loss_matrix[:,epoch]
    else:
        loss = np.sum(loss_matrix[:,:epoch+1],axis =1)
    loss_sort = np.argsort(loss)
    true_num_noise = sum(noise_or_not)
    num_noise_dectect = sum(noise_or_not[loss_sort[-true_num_noise:]])
    ratio = num_noise_dectect*1.0/true_num_noise
    if ratio > best_ratio[0]:
        best_ratio[0] = ratio
        np.save(save_dir+'/'+args.noise_type + str(args.noise_rate)+'detect.npy',loss_sort[-true_num_noise:])
    return ratio

def loss_cross_entropy(epoch, y, t,class_list, ind, noise_or_not,loss_all,loss_div_all):
    ##Record loss and loss_div for further analysis
    loss = F.cross_entropy(y, t, reduce = False)
    loss_numpy = loss.data.cpu().numpy()
    num_batch = len(loss_numpy)
    loss_all[ind,epoch] = loss_numpy
    return torch.sum(loss)/num_batch




# Modify from "loss_cross_entropy"
def loss_pls(epoch, y, t, class_list, ind, noise_or_not,loss_all,loss_div_all):
    smooth_rate = 0.6
    confidence = 1 - smooth_rate
    loss = F.cross_entropy(y, t, reduce = False)
    loss_ = -torch.log(F.softmax(y) + 1e-8)
    loss =  confidence*loss + smooth_rate*torch.mean(loss_,1)
    loss_numpy = loss.data.cpu().numpy()
    num_batch = len(loss_numpy)
    loss_all[ind,epoch] = loss_numpy
    return torch.sum(loss)/num_batch
    
 
# Modify from "loss_cross_entropy"
def loss_nls(epoch, y, t, class_list, ind, noise_or_not,loss_all,loss_div_all):
    smooth_rate = -6.0
    confidence = 1 - smooth_rate
    loss = F.cross_entropy(y, t, reduce = False)
    loss_ = -torch.log(F.softmax(y) + 1e-8)
    loss =  confidence*loss + smooth_rate*torch.mean(loss_,1)
    loss_numpy = loss.data.cpu().numpy()
    num_batch = len(loss_numpy)
    loss_all[ind,epoch] = loss_numpy
    return torch.sum(loss)/num_batch

def activation(x): return -torch.mean(torch.tanh(x) / 2.)
  
def conjugate(x): return -torch.mean(torch.tanh(x) / 2.)

def loss_fdiv(epoch, y, y_peer, t, t_peer, class_list, ind, noise_or_not,loss_all,loss_div_all):
    prob_acti = -F.nll_loss(y, t, reduce = False)
    prob_conj = -F.nll_loss(y_peer, t_peer, reduce = False)
    loss = activation(prob_acti) - conjugate(prob_conj)
    # loss_numpy = loss.cpu().numpy()
    # num_batch = loss_numpy.shape[0]
    # loss_all[ind,epoch] = loss_numpy
    return torch.sum(loss)/y.shape[0]


def loss_spl(epoch, y, t,class_list, ind, noise_or_not,loss_all,loss_div_all):
    if epoch <=20:
        return F.cross_entropy(y,t)
    else:
        alpha = f_spl_alpha_hard(epoch)
        loss = F.cross_entropy(y, t, reduce = False)
        loss_numpy = loss.data.cpu().numpy()
        num_batch = len(loss_numpy)
        loss_v = np.zeros(num_batch)
    for i in range(len(loss_numpy)):
        if loss_numpy[i] <= alpha:
            loss_v[i] = 1.0
    loss_v = loss_v.astype(np.float32)
    loss_v_var = Variable(torch.from_numpy(loss_v)).cuda()
    loss_ = loss_v_var * loss
    if sum(loss_v) == 0.0:
        return torch.mean(loss_)/100000000
    else:
        return torch.sum(loss_)/sum(loss_v)
        





def loss_div_hard_ave_peer(epoch, y, t,class_list, ind, noise_or_not,loss_all,loss_div_all, noise_prior = None):
    alpha = f_alpha_hard(epoch)
    beta = f_beta(epoch)
    # if epoch == 1:
    #     print(f'current beta is {beta}')
    loss = F.cross_entropy(y, t, reduce = False)
    loss_numpy = loss.data.cpu().numpy()
    num_batch = len(loss_numpy)
    loss_v = np.zeros(num_batch)
    loss_div_numpy = float(np.array(0))
    loss_ = -torch.log(F.softmax(y) + 1e-8)
    # sel metric
    loss_sel =  loss - torch.mean(loss_,1)
    if noise_prior is None:
        loss =  loss - beta*torch.mean(loss_,1)
    else:
        loss =  loss - beta*torch.sum(torch.mul(noise_prior, loss_),1)
    
    loss_div_numpy = loss_sel.data.cpu().numpy()
    loss_all[ind,epoch] = loss_numpy
    loss_div_all[ind,epoch] = loss_div_numpy
    # if epoch<=5:
    #     return F.cross_entropy(y, t)
    # else:
    for i in range(len(loss_numpy)):
        if loss_div_numpy[i] <= alpha:
            loss_v[i] = 1.0
    loss_v = loss_v.astype(np.float32)
    loss_v_var = Variable(torch.from_numpy(loss_v)).cuda()
    loss_ = loss_v_var * loss
    if sum(loss_v) == 0.0:
        return torch.mean(loss_)/100000000
    else:
        return torch.sum(loss_)/sum(loss_v), loss_v.astype(int)
    # return torch.sum(loss_)/sum(loss_v)
    # return torch.mean(loss)

def loss_peer(epoch, y, y_peer, t, t_peer, class_list, ind, noise_or_not,loss_all,loss_div_all, noise_prior = None):
    alpha = f_alpha_hard(epoch)
    # beta = f_beta(epoch, 'peerloss')
    beta = 1.0
    # if epoch == 1:
    #     print(f'current beta is {beta}')
    loss = F.cross_entropy(y, t, reduce = False)
    loss_numpy = loss.data.cpu().numpy()
    num_batch = len(loss_numpy)
    loss_v = np.zeros(num_batch)
    loss_div_numpy = float(np.array(0))
    loss_ = -torch.log(F.softmax(y_peer) + 1e-8)
    loss_peer = torch.gather(loss_,1,t_peer.view(-1,1)).view(-1)
    # print(f'loss_peer.shape: {loss_peer.shape}')
    # print(f'loss.shape: {loss.shape}')
    # sel metric
    loss_sel =  loss - torch.mean(loss_,1)
    if epoch > 10:
        loss = loss - beta*loss_peer
    else:
        loss = loss
    
    # if noise_prior is None:
    #     loss =  loss - beta*torch.mean(loss_,1)
    # else:
    #     loss =  loss - beta*torch.sum(torch.mul(noise_prior, loss_),1)
    
    loss_div_numpy = loss_sel.data.cpu().numpy()
    loss_all[ind,epoch] = loss_numpy
    loss_div_all[ind,epoch] = loss_div_numpy
    # if epoch<=5:
    #     return F.cross_entropy(y, t)
    # else:
    for i in range(len(loss_numpy)):
        if loss_div_numpy[i] <= alpha:
            loss_v[i] = 1.0
    loss_v = loss_v.astype(np.float32)
    loss_v_var = Variable(torch.from_numpy(loss_v)).cuda()
    loss_ = loss_v_var * loss
    if sum(loss_v) == 0.0:
        return torch.mean(loss_)/100000000
    else:
        return torch.sum(loss_)/sum(loss_v), loss_v.astype(int)
    # return torch.sum(loss_)/sum(loss_v)
    # return torch.mean(loss)

def loss_only_peer(epoch, y, t,class_list, ind, noise_or_not,loss_all,loss_div_all, noise_prior = None):
    #alpha = f_alpha_hard(epoch)
    beta = f_beta(epoch)
    # if epoch == 1:
    #     print(f'current beta is {beta}')
    loss = F.cross_entropy(y, t, reduce = False)
    loss_numpy = loss.data.cpu().numpy()
    num_batch = len(loss_numpy)
    loss_v = np.zeros(num_batch)
    loss_div_numpy = float(np.array(0))
    loss_ = -torch.log(F.softmax(y) + 1e-8)
    # sel metric
    loss_sel =  loss - torch.mean(loss_,1)
    if noise_prior is None:
        loss =  loss - beta*torch.mean(loss_,1)
    else:
        loss =  loss - beta*torch.sum(torch.mul(noise_prior, loss_),1)
    
    loss_div_numpy = loss_sel.data.cpu().numpy()
    loss_all[ind,epoch] = loss_numpy
    loss_div_all[ind,epoch] = loss_div_numpy
    # if epoch<=5:
    #     return F.cross_entropy(y, t)
    # else:
    for i in range(len(loss_numpy)):
        if loss_div_numpy[i] <= 1000000:
            loss_v[i] = 1.0
    loss_v = loss_v.astype(np.float32)
    loss_v_var = Variable(torch.from_numpy(loss_v)).cuda()
    loss_ = loss_v_var * loss
    if sum(loss_v) == 0.0:
        return torch.mean(loss_)/100000000
    else:
        return torch.sum(loss_)/sum(loss_v), loss_v.astype(int)

def f_alpha_hard(epoch):
    alpha = [100000]*300 + [0]*10 + [-1]*10+[-2]*10 + [-3]*10 + [-4]*10+[-5]*10 + [-6]*10 + [-7]*10+[-8]*10
    return alpha[epoch]







def loss_cross_entropy(epoch, y, t,class_list, ind, noise_or_not,loss_all,loss_div_all):
    ##Record loss and loss_div for further analysis
    loss = F.cross_entropy(y, t, reduce = False)
    loss_numpy = loss.data.cpu().numpy()
    num_batch = len(loss_numpy)
    loss_all[ind,epoch] = loss_numpy
    return torch.sum(loss)/num_batch



        

def loss_cores(epoch, y, t,class_list, ind, noise_or_not,loss_all,loss_div_all, noise_prior = None):
    beta = f_beta(epoch)
    # if epoch == 1:
    #     print(f'current beta is {beta}')
    loss = F.cross_entropy(y, t, reduce = False)
    loss_numpy = loss.data.cpu().numpy()
    num_batch = len(loss_numpy)
    loss_v = np.zeros(num_batch)
    loss_div_numpy = float(np.array(0))
    loss_ = -torch.log(F.softmax(y) + 1e-8)
    # sel metric
    loss_sel =  loss - torch.mean(loss_,1)
    if noise_prior is None:
        loss =  loss - beta*torch.mean(loss_,1)
    else:
        loss =  loss - beta*torch.sum(torch.mul(noise_prior, loss_),1)
    
    loss_div_numpy = loss_sel.data.cpu().numpy()
    loss_all[ind,epoch] = loss_numpy
    loss_div_all[ind,epoch] = loss_div_numpy
    for i in range(len(loss_numpy)):
        if epoch <=30:
            loss_v[i] = 1.0
        elif loss_div_numpy[i] <= -0.0:  # cifar10 synthetic 0.0, otherwise -8.0
            loss_v[i] = 1.0
    loss_v = loss_v.astype(np.float32)
    loss_v_var = Variable(torch.from_numpy(loss_v)).cuda()
    loss_ = loss_v_var * loss
    if sum(loss_v) == 0.0:
        return torch.mean(loss_)/100000000
    else:
        return torch.sum(loss_)/sum(loss_v), loss_v.astype(int)

def f_beta(epoch, loss_type='cores'):
    if loss_type == 'cores':
        beta_max = 2.0
    else:
        beta_max = 1.0
    beta1 = np.linspace(0.0, 0.0, num=10)
    beta2 = np.linspace(0.0, beta_max, num=30)
    beta3 = np.linspace(beta_max, beta_max, num=80)
 
    beta = np.concatenate((beta1,beta2,beta3),axis=0)
    return beta[epoch]



def DMI_loss(output, target):
    outputs = F.softmax(output, dim=1)
    targets = target.reshape(target.size(0), 1).cpu()
    y_onehot = torch.FloatTensor(target.size(0), 10).zero_()
    y_onehot.scatter_(1, targets, 1)
    y_onehot = y_onehot.transpose(0, 1).cuda()
    mat = y_onehot @ outputs
    return -1.0 * torch.log(torch.abs(torch.det(mat.float())) + 0.001)

def forward_loss(output, target, trans_mat):
    # l_{forward}(y, h(x)) = l_{ce}(y, h(x) @ T)
    outputs = F.softmax(output, dim=1)
    outputs = outputs @ trans_mat.cuda()
    outputs = torch.log(outputs)
    #loss = CE(outputs, target)
    loss = F.cross_entropy(outputs,target)
    return loss

def backward_loss(output, target, trans_mat):
    # l_{forward}(y, h(x)) = l_{ce}(y, h(x) @ T)
    trans_mat_inv = torch.inverse(trans_mat).cuda()
    outputs = F.softmax(output, dim=1)
    outputs = torch.log(outputs)
    #loss = CE(outputs, target)
    loss = -torch.mean(torch.sum((F.one_hot(target,trans_mat.shape[0]).float() @ trans_mat_inv) * outputs,axis=1),axis = 0)
    # loss = F.cross_entropy(outputs,target @ trans_mat_inv)  # TODO
    return loss



def lq_loss(epoch,outputs, target):
    # loss = (1 - h_j(x)^q) / q
    loss = torch.tensor(0.0).cuda()
    '''
    if epoch==0:
        print(outputs[0])
        print(target[0])
        print((1.0 - (outputs[0][target[0]]) ** 0.7) / 0.7)
    '''
    outputs = F.softmax(outputs, dim=1) + 1e-12
    for i in range(outputs.size(0)):
        loss += (1.0 - (outputs[i][target[i]]) ** 0.7) / 0.7
    # print(f'1{loss}')
    
    loss = loss / outputs.size(0)
    if torch.isnan(loss):
        print(outputs)
    # print(loss)
    return loss
