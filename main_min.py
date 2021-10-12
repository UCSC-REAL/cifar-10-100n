# -*- coding:utf-8 -*-
import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from data.datasets import input_dataset
from models import *
import argparse
import numpy as np


from loss import forward_loss, loss_cross_entropy,loss_spl, f_beta, f_alpha_hard, lq_loss, loss_peer, loss_cores, forward_loss
from torch.utils.data import RandomSampler

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type = float, default = 0.05)
parser.add_argument('--loss', type = str, help = 'ce, gce, dmi, flc, uspl,spl,peerloss', default = 'ce')
parser.add_argument('--result_dir', type = str, help = 'dir to save result txt files', default = '/home/jovyan/results')
parser.add_argument('--noise_type', type = str, help='clean_label, aggre_label, worse_label, random_label1, random_label2, random_label3', default='clean_label')
parser.add_argument('--noise_path', type = str, help='path of CIFAR-10_human.pt', default='./data/cifar-10-batches-py/noise_label/CIFAR-10_human.pt')
parser.add_argument('--top_bn', action='store_true')
parser.add_argument('--ideal', action='store_true')
parser.add_argument('--dataset', type = str, help = ' cifar10 or fakenews', default = 'cifar10')
parser.add_argument('--model', type = str, help = 'cnn,resnet', default = 'resnet')
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--is_human', action='store_true', default=False)

# Adjust learning rate and for SGD Optimizer
def adjust_learning_rate(optimizer, epoch,alpha_plan,loss_type='cores'):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]/(1+f_beta(epoch,loss_type))
        

def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# Train the Model
def train(epoch, num_classes, train_loader,peer_loader_x, peer_loader_y, model, optimizer,loss_all,loss_div_all,loss_type, noise_prior = None):
    train_total=0
    train_correct=0
    print(f'current beta is {f_beta(epoch,loss_type)}')
    v_list = np.zeros(num_training_samples)
    idx_each_class_noisy = [[] for i in range(num_classes)]
    if not isinstance(noise_prior, torch.Tensor):
        noise_prior = torch.tensor(noise_prior.astype('float32')).cuda().unsqueeze(0)
    peer_iter_x = iter(peer_loader_x)
    peer_iter_y = iter(peer_loader_y)
    for i, (images, labels, indexes) in enumerate(train_loader):
        ind=indexes.cpu().numpy().transpose()
        batch_size = len(ind)
        class_list = range(num_classes)

        if loss_type=='peerloss':
            x_peer, _, _ = peer_iter_x.next()
            _, label_peer, _ = peer_iter_y.next()
            x_peer = Variable(x_peer).cuda()
            label_peer = Variable(label_peer).cuda()

        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
       
        # Forward + Backward + Optimize
        logits = model(images)
        if loss_type=='peerloss':
            logits_peer = model(x_peer)
        prec, _ = accuracy(logits, labels, topk=(1, 5))
        # prec = 0.0
        train_total+=1
        train_correct+=prec
        if loss_type=='ce':
            loss = loss_cross_entropy(epoch,logits, labels,class_list,ind, noise_or_not, loss_all, loss_div_all)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % args.print_freq == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Training Accuracy: %.4F, Loss: %.4f'
                  %(epoch+1, args.n_epoch, i+1, len(train_dataset)//batch_size, prec, loss.data))


    train_acc=float(train_correct)/float(train_total)
    return train_acc

# Evaluate the Model
def evaluate(test_loader,model,save=False,epoch=0,best_acc_=0,args=None):
    model.eval()    # Change model to 'eval' mode.
    print('previous_best', best_acc_)
    correct = 0
    total = 0
    for images, labels, _ in test_loader:
        images = Variable(images).cuda()
        logits = model(images)
        outputs = F.softmax(logits, dim=1)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred.cpu() == labels).sum()
    acc = 100*float(correct)/float(total)

    if save:
        if acc > best_acc_:
            state = {'state_dict': model.state_dict(),
                     'epoch':epoch,
                     'acc':acc,
            }
            save_path= os.path.join(save_dir,args.loss + args.noise_type +'best.pth.tar')
            torch.save(state,save_path)
            best_acc_ = acc
            print(f'model saved to {save_path}!')
        if epoch == args.n_epoch -1:
            state = {'state_dict': model.state_dict(),
                     'epoch':epoch,
                     'acc':acc,
            }
            torch.save(state,os.path.join(save_dir,args.loss + args.noise_type +'last.pth.tar'))
    return acc, best_acc_



#####################################main code ################################################
args = parser.parse_args()
# Seed
# torch.manual_seed(args.seed)
# torch.cuda.manual_seed(args.seed)

# Hyper Parameters

batch_size = 128
learning_rate = args.lr
noise_type_map = {'clean':'clean_label', 'worst': 'worse_label', 'aggre': 'aggre_label', 'rand1': 'random_label1', 'rand2': 'random_label2', 'rand3': 'random_label3', 'clean100': 'clean_label', 'noisy100': 'noisy_label'}
args.noise_type = noise_type_map[args.noise_type]
# load dataset
train_dataset,test_dataset,num_classes,num_training_samples = input_dataset(args.dataset,args.noise_type, args.noise_path, args.is_human)
noise_prior = train_dataset.noise_prior
noise_or_not = train_dataset.noise_or_not
print('train_labels:', len(train_dataset.train_labels), train_dataset.train_labels[:10])
# load model
print('building model...')
if args.model == 'cnn':
    model = CNN(input_channel=3, n_outputs=num_classes)
else:
    model = ResNet34(num_classes)
print('building model done')
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.0005, momentum=0.9)
# Creat loss and loss_div for each sample at each epoch
loss_all = np.zeros((num_training_samples,args.n_epoch))
loss_div_all = np.zeros((num_training_samples,args.n_epoch))
### save result and model checkpoint #######
save_dir = args.result_dir +'/' +args.dataset + '/' + args.model
if not os.path.exists(save_dir):
    os.system('mkdir -p %s' % save_dir)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                   batch_size = 128,
                                   num_workers=args.num_workers,
                                   shuffle=True)
peer_sampler_x = RandomSampler(train_dataset, replacement=True)
peer_sampler_y = RandomSampler(train_dataset, replacement=True)

peer_loader_x = torch.utils.data.DataLoader(dataset = train_dataset,
                                   batch_size = 128,
                                   num_workers=args.num_workers,
                                   shuffle=False,
                                   sampler=peer_sampler_x)

peer_loader_y = torch.utils.data.DataLoader(dataset = train_dataset,
                                   batch_size = 128,
                                   num_workers=args.num_workers,
                                   shuffle=False,
                                   sampler=peer_sampler_y)


test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                  batch_size = 64,
                                  num_workers=args.num_workers,
                                  shuffle=False)
alpha_plan = [0.1] * 60 + [0.01] * 40
#alpha_plan = []
#for ii in range(args.n_epoch):
#    alpha_plan.append(learning_rate*pow(0.95,ii))
model.cuda()
txtfile=save_dir + '/' +  args.loss + args.noise_type + '.txt'
if os.path.exists(txtfile):
    os.system('rm %s' % txtfile)
# import pdb
# pdb.set_trace()
with open(txtfile, "a") as myfile:
    myfile.write('epoch: train_acc test_acc \n')

epoch=0
train_acc = 0
best_acc_ = 0.0
#print(best_acc_)
# training
noise_prior_cur = noise_prior
for epoch in range(args.n_epoch):
# train models
    print(f'epoch {epoch}')
    adjust_learning_rate(optimizer, epoch, alpha_plan, loss_type=args.loss)
    model.train()
    train_acc, noise_prior_delta = train(epoch,num_classes,train_loader,peer_loader_x,peer_loader_y, model, optimizer,loss_all,loss_div_all,args.loss,noise_prior = noise_prior_cur)
    noise_prior_cur = noise_prior*num_training_samples - noise_prior_delta
    noise_prior_cur = noise_prior_cur/sum(noise_prior_cur)
    print(f'noise_prior_cur: {noise_prior_cur}')
    print(f'noise_prior_delta: {noise_prior_delta}')

# evaluate models
    test_acc, best_acc_ = evaluate(test_loader=test_loader, save=True, model=model,epoch=epoch,best_acc_=best_acc_,args=args)
# save results
    print('train acc on train images is ', train_acc)
    print('test acc on test images is ', test_acc)

    with open(txtfile, "a") as myfile:
        myfile.write(str(int(epoch)) + ': '  + str(train_acc) +' ' + str(test_acc) + "\n")
