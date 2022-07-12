# -*- coding:utf-8 -*-


from data.datasets import input_dataset
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type = float, default = 0.1)
parser.add_argument('--val_ratio', type = float, default = 0.1)
parser.add_argument('--noise_type', type = str, help='clean, aggre, worst, rand1, rand2, rand3, clean100, noisy100', default='clean')
parser.add_argument('--noise_path', type = str, help='path of CIFAR-10_human.pt', default=None)
parser.add_argument('--dataset', type = str, help = ' cifar10 or cifar100', default = 'cifar10')
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--seed', type=int, default=10)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')




def evaluate(noisy_or_not_predict, noisy_or_not_real):
    precision = np.sum(noisy_or_not_predict * noisy_or_not_real)/np.sum(noisy_or_not_predict)
    recall = np.sum(noisy_or_not_predict * noisy_or_not_real)/np.sum(noisy_or_not_real)
    fscore = 2.0 * precision * recall / (precision + recall)


    return precision, recall, fscore


##################################### main code ################################################
args = parser.parse_args()

set_global_seeds(args.seed)
learning_rate = args.lr
noise_type_map = {'clean':'clean_label', 'worst': 'worse_label', 'aggre': 'aggre_label', 'rand1': 'random_label1', 'rand2': 'random_label2', 'rand3': 'random_label3', 'clean100': 'clean_label', 'noisy100': 'noisy_label'}
args.noise_type = noise_type_map[args.noise_type]
# load dataset
if args.noise_path is None:
    if args.dataset == 'cifar10':
        args.noise_path = './data/CIFAR-10_human.pt'
    elif args.dataset == 'cifar100':
        args.noise_path = './data/CIFAR-100_human.pt'
    else: 
        raise NameError(f'Undefined dataset {args.dataset}')


train_dataset, val_dataset, test_dataset, num_classes, num_training_samples = input_dataset(args.dataset,args.noise_type, args.noise_path, args.seed, is_human = True, val_ratio = 0.0)

YOUR_RESULT = np.load('detection.npy')
noisy_or_not_predict = YOUR_RESULT # should be your result. N-dim boolean numpy array

# # The following two lines show one toy example.
# noisy_or_not_predict = train_dataset.noise_or_not.copy()
# noisy_or_not_predict[25000:] = False
# np.save('detection.npy', noisy_or_not_predict)
# # # Should return: 
# # precision is 1.0
# # recall is 0.5045264623955432
# # The f-score is 0.6706780837769035


precision, recall, fscore = evaluate(noisy_or_not_predict, train_dataset.noise_or_not)
print(f'precision is {precision}')
print(f'recall is {recall}')
print(f'The f-score is {fscore}')



