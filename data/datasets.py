import numpy as np
import copy
import torchvision.transforms as transforms
from .cifar import CIFAR10, CIFAR100



train_cifar10_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4), 
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_cifar10_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_cifar100_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

test_cifar100_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

def input_dataset(dataset, noise_type, noise_path, is_human, val_ratio = 0.1):
    num_training_samples = 50000
    idx_full = np.arange(num_training_samples)
    np.random.shuffle(idx_full)
    if dataset == 'cifar10':
        num_classes = 10

        train_dataset_full = CIFAR10(root='~/data/',
                                download=True,  
                                train=True, 
                                transform = train_cifar10_transform,
                                noise_type = noise_type,
                                noise_path = noise_path, 
                                is_human=is_human,
                           )

        test_dataset = CIFAR10(root='~/data/',
                                download=False,  
                                train=False, 
                                transform = test_cifar10_transform,
                                noise_type=noise_type,
                          )
        
        
    elif dataset == 'cifar100':
        num_classes = 100
        train_dataset_full = CIFAR100(root='~/data/',
                                download=True,  
                                train=True, 
                                transform=train_cifar100_transform,
                                noise_type=noise_type,
                                noise_path = noise_path, is_human=is_human
                            )



        test_dataset = CIFAR100(root='~/data/',
                                download=False,  
                                train=False, 
                                transform=test_cifar100_transform,
                                noise_type=noise_type
                            )
        
    train_dataset = copy.copy(train_dataset_full)
    train_dataset.train_data = train_dataset.train_data[idx_full[int(num_training_samples*val_ratio):]]
    train_dataset.train_noisy_labels = (np.array(train_dataset.train_noisy_labels)[idx_full[int(num_training_samples*val_ratio):]]).tolist()
    print(f'Train with {len(train_dataset.train_noisy_labels)} noisy instances.')
    
    val_dataset = copy.copy(train_dataset_full)
    val_dataset.transform = test_cifar10_transform
    val_dataset.train_data = val_dataset.train_data[idx_full[:int(num_training_samples*val_ratio)]]
    val_dataset.train_noisy_labels = (np.array(val_dataset.train_noisy_labels)[idx_full[:int(num_training_samples*val_ratio)]]).tolist()
    print(f'Validate with {len(val_dataset.train_noisy_labels)} noisy instances.')
    return train_dataset, val_dataset, test_dataset, num_classes, num_training_samples








