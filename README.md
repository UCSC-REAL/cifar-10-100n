# Dataloader for CIFAR-N

### CIFAR-10N 
```python
noise_label = torch.load('./data/CIFAR-10_human.pt')
clean_label = noise_label['clean_label']
worst_label = noise_label['worse_label']
aggre_label = noise_label['aggre_label']
random_label1 = noise_label['random_label1']
random_label2 = noise_label['random_label2']
random_label3 = noise_label['random_label3']
```

### CIFAR-100N 
```python
noise_label = torch.load('./data/CIFAR-100_human.pt')
clean_label = noise_label['clean_label']
noisy_label = noise_label['noisy_label']
```

# Training on CIFAR-N with the Cross-Entropy loss
### CIFAR-10N 
```shell
# NOISE_TYPE: [clean, aggre, worst, rand1, rand2, rand3]
# Use human annotations
CUDA_VISIBLE_DEVICES=0 python3 main.py --dataset cifar10 --noise_type NOISE_TYPE --is_human
# Use the synthetic noise that has the same noise transition matrix as human annotations
CUDA_VISIBLE_DEVICES=0 python3 main.py --dataset cifar10 --noise_type NOISE_TYPE
```

### CIFAR-100N 
```shell
# NOISE_TYPE: [clean100, noisy100]
# Use human annotations
CUDA_VISIBLE_DEVICES=0 python3 main.py --dataset cifar100 --noise_type NOISE_TYPE --is_human
# Use the synthetic noise that has the same noise transition matrix as human annotations
CUDA_VISIBLE_DEVICES=0 python3 main.py --dataset cifar100 --noise_type NOISE_TYPE
```
