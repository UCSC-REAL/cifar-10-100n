CUDA_VISIBLE_DEVICES=0 nohup python3 ce_baseline.py --dataset cifar10 --noise_type worst --val_ratio 0.1 > c10_worst.log &  

CUDA_VISIBLE_DEVICES=0 nohup python3 ce_baseline.py --dataset cifar10 --noise_type rand1 --val_ratio 0.1 > c10_rand1.log &  

CUDA_VISIBLE_DEVICES=1 nohup python3 ce_baseline.py --dataset cifar10 --noise_type aggre --val_ratio 0.1 > c10_aggre.log & 

CUDA_VISIBLE_DEVICES=1 nohup python3 ce_baseline.py --dataset cifar100 --noise_type noisy100 --val_ratio 0.1 > c100.log & 