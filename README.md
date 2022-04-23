This repository is the official dataset release and Pytorch implementation of "[Learning with Noisy Labels Revisited: A Study Using Real-World Human Annotations](https://openreview.net/forum?id=TBWA6PLJZQm&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2022%2FConference%2FAuthors%23your-submissions))" accepted by ICLR2022. We collected and published re-annotated versions of the CIFAR-10 and CIFAR-100 data which contains real-world human annotation errors. We show how these noise patterns deviate from the classically assumed ones and what the new challenges are. The website of CIFAR-N is available at [http://www.noisylabels.com/](http://www.noisylabels.com/).

**Competition:** Please refer to the branch `ijcai-lmnl-2022` for details of 1st Learning with Noisy Labels Challenge in IJCAI 2022. Also available at [http://competition.noisylabels.com/](http://competition.noisylabels.com/).

# Dataloader for CIFAR-N (PyTorch)

### CIFAR-10N 
```python
import torch
noise_file = torch.load('./data/CIFAR-10_human.pt')
clean_label = noise_file['clean_label']
worst_label = noise_file['worse_label']
aggre_label = noise_file['aggre_label']
random_label1 = noise_file['random_label1']
random_label2 = noise_file['random_label2']
random_label3 = noise_file['random_label3']
```

### CIFAR-100N 
```python
import torch
noise_file = torch.load('./data/CIFAR-100_human.pt')
clean_label = noise_file['clean_label']
noisy_label = noise_file['noisy_label']
```

# Dataloader for CIFAR-N (Tensorflow)

Note: Image order of tensorflow dataset (tfds.load, binary version of CIFAR) does not match with PyTorch dataloader (python version of CIFAR).

### CIFAR-10N 
```python
import numpy as np
noise_file = np.load('./data/CIFAR-10_human_ordered.npy', allow_pickle=True)
clean_label = noise_file.item().get('clean_label')
worst_label = noise_file.item().get('worse_label')
aggre_label = noise_file.item().get('aggre_label')
random_label1 = noise_file.item().get('random_label1')
random_label2 = noise_file.item().get('random_label2')
random_label3 = noise_file.item().get('random_label3')
# The noisy label matches with following tensorflow dataloader
train_ds, test_ds = tfds.load('cifar10', split=['train','test'], as_supervised=True, batch_size = -1)
train_images, train_labels = tfds.as_numpy(train_ds) 
# You may want to replace train_labels by CIFAR-N noisy label sets
```

### CIFAR-100N 
```python
import numpy as np
noise_file = np.load('./data/CIFAR-100_human_ordered.npy', allow_pickle=True)
clean_label = noise_file.item().get('clean_label')
noise_label = noise_file.item().get('noise_label')
# The noisy label matches with following tensorflow dataloader
train_ds, test_ds = tfds.load('cifar100', split=['train','test'], as_supervised=True, batch_size = -1)
train_images, train_labels = tfds.as_numpy(train_ds) 
# You may want to replace train_labels by CIFAR-N noisy label sets
```

The image order from tfds to pytorch dataloader is given below:
- **image_order_c10.npy:** a numpy array with length 50K, the i-th element denotes the index of i-th unshuffled tfds (binary-version) CIFAR-10 training image in the Pytorch (python-version) ones.
- **image_order_c100.npy:** a numpy array with length 50K, the i-th element denotes the index of i-th unshuffled tfds (binary-version) CIFAR-100 training image in the Pytorch (python-version) ones.


# Training on CIFAR-N with Cross-Entropy (PyTorch)
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

# Additional dataset information
We include additional side information during the noisy-label collection in <code>side_info_cifar10N.csv</code> and <code>side_info_cifar100N.csv</code>.
A brief introduction of these two files:
- **Image-batch:** a subset of indexes of the CIFAR training images.
- **Worker-id:** the encrypted worker id on Amazon Mechanical Turk.
- **Work-time-in-seconds:** the time (in seconds) a worker spent on annotating the corresponding image batch.
