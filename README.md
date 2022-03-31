# 1st Learning with Noisy Labels Challenge
Table of Contents
=================

- [1st Learning with Noisy Labels Challenge](#1st-learning-with-noisy-labels-challenge)
- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [This Branch](#this-branch)
- [General Requirements for the Competition](#general-requirements-for-the-competition)
- [Tasks and Evaluation Policy](#tasks-and-evaluation-policy)
  - [Learning](#learning)
    - [Background](#background)
    - [Goal](#goal)
    - [Evaluation metric `learning.py`](#evaluation-metric-learningpy)
    - [Note](#note)
  - [Detection](#detection)
    - [Background](#background-1)
    - [Goal](#goal-1)
    - [Evaluation metric `detection.py`](#evaluation-metric-detectionpy)
    - [Note](#note-1)
- [Submission Policy](#submission-policy)
  - [Code submission and evaluation](#code-submission-and-evaluation)
  - [Report submission and evaluation](#report-submission-and-evaluation)
  - [Dual submission](#dual-submission)
- [Quick Start](#quick-start)
- [Dataset information](#dataset-information)
  - [Dataloader for CIFAR-N (PyTorch)](#dataloader-for-cifar-n-pytorch)
    - [CIFAR-10N](#cifar-10n)
    - [CIFAR-100N](#cifar-100n)
  - [Dataloader for CIFAR-N (Tensorflow)](#dataloader-for-cifar-n-tensorflow)
    - [CIFAR-10N](#cifar-10n-1)
    - [CIFAR-100N](#cifar-100n-1)
  - [Additional dataset information](#additional-dataset-information)

# Introduction

This repository is the official dataset release and Pytorch implementation of "[Learning with Noisy Labels Revisited: A Study Using Real-World Human Annotations](https://openreview.net/forum?id=TBWA6PLJZQm&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2022%2FConference%2FAuthors%23your-submissions))" accepted by ICLR2022. We collected and published re-annotated versions of the CIFAR-10 and CIFAR-100 data which contains real-world human annotation errors. We show how these noise patterns deviate from the classically assumed ones and what the new challenges are. The website of CIFAR-N is available at [http://www.noisylabels.com/](http://www.noisylabels.com/).


# This Branch

This branch provides example codes and policies for the 1st learning with noisy labels competition. We have two tasks: 1) Learning and 2) Detection.





# General Requirements for the Competition
- Participants can only use standard training images for CIFAR and the CIFAR-N noisy training labels.
- Can not use CIFAR-published training labels, test images, and test labels, to perform training and model selection.
- Participants can use our provided [additional information](#additional-dataset-information). 
- Learn from scratch. Any pre-trained model cannot be used.


# Tasks and Evaluation Policy
There are two tasks: learning and detection. Each task includes two subcompetitions: CIFAR-10 and CIFAR-100. Thus there are four subcompetitions. Each participant can choose to participate in any subcompetitions.   

## Learning
### Background
Image classification task in deep learning requires assigning labels to specific images. Annotating labels for training use often requires tremendous expenses on the payment for hiring human annotators. The pervasive noisy labels from data annotation present significant challenges to training a quality machine learning model.

### Goal
The goal of this task is to explore the potential of AI approaches when we only have access to human annotated noisy labels (CIFAR-N). Specifically, for each label-noise setting, the proposed method will train only on the noisy labels with the corresponding training images. The evaluation is based on the accuracy of the trained models on the CIFAR test datasets.

This task does not have specific requirements on the experiment settings, i.e., the model architecture, data augmentation strategies, etc. However, the use of clean labels or pre-trained models on CIFAR datasets,  is not allowed.    

### Evaluation metric `learning.py`
Each submission will be evaluated according to the model's achieved accuracy on the corresponding CIFAR-10/100 test data: denote by $h$ the final model from the submission. We will use  
    $$
    \frac{\sum_{(X,Y) \in \text{CIFAR Test data}} \mathbb 1(h(X) = Y)}{|\text{Test data}|}
    $$
    to make the final evaluation. 

### Note
 The hyperparameter settings should be consistent for different noise regimes in the same dataset, i.e., there will be at most two sets of hyperparameters, one for CIFAR-10N (aggre, rand1, \& worst), one for CIRFAR-100N.


## Detection

### Background
Label noise in real-world datasets encodes wrong correlation patterns and impairs the generalization of deep neural networks (DNNs). Employing human workers to clean annotations is one reliable way to improve the label quality, but it is too expensive and time-consuming for a large-scale dataset. One promising way to automatically clean up label errors is to first algorithmically detect possible label errors from a large-scale dataset, and then correct them using either algorithm or crowdsourcing.

### Goal
The goal is to encourage the design of an algorithmic detection approach to improving the corrupted label detection (a.k.a, finding label errors) on CIFAR-N.

This task does not have specific requirements on the experiment settings, i.e., the model architecture, data augmentation strategies, etc. However, the use of clean labels or pre-trained models on CIFAR datasets,  is not allowed.    

### Evaluation metric `detection.py`
The performance is measured by the $F_1$-score of the detected corrupted instances, which is the harmonic mean of the precision and recall, i.e. 
$$F_1=\frac{2}{{\tt Precision}^{-1} + {\tt Recall}^{-1}}.$$
Let $\mathbb 1(\cdot)$ be the indicator function that takes value $1$ when the specified condition is satisfied and $0$ otherwise. Let $v_n=1$ indicate that $\tilde y_n$ is detected as a corrupted label, and $v_n=0$ if $\tilde y_n$ is detected to be clean. Then the precision and recall of finding corrupted labels can be calculated as 
$$
{\tt Precision} = \frac{\sum_{n\in[N]}\mathbb 1(v_n=1, \tilde y_n \ne y_n)}{\sum_{n\in[N]}\mathbb 1(v_n=1)}, ~{\tt Recall}=\frac{\sum_{n\in[N]}\mathbb 1(v_n=1, \tilde y_n \ne y_n)}{\sum_{n\in[N]} \mathbb 1 (\tilde y_n \ne y_n)}.
$$

### Note
The hyperparameter settings should be consistent for different clusters in the same dataset, i.e., there will be at most two sets of hyperparameters, one for CIFAR-10N, one for CIRFAR-100N. 





# Submission Policy
## Code submission and evaluation
- Participants must submit reproducible code with a downloadable link, e.g., GitHub.
- The script `run.sh` for running the code must be provided. 
- Environments must be specified in `requirements.txt`.
- We will run `run.sh` with 5 pre-fixed seeds and take the average performance.
- For CIFAR-10, there are three noise types: `rand1, worst, aggre`.
Each participant will receive three ranks. No submission equals the last rank. The scores $s$ for rank $i$ is $s = \max(10 - i, 0)$. The accumulated scores over three noise regimes determine the final score.
- For CIFAR-100, there is only one dataset. The average performance over 5 seeds determines the winner.
- We will test the performance by `learning.py` for the learning task or `detection.py` for the detection task.
- **IMPORTANT:** This competition is time-constrained. We do not recommend spending too much time on CIFAR. Thus the training will be stopped at `3xBaselineTime`. The baseline code (train with cross-entropy and ResNet34) is available at `ce_baseline.py`. For example, if you take 1 hour to run `ce_baseline.py` in your device, your method should not be longer than 3 hours. We will use the best model selected by noisy validation data within `3xBaselineTime`.


## Report submission and evaluation
- We use EasyChair for submission. Link is [here](https://easychair.org/conferences/?conf=1stlnlc).
- A report is preferred, which will be reviewed by our reviewing committee. We have one **Best Innovation Award** for the best report. 
- We expect the report to be no more than 8 pages using [IJCAI template](https://www.ijcai.org/authors_kit).
- If you are not willing to submit a report, please submit the link to your code to EasyChair.


## Dual submission
It is not appropriate to submit codes/reports that are identical (or substantially similar) to versions that are also submitted to this competition. In other words, please DO NOT make multiple submissions by simply changing hyperparameters to improve the chance of getting awarded. Such submissions violate our dual submission policy, and the organizers have the right to reject such submissions. But the codes/reports/papers can be previously published, accepted for publication, or submitted in parallel to other conferences or journals.


<!-- # Award
-Winners (for each subcompetition): \$1000.

-Runner ups (for each subcompetition): \$500.

-Best innovation award (one award): \$??? -->


# Quick Start
```shell
bash run.sh
```

------------------------------------------------------------------
# Dataset information

## Dataloader for CIFAR-N (PyTorch)

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

## Dataloader for CIFAR-N (Tensorflow)

Note: image order of tensorflow dataset does not match with CIFAR-N (PyTorch dataloader)
TODO: (1) Customize tensorflow dataloader with ziped images; (2) Similarity comparisons among images to obtain the order mapping.

### CIFAR-10N 
```python
import numpy as np
noise_file = np.load('./data/CIFAR-10_human.npy', allow_pickle=True)
clean_label = noise_file.item().get('clean_label')
worst_label = noise_file.item().get('worse_label')
aggre_label = noise_file.item().get('aggre_label')
random_label1 = noise_file.item().get('random_label1')
random_label2 = noise_file.item().get('random_label2')
random_label3 = noise_file.item().get('random_label3')
```

### CIFAR-100N 
```python
import numpy as np
noise_file = np.load('./data/CIFAR-100_human.npy', allow_pickle=True)
clean_label = noise_file.item().get('clean_label')
noise_label = noise_file.item().get('noise_label')
```

  


## Additional dataset information
We include additional side information during the noisy-label collection in <code>side_info_cifar10N.csv</code> and <code>side_info_cifar100N.csv</code>.
A brief introduction of these two files:
- **Image-batch:** a subset of indexes of the CIFAR training images.
- **Worker-id:** the encrypted worker id on Amazon Mechanical Turk.
- **Work-time-in-seconds:** the time (in seconds) a worker spent on annotating the corresponding image batch.
