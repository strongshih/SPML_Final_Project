## Pytorch Adversarial Training on CIFAR-10

* This repository provides simple PyTorch implementations for adversarial training methods on CIFAR-10.
* This repository shows fine accuracies which are similar to the accuracies in the original papers.
* If you have questions about this repository, please send an e-mail to me (dongbinna@postech.ac.kr) or make an issue.

### Experiment Settings

* The basic experiment setting used in this repository follows the setting used in [Madry Laboratory](https://github.com/MadryLab/cifar10_challenge).
* Dataset: CIFAR-10 (10 classes)
* Attack method: PGD attack (Epsilon size is 0.0314 in L-infinity bound)
* Training batch size: 128
* Weight decay: 0.0002
* Momentum: 0.9
* Learning rate adjustment
  1) 0.1 for epoch [0, 100)
  2) 0.01 for epoch [100, 150)
  3) 0.001 for epoch [150, 200)
* The ResNet-18 architecture used in this repository is smaller than Madry Laboratory, but performance is similar.

### Training Methods

#### 1. Basic Training

* The basic training method adopts ResNet-18 architecture proposed by Kaiming He in [CVPR 2016](https://arxiv.org/pdf/1512.03385.pdf).
<pre>
python3 basic_training.py
</pre>
||This repository|Original paper (wide)|
|------|---|---|
|Benign accuarcy|95.28%|95.2%|
|Robust accuracy (L-infinity PGD)|1.02%|3.50%|
* Training time: 2 hours 24 minutes using 1 Titan XP
* [Trained model download: Basic Training](https://postechackr-my.sharepoint.com/:u:/g/personal/dongbinna_postech_ac_kr/EcpGMF03mR9Ko1MM-kMSmloB4ceabuYCvnHaGZPgLNMzrA?e=dMBBRA)

#### 2. PGD Adversarial Training

* This defense method was proposed by Aleksander Madry in [ICLR 2018](https://arxiv.org/pdf/1706.06083.pdf).
<pre>
python3 pgd_adversarial_training.py
</pre>
||This repository|Original paper (wide)|
|------|---|---|
|Benign accuarcy|83.53%|87.30%|
|Robust accuracy (L-infinity PGD)|46.07%|50.00%|
* Training time: 11 hours 12 minutes using 1 Titan XP
* [Trained model download: PGD Adversarial Training](https://postechackr-my.sharepoint.com/:u:/g/personal/dongbinna_postech_ac_kr/Efy7BpBGApRHi97u00A34t8BuNp_64Yswk5s_MPv2z15yA?e=RcL2iC)

#### 3. Interpolated Adversarial Training (IAT)

* This defense method was proposed by Alex Lamb in [AISec 2019](https://arxiv.org/pdf/1906.06784.pdf).
<pre>
python3 interpolated_adversarial_training.py
</pre>
||This repository|Original paper|
|------|---|---|
|Benign accuarcy|91.86%|89.88%|
|Robust accuracy (L-infinity PGD)|44.76%|44.57%|
* Training time: 15 hours 18 minutes using 1 Titan XP
* [Trained model download: Interpolated Adversarial Training](https://postechackr-my.sharepoint.com/:u:/g/personal/dongbinna_postech_ac_kr/EWP0H_Q21vZOvb6njchzHZkBshcdTxJXse17kNBk5H-qnA?e=ttYKts)

#### 4. Basic Training with Robust Dataset

* Normal dataset can be splited into robust dataset and non-robust dataset.
* Construction method for robust dataset proposed by Andrew Ilyas in [NIPS 2019](https://arxiv.org/pdf/1905.02175.pdf).
* [Dataset download: Robust Dataset](https://postechackr-my.sharepoint.com/:u:/g/personal/dongbinna_postech_ac_kr/ET9LWRoUc9ZCjU0-szWt55ABQepaeB64I8ZAruOlwNDQHg?e=FOmeb5)
<pre>
python3 basic_training_with_robust_dataset.py
</pre>
||This repository|Original paper (wide)|
|------|---|---|
|Benign accuarcy|78.69%|84.10%|
|Robust accuracy (L-infinity PGD)|37.96%|48.27%|
* [Trained model download: Basic Training with Robust Dataset](https://postechackr-my.sharepoint.com/:u:/g/personal/dongbinna_postech_ac_kr/EUzfqsw2k8VGkE92kdEWEtoB2AhznrfmVv-XPpo8NCn8QA?e=xKABMd)

#### 5. Basic Training with Non-robust Dataset

* Normal dataset can be splited into robust dataset and non-robust dataset.
* Construction method for non-robust dataset proposed by Andrew Ilyas in [NIPS 2019](https://arxiv.org/pdf/1905.02175.pdf).
* [Dataset download: Non-robust Dataset](https://postechackr-my.sharepoint.com/:u:/g/personal/dongbinna_postech_ac_kr/EZ9_ujc-biRFvVsjKU6QSk0BsiPma8kBpZDwSM20ryYqfg?e=bhpMYg)
<pre>
python3 basic_training_with_non_robust_dataset.py
</pre>
||This repository|Original paper (wide)|
|------|---|---|
|Benign accuarcy|82.00%|87.68%|
|Robust accuracy (L-infinity PGD)|0.10%|0.82%|
* [Trained model download: Basic Training with Non-robust Dataset](https://postechackr-my.sharepoint.com/:u:/g/personal/dongbinna_postech_ac_kr/ESxDKKWp_f5GtO2GBCCKJBsBDJSXmgfGaQDKp3jnLKg_nw?e=0eoRTq)

### How to Test

* The attack method is PGD attack (Epsilon size is 0.0314 in L-infinity bound).
* All pre-trained models are provided in this repository :)
<pre>
python3 test.py
</pre>
