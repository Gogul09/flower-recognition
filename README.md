# Flower Species Recognition System #

This repo contains the code for conference paper titled **Flower Species Recognition System using Convolutional Neural Networks and Transfer Learning**, by I.Gogul and V.Sathiesh Kumar, Proceedings of ICSCN-2017 conference, IEEE Xplore Digital Library (Presented, to be uploaded).

### Summary of the project ###

* Pretrained state-of-the-art neural networks are used on *University of Oxford's* **FLOWERS17** and **FLOWERS102** dataset.
* Models used     - **Xception, Inception-v3, OverFeat, ResNet50, VGG16, VGG19**.
* Weights used    - **ImageNet**
* Classifier used - **Logistic Regression**
* Tutorial for this work is available at - [Flower Recognition using Deep Learning](https://gogul09.github.io/flower-recognition-deep-learning/).

### Dependencies ###
* Theano or TensorFlow
* Keras
* NumPy
* matplotlib
* seaborn
* h5py
* scikit-learn
* cPickle

### Usage ###
* Organize dataset                      - `python organize_flowers17.py`
* Feature extraction using CNN          - `python extract_features.py`
* Train model using Logistic Regression - `python train.py`

### Conclusion ###
* **Inception-v3** and **Xception** outperformed all the other architectures.
* This could be due to the presence of **network-in-a-network** architecture codenamed as **Inception** module.

### Show me the numbers ###
The below tables shows the accuracies obtained for every Deep Neural Net model used to extract features from **FLOWERS17** dataset using different parameter settings.

* Result-1
  
  * test_size  : **0.10**
  * classifier : **Logistic Regression**
  
| Model        | Rank-1 accuracy | Rank-5 accuracy |
|--------------|-----------------|-----------------|
| Xception     | **97.06%**      | **99.26%**      |
| Inception-v3 | 96.32%          | **99.26%**      |
| VGG16        | 85.29%          | 98.53%          |
| VGG19        | 88.24%          | **99.26%**      |
| ResNet50     | 56.62%          | 90.44%          |

* Result-2
  
  * test_size  : **0.30**
  * classifier : **Logistic Regression**

| Model        | Rank-1 accuracy | Rank-5 accuracy |
|--------------|-----------------|-----------------|
| Inception-v3 | **96.81%**      | **99.51%**      |
| VGG16        | 88.24%          | 99.02%          |
| VGG19        | 88.73%          | 98.77%          |
| ResNet50     | 59.80%          | 86.52%          |
