# Flower Species Recognition System #

This repo contains the code for conference paper titled **Flower Species Recognition System using Convolutional Neural Networks and Transfer Learning**, by I.Gogul and V.Sathiesh Kumar, Proceedings of ICSCN-2017 conference, IEEE Xplore Digital Library (Presented, to be uploaded).

### Summary of the project ###

* Pretrained state-of-the-art neural networks are used on *University of Oxford's* **FLOWERS17** and **FLOWERS102** dataset.
* Models used     - **Xception, Inception-v3, OverFeat, ResNet50, VGG16, VGG19**.
* Weights used    - **ImageNet**
* Classifier used - **Logistic Regression**
* Tutorial for this work is available at - [Flower Recognition using Deep Learning](https://gogul09.github.io/flower-recognition-deep-learning/).

### Dependencies ###
* Theano or TensorFlow `sudo pip install theano` or `sudo pip install tensorflow`
* Keras `sudo pip install keras`
* NumPy `sudo pip install numpy`
* matplotlib `sudo pip install matplotlib` and you also need to do this `sudo apt-get install python-dev`
* seaborn `sudo pip install seaborn`
* h5py `sudo pip install h5py`
* scikit-learn `sudo pip install scikit-learn`
* cPickle (already installed with Python 2.7 and Python 3.4)

### System requirements
* This project used Windows 10 for development purposes and Odroid-XU4 for testing purposes.

### Licence
MIT License

### Usage ###
* Organize dataset                      - `python organize_flowers17.py`
* Feature extraction using CNN          - `python extract_features.py`
* Train model using Logistic Regression - `python train.py`

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
| Xception     | 93.38%          | **99.75%**      |
| Inception-v3 | **96.81%**      | 99.51%          |
| VGG16        | 88.24%          | 99.02%          |
| VGG19        | 88.73%          | 98.77%          |
| ResNet50     | 59.80%          | 86.52%          |

### Conclusion ###
* **Inception-v3** and **Xception** outperformed all the other architectures.
* This could be due to the presence of **network-in-a-network** architecture codenamed as **Inception** module.
