# Flower Species Recognition System #

![alt text](https://github.com/Gogul09/flower-recognition/blob/master/out.gif)

This repo contains the code for conference paper titled **Flower Species Recognition System using Convolutional Neural Networks and Transfer Learning**, by I.Gogul and V.Sathiesh Kumar, Proceedings of ICSCN-2017 conference, IEEE Xplore Digital Library.

### Summary of the project ###

* Pretrained state-of-the-art neural networks are used on *University of Oxford's* **FLOWERS17** and **FLOWERS102** dataset.
* Models used     - **Xception, Inception-v3, OverFeat, ResNet50, VGG16, VGG19**.
* Weights used    - **ImageNet**
* Classifier used - **Logistic Regression**
* Tutorial for this work is available at - [Using Pre-trained Deep Learning models for your own dataset](https://gogul09.github.io/software/flower-recognition-deep-learning)

**Update (16/12/2017)**: Included two new deep neural net models namely `InceptionResNetv2` and `MobileNet`.

### Dependencies ###
* Theano or TensorFlow `sudo pip install theano` or `sudo pip install tensorflow`
* Keras `sudo pip install keras`
* NumPy `sudo pip install numpy`
* matplotlib `sudo pip install matplotlib` and you also need to do this `sudo apt-get install python-dev`
* seaborn `sudo pip install seaborn`
* h5py `sudo pip install h5py`
* scikit-learn `sudo pip install scikit-learn`

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
| Xception     | 97.06%      	 | 99.26%      	   |
| Inception-v3 | 96.32%          | 99.26%          |
| VGG16        | 85.29%          | 98.53%          |
| VGG19        | 88.24%          | 99.26%          |
| ResNet50     | 56.62%          | 90.44%          |
| MobileNet     | <b>98.53%</b>          | <b>100.00%</b>         |
| Inception<br>ResNetV2     | 91.91%          | 98.53%          |
* Result-2
  
  * test_size  : **0.30**
  * classifier : **Logistic Regression**

| Model        | Rank-1 accuracy | Rank-5 accuracy |
|--------------|-----------------|-----------------|
| Xception     | 93.38%          | <b>99.75%</b>          |
| Inception-v3 | <b>96.81%</b>          | 99.51%          |
| VGG16        | 88.24%          | 99.02%          |
| VGG19        | 88.73%          | 98.77%          |
| ResNet50     | 59.80%          | 86.52%          |
| MobileNet     | 96.32%         | <b>99.75%</b>         |
| Inception<br>ResNetV2     | 88.48%          | 99.51%          |
