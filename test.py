# test script to preform prediction on test images inside 
# dataset/test/
#   -- image_1.jpg
#   -- image_2.jpg
#   ...

# organize imports
from __future__ import print_function

# keras imports
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.applications.xception import Xception, preprocess_input
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.models import model_from_json
from keras.layers import Input

# other imports
from sklearn.linear_model import LogisticRegression
import numpy as np
import os
import json
import pickle
import cv2

# load the user configs
with open('conf/conf.json') as f:    
	config = json.load(f)

# config variables
model_name 		= config["model"]
weights 		= config["weights"]
include_top 	= config["include_top"]
train_path 		= config["train_path"]
test_path 		= config["test_path"]
features_path 	= config["features_path"]
labels_path 	= config["labels_path"]
test_size 		= config["test_size"]
results 		= config["results"]
model_path 		= config["model_path"]
seed 			= config["seed"]
classifier_path = config["classifier_path"]

# load the trained logistic regression classifier
print ("[INFO] loading the classifier...")
classifier = pickle.load(open(classifier_path, 'rb'))

# pretrained models needed to perform feature extraction on test data too!
if model_name == "vgg16":
	base_model = VGG16(weights=weights)
	model = Model(input=base_model.input, output=base_model.get_layer('fc1').output)
	image_size = (224, 224)
elif model_name == "vgg19":
	base_model = VGG19(weights=weights)
	model = Model(input=base_model.input, output=base_model.get_layer('fc1').output)
	image_size = (224, 224)
elif model_name == "resnet50":
	base_model = ResNet50(weights=weights)
	model = Model(input=base_model.input, output=base_model.get_layer('flatten').output)
	image_size = (224, 224)
elif model_name == "inceptionv3":
	base_model = InceptionV3(include_top=include_top, weights=weights, input_tensor=Input(shape=(299,299,3)))
	model = Model(input=base_model.input, output=base_model.get_layer('custom').output)
	image_size = (299, 299)
elif model_name == "inceptionresnetv2":
	base_model = InceptionResNetV2(include_top=include_top, weights=weights, input_tensor=Input(shape=(299,299,3)))
	model = Model(input=base_model.input, output=base_model.get_layer('custom').output)
	image_size = (299, 299)
elif model_name == "mobilenet":
	base_model = MobileNet(include_top=include_top, weights=weights, input_tensor=Input(shape=(224,224,3)), input_shape=(224,224,3))
	model = Model(input=base_model.input, output=base_model.get_layer('custom').output)
	image_size = (224, 224)
elif model_name == "xception":
	base_model = Xception(weights=weights)
	model = Model(input=base_model.input, output=base_model.get_layer('avg_pool').output)
	image_size = (299, 299)
else:
	base_model = None

# get all the train labels
train_labels = os.listdir(train_path)

# get all the test images paths
test_images = os.listdir(test_path)

# loop through each image in the test data
for image_path in test_images:
	path 		= test_path + "/" + image_path
	img 		= image.load_img(path, target_size=image_size)
	x 			= image.img_to_array(img)
	x 			= np.expand_dims(x, axis=0)
	x 			= preprocess_input(x)
	feature 	= model.predict(x)
	flat 		= feature.flatten()
	flat 		= np.expand_dims(flat, axis=0)
	preds 		= classifier.predict(flat)
	prediction 	= train_labels[preds[0]]
	
	# perform prediction on test image
	print ("I think it is a " + train_labels[preds[0]])
	img_color = cv2.imread(path, 1)
	cv2.putText(img_color, "I think it is a " + prediction, (140,445), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
	cv2.imshow("test", img_color)

	# key tracker
	key = cv2.waitKey(0) & 0xFF
	if (key == ord('q')):
		cv2.destroyAllWindows()