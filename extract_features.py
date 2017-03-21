# filter warnings
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# keras imports
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.vgg19 import VGG19, preprocess_input
#from keras.applications.xception import Xception, preprocess_input 
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.models import model_from_json
import cPickle

# other imports
from sklearn.preprocessing import LabelEncoder
import numpy as np
import glob
import cv2
import h5py
import os
import json
import datetime

import time

# load the user configs
with open('conf/conf.json') as f:    
	config = json.load(f)

# config variables
model_name			= config["model"]
weights 			= config["weights"]
include_top 		= config["include_top"]
train_path 			= config["train_path"]
features_path		= config["features_path"]
labels_path 		= config["labels_path"]
test_size			= config["test_size"]
results				= config["results"]

# start time
print "[STATUS] start time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")) 
start = time.time()

# create the pretrained models 
# check for pretrained weight usage or not
# check for top layers to be included or not
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
	base_model = InceptionV3(weights=weights)
	model = Model(input=base_model.input, output=base_model.get_layer('flatten').output)
	image_size = (299, 299)
elif model_name == "xception":
	base_model = Xception(weights=weights)
	model = Model(input=base_model.input, output=base_model.get_layer('avg_pool').output)
	image_size = (299, 299)
else:
	base_model = None

print "[INFO] successfully loaded base model and model..."

# path to training dataset
train_labels = os.listdir(train_path)

# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
le.fit([tl for tl in train_labels])

# variables to hold features and labels
features = []
labels   = []

# loop over all the labels in the folder
i = 0
for label in train_labels:
	cur_path = train_path + "/" + label
	for image_path in glob.glob(cur_path + "/*.jpg"):
		img = image.load_img(image_path, target_size=image_size)
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
		feature = model.predict(x)
		flat = feature.flatten()
		features.append(flat)
		labels.append(label)
		print "[INFO] processed - {}".format(i)
		i += 1
	print "[INFO] completed label - {}".format(label)

# encode the labels using LabelEncoder
targetNames = np.unique(labels)
le = LabelEncoder()
le_labels = le.fit_transform(labels)

# get the shape of training labels
print "[STATUS] training labels: {}".format(le_labels)
print "[STATUS] training labels shape: {}".format(le_labels.shape)

# save features and labels 
h5f_data = h5py.File(features_path, 'w')
h5f_data.create_dataset('dataset_1', data=np.array(features))

h5f_label = h5py.File(labels_path, 'w')
h5f_label.create_dataset('dataset_1', data=np.array(le_labels))

h5f_data.close()
h5f_label.close()

# # save model and weights
# model_json = model.to_json()
# with open(model_path + str(test_size) + ".json", "w") as json_file:
# 	json_file.write(model_json)

# # save weights
# model.save_weights(model_path + str(test_size) + ".h5")
# print("[STATUS] saved model and weights to disk..")

print "[STATUS] features and labels saved.."

# end time
end = time.time()
print "[STATUS] end time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")) 