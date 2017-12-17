# organize imports
import os
import glob
import datetime

# print start time
print ("[INFO] program started on - " + str(datetime.datetime.now))

# get the input and output path
input_path  = "G:\\workspace\\machine-intelligence\\deep-learning\\flower-recognition\\17flowers\\jpg"
output_path = "G:\\workspace\\machine-intelligence\\deep-learning\\flower-recognition\\dataset\\train"

# get the class label limit
class_limit = 17

# take all the images from the dataset
image_paths = glob.glob(input_path + "\\*.jpg")

# variables to keep track
label = 0
i = 0
j = 80

# flower17 class names
class_names = ["daffodil", "snowdrop", "lilyvalley", "bluebell", "crocus",
			   "iris", "tigerlily", "tulip", "fritillary", "sunflower", 
			   "daisy", "coltsfoot", "dandelion", "cowslip", "buttercup",
			   "windflower", "pansy"]

# change the current working directory
os.chdir(output_path)

# loop over the class labels
for x in range(1, class_limit+1):
	# create a folder for that class
	os.system("mkdir " + class_names[label])
	# get the current path
	cur_path = output_path + "\\" + class_names[label] + "\\"
	# loop over the images in the dataset
	for image_path in image_paths[i:j]:
		original_path = image_path
		image_path = image_path.split("\\")
		image_path = image_path[len(image_path)-1]
		os.system("copy " + original_path + " " + cur_path + image_path)
	i += 80
	j += 80
	label += 1

# print end time
print ("[INFO] program ended on - " + str(datetime.datetime.now))