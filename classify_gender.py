# import necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import cv2
import os

# handle command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-m", "--model", required=True,
	help="path to trained model file")
args = ap.parse_args()

# read input image
image = cv2.imread(args.image)

if image is None:
    print("Could not read input image")
    exit()

# preprocessing
output = np.copy(image)
image = cv2.resize(image, (96,96))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# load pre-trained model
model = load_model(args.model)

# run inference on input image
confidence = model.predict(image)[0]

# write predicted gender and confidence on image (top-left corner)
classes = ["man", "woman"]    
idx = np.argmax(confidence)
label = classes[idx]
label = "{}: {:.2f}%".format(label, confidence[idx] * 100)

cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)

# print confidence for each class in terminal
print(args.image)
print(classes)
print(confidence)

# display output image
cv2.imshow("Gender classification", output)

# press any key to close image window
cv2.waitKey()

# save output image
cv2.imwrite("gender-classification.jpg", output)

# release resources
cv2.destroyAllWindows()
