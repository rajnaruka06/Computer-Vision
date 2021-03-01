## Trains face recognizer used in Face Recognition Try 2


import numpy as np
import os
from PIL import Image  ## Python Inage Library  ##Pillow
import cv2
import pickle

## face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")   ## Would be needed if images have more than face
recognizer = cv2.face.LBPHFaceRecognizer_create()
Base_Dir = os.path.dirname(os.path.abspath(__file__))  ## Path where the "Faces_Train.py" file is saved
Image_Dir = os.path.join(Base_Dir, "Images")

current_id = 0
label_ids={}  ## Dict of label id and labels
x_train = []
y_labels = []

for root, dirs, files in os.walk(Image_Dir):

	for file in files:

		## If file.endswith("png") or file.endswith("jpg")  ## DO this when we have other data as well

		path = os.path.join(root, file)
		label = os.path.basename(os.path.dirname(path)).replace(" ", "_").lower()  ## String function to better unerstand ## Not necessory

		## print(label, path)

		## Map labels with label ids
		if label in label_ids:
			pass
		else:
			label_ids[label] = current_id
			current_id += 1

		id_ = label_ids[label]
		## x_train.append(path) ## Varify this image, Turn into numpy array, convert into gray.
		## y_train.append(label)  ## Need some Number

		pil_image = Image.open(path).convert("L")  ## Get Image, Convert into grayscale
		image_array = np.array(pil_image, "uint8") ## Get numpy array from image

		## Faces = face_cascade(image_array, scaleFactor=1.5, minNeighbors=5)
		## Now, get the face and append into x_train

		x_train.append(image_array)		
		y_labels.append(id_)


with open("labels.pickle", "wb") as f :
	pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("face_trainer.yml")


