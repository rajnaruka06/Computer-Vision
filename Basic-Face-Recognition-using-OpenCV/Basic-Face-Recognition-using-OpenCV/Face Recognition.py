import cv2
import numpy as np
import pickle


cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")   
## eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")   
## smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")   
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_trainer.yml")
labels = {}
with open("labels.pickle", "rb") as f :
	labels = pickle.load(f)
	labels = {v:k for k,v in labels.items()}   ## Need to do this because in "Face_Train.py" While creting ditionary we mapped label:id not id:label

while True:

	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
	for x,y,w,h in faces:
		
		## print(x,y,w,h)
		## Refion Of Interest :: WHere the face is
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = frame[y:y+h, x:x+w]

		## Recognizer was trained on grayscale images:

		id_, conf = recognizer.predict(roi_gray)
		if conf>=45:
			name = labels[id_]
			print(name)
			font = cv2.FONT_HERSHEY_DUPLEX
			color = (200,105,25)
			stroke = 2
			frame = cv2.putText(frame, name, (x,y), font, stroke, color)

		## Draw Rectangle
		color = (25, 0, 50)  ## BGR: 0-255
		stroke = 2
		x_end = x+w
		y_end = y+h
		cv2.rectangle(frame, (x,y), (x_end, y_end), color, stroke)

		## eyes = eye_cascade.detectMultiScale(roi_gray)
		## for ex, ey, ew, eh in eyes:
		## 	cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), color, stroke)
		## smile = smile_cascade.detectMultiScale(roi_gray)
		## for sx, sy, sw, sh in smile:
		## 	cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), color, stroke)
		## Display Resulting Frame
	
	cv2.imshow("image", frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()