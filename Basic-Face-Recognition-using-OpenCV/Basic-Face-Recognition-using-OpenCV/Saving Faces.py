import cv2
import numpy as np

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")   

count=0
while True:

	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, scaleFactor=2.5, minNeighbors=10)
	for x,y,w,h in faces:
		
		## print(x,y,w,h)
		## Refion Of Interest :: WHere the face is
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = frame[y:y+h, x:x+w]
		img_item = "Images//My_cascade_img{}.png".format(count)
		count+=1
		cv2.imwrite(img_item, roi_color)

		## Draw Rectangle
		color = (255, 0, 0)  ## BGR: 0-255
		stroke = 2
		x_end = x+w
		y_end = y+h
		cv2.rectangle(frame, (x,y), (x_end, y_end), color, stroke)
		
		count+=1
		
	cv2.imshow("image", frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()