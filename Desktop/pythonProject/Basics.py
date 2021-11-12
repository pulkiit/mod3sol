import cv2
import numpy as np
import face_recognition

# loading the images and converting into rgb

imgElon = face_recognition.load_image_file('images/elon.jpg')  #using face_recognition lib to load image file
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)        #converting into RGB (from BGR)
imgTest = face_recognition.load_image_file('images/elon-test.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgElon)[0]  # Step-2 # Finding the faces in our img & their encodings as well #since we r sending a single img we'll get only the 1st element of it
#gives us 4 different values, which are (top, right, bottom, left)
encodeElon = face_recognition.face_encodings(imgElon)[0] #encodes the face detected

#We're doing the face locations only to see where we have detected the faces, based on that we give x1,y1

cv2.rectangle(imgElon,(faceLoc[3], faceLoc[0]),(faceLoc[1], faceLoc[2]),(255,0,255),2) #This creates a rectangle on the detected faces on the img
#                               |  Face Locations  |                  Color of box  Thickness

#For test image

faceLoc = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLoc[3], faceLoc[0]),(faceLoc[1], faceLoc[2]),(255,0,255),2)

# Step-3:  Comparing these faces & finding the distance b/w them

#getting the encodings and compare them

results = face_recognition.compare_faces([encodeElon],encodeTest)
# print(results) #if the output shows [true], it means that it has found both these encodings to be similar, so they r of the same person


# Sometimes what happens is that there are a lot of images and there can be similarities, so what u have to find is how similar these imgs are
# So for that we find distance
faceDis = face_recognition.face_distance([encodeElon],encodeTest)
print(results,faceDis)  # Lower the distance the better is the match!

# Displaying on this result on the image:

cv2.putText(imgTest,f'{results}{round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)


cv2.imshow('Elon Musk',imgElon) # shows the output pf image files
cv2.imshow('Elon Test',imgTest)
cv2.waitKey(0)  #makes the waiting time = 0
