import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime ## THis is used to mark attendence w.r.t to time

#Here we'll ask our prog to find this folder & find the no. of images it has and then import it & find its encodings
path = '/Users/meetujamwal/Desktop/pythonProject/images'
##List (of all imgs we import)
images = []
img=[]
## NAMES OF ALL THESE IMGS (instead of mannually writting names)
classNames = []
## To grab the list of imgs in this folder
myList = os.listdir(path)
print(myList)

## Using these names and importing the images one by one:
for cl in myList:              ##Importing each classes
    currentImg = cv2.imread(f'{path}/{cl}')     ##read the current img
    images.append(currentImg)
    classNames.append(os.path.splitext(cl)[0])  ## This gives us the first name instead of whole filename
print(classNames)

## Encoding Process
# Creating a simple function that computes all the encodings for us

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        print(encode)
        encodeList.append(encode)
    return encodeList

## To mark the attendence:

def markAttendence(name):
    with open ('Attendence.csv','r+') as f: ## Opening the csv file & to read and write the the files we use 'r+'
        ## We do this so that one person only gets marked one time and not multiple times after he arrives
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            tString = now.strftime('%H: %M: %S')
            dString = now.strftime('%m/%d/%Y')
            f.writelines(f'\n{name},{tString},{dString}')

encodeListKnown = findEncodings(images)
print(len(encodeListKnown))
print('Encoding Complete')

# (Step-3): Finding the matches b/w our encodings
## To get this image we have to initialize the webcam
### Initializing the webcam:

cap = cv2.VideoCapture(0)

## while loop to get each frame one-by-one
while True:
    rval,img = cap.read()    # Gives us the img
    small_img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    rgb_small_img = small_img[:, :, ::-1]
    # success,img = cap.read()
    # rval, frame = cap.read()
    # if frame is not None:
    #     cv2.imshow("preview", frame)
    # rval, frame = cap.read()
    #
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
# Bcoz we r doing this in real-time, we want to reduce the size of our img, coz it'll help us in speeding the process
#     imgS = cv2.resize(img,(0,0),None,0.25,0.25)  # Reduces the size of the img
#     rgb_small_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    # Convert into RGB
    ## In the webcam we might actually find multiple faces, so for that we r going to find the location of these faces and then we r going to send these locations to our encoding function
    facesCurFrame = face_recognition.face_locations(rgb_small_img)
## NEXT STEP: is to find the encoding of the webcam
    encodesCurFrame = face_recognition.face_encodings(rgb_small_img,facesCurFrame)

## NEXT STEP: Finding the matches: We'll iterate through all the faces tht we have found in our current frame and then we'll compare all these faces w/ all the encodings we have found before

    # So to LOOP THROUGH we're going to use both (facesCurFrame & encodesCurFrame) of these lists
    # And to loop through them together, we can write:

    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame): # 1ST Thing we'll find is the encoding
        ## This for will grab one-by-one the face locations from the faces current frame list,
        ## And then it'll grab the encoding of encodeFace from encodesCurFrame
        ## So bcoz we want them in same loop, that's why we use ZIP.
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDis)
        matchindex = np.argmin(faceDis)

        if matches[matchindex]:
            name = classNames[matchindex].upper()
            print(name)
            top,bottom,left,right = faceLoc
            top, bottom, left, right = top*4,bottom*4,left*4,right*4
            # Draw a box around the face
            cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
            # Draw a label with a name below the face
            cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(img, name, (left + 6, bottom - 6), font, 0.67, (255, 255, 255), 1)
            markAttendence(name)
            # y1,x2,y2,x1 = faceLoc
            # y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            # cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            # cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            # cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
        else:
            name = "Unknown"
            print(name)
            top, bottom, left, right = faceLoc
            top, bottom, left, right = top * 4, bottom * 4, left * 4, right * 4
            # Draw a box around the face
            cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
            # Draw a label with a name below the face
            cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(img, name, (left + 6, bottom - 6), font, 0.67, (255, 255, 255), 1)
            markAttendence(name)


    ## Creating the bonding box and mentioning its name
    cv2.imshow('Webcam',img)
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # for (top, right, bottom, left), name in zip(encodesCurFrame, facesCurFrame):
    #     # Scale back up face locations since the frame we detected in was scaled to 1/4 size
    #     top *= 4
    #     right *= 4
    #     bottom *= 4
    #     left *= 4
    #
    #     # Draw a box around the face
    #     cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
    #
    #     # Draw a label with a name below the face
    #     cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
    #     font = cv2.FONT_HERSHEY_DUPLEX
    #     cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    #
    # # Display the resulting image
    #     cv2.imshow('Video', img)
    #
    # # Hit 'q' on the keyboard to quit!
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break






# loading the images and converting into rgb

# imgElon = face_recognition.load_image_file('images/elon.jpeg')  #using face_recognition lib to load image file
# imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)        #converting into RGB (from BGR)
# imgTest = face_recognition.load_image_file('images/elon-test.jpeg')
# imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)
# #loading every image will be difficult so we'll just make the list that can get the imags from our folder automatically and create the encodings automatically
# # It'll also detect it in the webcam
