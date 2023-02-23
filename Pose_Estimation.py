import cvzone
import cv2
import mediapipe as mp
import socket
import numpy as np
import json

mpPose = mp.solutions.pose
pose = mpPose.Pose(static_image_mode=False,
                                     smooth_landmarks=False,
                                     min_detection_confidence=0.7,
                                     min_tracking_confidence=0.6)
mpDraw = mp.solutions.drawing_utils
fpsReader = cvzone.FPS()

cap = cv2.VideoCapture(0)

############ Communication ##########
dataSocket=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
serverAddressPort=("127.0.0.1",8080)


while True:
    success, img = cap.read()

###### Flipping the image to fix left and right move like a mirror #####
    img = cv2.flip(img,1)  

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    data=([])
    if(results.pose_landmarks):
        land_mark = results.pose_landmarks.landmark
        res_dicts = list(land_mark)

        for j in range(0 , len(res_dicts)):
            if (j == 0 or j == 19 or j == 20):
                
                myDic = {
                    "visibility" : str(res_dicts[j].visibility) ,
                    "x" : str(res_dicts[j].x),
                    "y" : str(res_dicts[j].y),
                    "z" : str(res_dicts[j].z)
                }
                myDic2 = {
                    "pose" : str(j),
                    "data": str(myDic)
                }
                data.append(myDic2)
            
        #print(data)

     ######## Send data  ##########
        if(data):
            data = str.encode(json.dumps(data))
            ##dataSocket.sendto(str.encode(str(data)),serverAddressPort)
            dataSocket.sendto(data , serverAddressPort)
            ##### ** Hint ** #####
            ##print(str.encode(str(data)))
            #print(data[0][1])
        mpDraw.draw_landmarks(img, results.pose_landmarks,mpPose.POSE_CONNECTIONS)
        
    else:
        continue

################### FPS ###################
    fps , img = fpsReader.update(img , pos=(30,35), color= (0,255,255), scale=1.8 , thickness=3)

############ Result ###########
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()