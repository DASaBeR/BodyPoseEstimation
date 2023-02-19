import cvzone
import cv2
import mediapipe as mp
import socket

mpPose = mp.solutions.pose
pose = mpPose.Pose(static_image_mode=False,
                                     smooth_landmarks=True,
                                     min_detection_confidence=0.5,
                                     min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils
fpsReader = cvzone.FPS()

cap = cv2.VideoCapture(0)

############ Communication ##########
dataSocket=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
serverAddressPort=("127.0.0.1",5052)

while True:
    success, img = cap.read()

###### Flipping the image to fix left and right move like a mirror #####
    img = cv2.flip(img,1)  

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    data=[]
    if(results.pose_landmarks):
        land_mark = results.pose_landmarks.landmark
        res_dict = list(land_mark)
        x = res_dict[0].x
        y = res_dict[0].y
        z = res_dict[0].z
      
        data.extend([x , y , z])
     ######## Send data  ##########
        if(data):
            dataSocket.sendto(str.encode(str(data)),serverAddressPort)

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