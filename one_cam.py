import argparse
import random
import time
from pythonosc import osc_message_builder
from pythonosc import udp_client

import numpy as np
import cv2
import socket

#make parameter for camera num
#make install script

face_cascade = cv2.CascadeClassifier('haar_libs/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haar_libs/haarcascade_eye.xml')


if __name__ == "__main__":
    #cam parameter
    cam_num = 0
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", help="cam to use")
    args = parser.parse_args()

    if args.cam:
        cam_num = args.cam

    cap = cv2.VideoCapture(cam_num)

    while 1:
        #setup osc
        parser = argparse.ArgumentParser()
        parser.add_argument("--ip", default="192.168.0.5", help="The ip of the OSC server")
        parser.add_argument("--port", type=int, default=7003, help="The port the OSC server is listening on")
        args = parser.parse_args()
        client = udp_client.SimpleUDPClient(args.ip, args.port)


        ret, img00 = cap.read()
        img = cv2.resize(img00,(560,340))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        num_faces = len(faces)
        client.send_message("/people", num_faces)
        # print(len(faces))

        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


        # cv2.imshow('img',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break


cap.release()
cv2.destroyAllWindows()