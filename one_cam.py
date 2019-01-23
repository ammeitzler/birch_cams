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
profile_cascade = cv2.CascadeClassifier('haar_libs/haarcascade_profileface.xml')


def main():
    #cam parameter
    cam_num = 0
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", help="cam to use")
    args = parser.parse_args()

    if args.cam:
        cam_num = args.cam

    c = int(cam_num)
    cap = cv2.VideoCapture(c)
    print("detecting faces")

    while 1:
        #setup osc
        ip = "127.0.0.1"
        port = 7003
        client = udp_client.SimpleUDPClient(ip, port)

        ret, img00 = cap.read()
        img = cv2.resize(img00,(560,340))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        prof_face = profile_cascade.detectMultiScale(gray, 1.3, 5)
        numprof_face = len(prof_face)
        for (x,y,w,h) in prof_face:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
        num_eyes = len(eyes)
        for (x,y,w,h) in eyes:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        num_faces = len(faces)
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

        total_num = num_faces + numprof_face + num_eyes
        print(total_num)
        client.send_message("/people", total_num)

        # cv2.imshow('img',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
main()

cap.release()
cv2.destroyAllWindows()