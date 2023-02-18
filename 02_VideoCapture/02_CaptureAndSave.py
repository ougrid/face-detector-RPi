import cv2
import numpy as np
import time
import os 

camera_id = 0 # 0: built-in camera
cap = cv2.VideoCapture(camera_id)
if not cap.isOpened():
    raise ValueError("Camera can't open !")

folderProj = os.path.dirname(__file__)
print(folderProj)
countCap = 0
while True:
    print(f'capturing {countCap}...')
    ret, frame = cap.read()

    # if frame is read correctly()
    if not ret:
        print("frame is not read correctly !, stop stream")
        break

    print('writing...')
    img_name = 'frame_current.jpg'
    pathSave = os.path.join(folderProj, img_name)
    cv2.imwrite(pathSave, frame)

    if countCap > 255:
        countCap = 0
    
    else:
        countCap = countCap + 1

    time.sleep(1)
