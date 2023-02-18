import cv2
import numpy as np
import time
import os 
import gpiozero as gz

my_device = gz.DigitalOutputDevice(17)

camera_id = 0 # 0: built-in camera
cap = cv2.VideoCapture(camera_id)
if not cap.isOpened():
    raise ValueError("Camera can't open !")

folderProj = os.path.dirname(__file__)

path_haar_cascade = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(path_haar_cascade)

def areaRatioFace(frame, face_single):
    if face_single is None:
        return None
    
    frame_area = frame.shape[0] * frame.shape[1]

    #                w               h
    face_single = face_single.flatten()
    face_area = face_single[2] * face_single[3]

    area_ratio = face_area/frame_area

    return area_ratio

def getClosestFace(frame, face_bounding_box):

    area_ratio_max = 0
    n_faces = face_bounding_box.shape[0]

    faces_max = np.zeros((1,4), dtype=int)
    # for (x, y, w, h) in faces:s
    for i in range(n_faces):
        area_ratio = areaRatioFace(frame, face_bounding_box[i,:])
        
        if area_ratio > area_ratio_max:
            area_ratio_max = area_ratio
            index_max = i

    faces_max[0,:] = face_bounding_box[index_max,:]

    return faces_max

def drawBoundingFace(frame, face_bounding_box, find_area_ratio=False, closest_face=False):

    if not isinstance(face_bounding_box, np.ndarray):
        return frame

    if closest_face:
        face_bounding_box = getClosestFace(frame, face_bounding_box)

    n_faces = face_bounding_box.shape[0]

    frame_draw = frame.copy()

    for i in range(n_faces):
        x = face_bounding_box[i, 0]
        y = face_bounding_box[i, 1]
        w = face_bounding_box[i, 2]
        h = face_bounding_box[i, 3]

        # draw face 
        point_start = (x, y)
        point_end = (x + w, y + h)
        line_rgb = (255, 0, 0)
        line_thickness = 4
        frame_draw = cv2.rectangle(frame_draw, point_start, point_end, line_rgb, line_thickness)

        # if find_area_ratio:
        #     area_ratio = areaRatioFace(frame, face_bounding_box[i])

        #     # draw text
        #     font = cv2.FONT_HERSHEY_SIMPLEX
        #     point_text_start = (x, y - 15)
        #     fontScale = 1
        #     text_rgb = (0, 0, 255)
        #     text_thickness = 2
        #     area_ratio = np.round(area_ratio, 3)
        #     text = 'area_ratio:' + str(area_ratio)
        #     frame_draw = cv2.putText(frame_draw, text, point_text_start, font,
        #                         fontScale, text_rgb, text_thickness, cv2.LINE_AA)

        # cv2.imshow('frame_with_face', frame_draw)

    return frame_draw

def findAndDrawAreaRatio(frame, face_bounding_box):
    if isinstance(face_bounding_box, np.ndarray):
        area_ratio = areaRatioFace(frame, face_bounding_box)

        # draw text
        face_bounding_box = face_bounding_box.flatten()
        x = face_bounding_box[0]
        y = face_bounding_box[1]
        w = face_bounding_box[2]
        h = face_bounding_box[3]

        font = cv2.FONT_HERSHEY_SIMPLEX
        point_text_start = (x, y - 15)
        fontScale = 1
        text_rgb = (0, 0, 255)
        text_thickness = 2
        area_ratio = np.round(area_ratio, 3)
        text = 'area_ratio:' + str(area_ratio)
        frame_draw = frame.copy()
        frame_draw = cv2.putText(frame_draw, text, point_text_start, font,
                            fontScale, text_rgb, text_thickness, cv2.LINE_AA)
    
        return frame_draw, area_ratio

    else:

        return None

def detectFace(haar_cascade, frame):
    # convert to gray scale 
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    #-- running detect faces
    face_bounding_box = haar_cascade.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=9)

    return face_bounding_box

def writeImgFace(frame, face_bounding_box):

    if not isinstance(face_bounding_box, np.ndarray):
        return

    n_faces = face_bounding_box.shape[0]

    for i in range(n_faces):
        x = face_bounding_box[i, 0]
        y = face_bounding_box[i, 1]
        w = face_bounding_box[i, 2]
        h = face_bounding_box[i, 3]

        faceROI = frame[y:y + h, x:x + w, :]

        name_img_face = f'face_{i}.jpg'
        cv2.imwrite(name_img_face, faceROI)

countFrameOk = 0
countFaceNotFound = 0
countCap = 0
while True:
    print(f'capturing {countCap}...')
    ret, frame = cap.read()

    # if frame is read correctly()
    if not ret:
        print("frame is not read correctly !, stop stream")
        break

    # detect position of face on frame
    face_box = detectFace(face_cascade, frame)

    if isinstance(face_box, np.ndarray):
        # found face 

        # draw line around detected face
        frame_draw = drawBoundingFace(frame, face_box, find_area_ratio=True, closest_face=True)

        face_box_closest = getClosestFace(frame, face_box)

        frame_draw, area_ratio = findAndDrawAreaRatio(frame_draw, face_box_closest)

        # turn on/off device
        if area_ratio > 0.2:
            # not too far
            countFaceNotFound = 0

            if countFrameOk < 6:
                countFrameOk = countFrameOk + 1

            else:
                # turn on device
                my_device.on()

        else:
            countFrameOk = 0

            if countFaceNotFound < 6:
                countFaceNotFound = countFaceNotFound + 1
            
            else:
                # turn off device
                my_device.off()

        # type 'q' to exit program
        # if cv2.waitKey(1) == ord('q'):
            # break

        print('writing...')
        img_name = 'frame_face_current.jpg'
        pathSave = os.path.join(folderProj, img_name)
        cv2.imwrite(pathSave, frame_draw)
    
    else:
        print('writing...')
        img_name = 'frame_face_current.jpg'
        pathSave = os.path.join(folderProj, img_name)
        cv2.imwrite(pathSave, frame)

    # writeImgFace(frame, face_bounding_box)
    if countCap > 255:
        countCap = 0
    
    else:
        countCap = countCap + 1
        
    time.sleep(0.5)
