from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
import time
import dlib
import cv2

#get the location of the eyes
def eye_aspect_ratio(eye):
    # compute the euclidean distances between the vertical landamrks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    eye_opening_ratio = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return eye_opening_ratio


# the consecuting frame factor tells us to consider this amount of farme.
ar_thresh = 0.3
eye_ar_consec_frame = 5
counter = 0
total = 0

# get the frontal face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cap = cv2.VideoCapture(0)
while(True):
    _,frame = cap.read()
    frame = cv2.flip(frame,1)
    frame = imutils.resize(frame, width=500,height=500)
        
    (lBegin, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rBegin, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    #preprocessing the image
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(gray)
    detections = detector(clahe_image,0)
    for detection in detections:
        shape = predictor(gray,detection)
        shape = face_utils.shape_to_np(shape)
        left_eye = shape[lBegin:lEnd]
        right_eye = shape[rBegin:rEnd]

        leftEyeHull = cv2.convexHull(left_eye)
        rightEyeHull= cv2.convexHull(right_eye)
        cv2.drawContours(clahe_image, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(clahe_image, [rightEyeHull], -1, (0, 255, 0), 1)
        #calculating the EAR
        left_eye_Ear = eye_aspect_ratio(left_eye)
        right_eye_Ear = eye_aspect_ratio(right_eye)

        avg_Ear = (left_eye_Ear + right_eye_Ear)/2.0

        if avg_Ear<ar_thresh:
            counter+=1
        else:
            if counter>eye_ar_consec_frame:
                total+= 1
            counter = 0	
        cv2.putText(clahe_image, "Blinks: {}".format(total), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(clahe_image, "EAR: {:.2f}".format(avg_Ear), (300, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Frame", clahe_image)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()



