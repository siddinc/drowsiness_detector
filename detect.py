import dlib
import cv2
import numpy as np
import time
import imutils
from imutils import face_utils
from imutils.video import VideoStream
from utility_functions import calculate_eye_aspect_ratio, play_alarm
from threading import Thread


arguments = {"shape_predictor": "shape_predictor_68_face_landmarks.dat",
             "alarm": "alarm.wav"}

eye_aspect_ratio_threshold = 0.5
min_number_of_consecutive_frames = 30

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(arguments["shape_predictor"])

(l_start, l_end) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(r_start, r_end) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

video_stream = VideoStream(src=0).start()
time.sleep(1)

frame_counter = 0
alarm_on = False

while True:
    frame = video_stream.read()
    frame = cv2.flip(frame, 1)
    
    if frame.any():
        frame = imutils.resize(frame, width=450)
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rectangles = detector(gray_image, 0) #0 means dont subtract mean layers from original image

        for rect in rectangles:
            shape = predictor(gray_image, rect)
            shape = face_utils.shape_to_np(shape)
            
            left_eye = shape[l_start: l_end]
            right_eye = shape[r_start: r_end]
            
            rar = calculate_eye_aspect_ratio(right_eye) #right eye aspect ratio
            lar = calculate_eye_aspect_ratio(left_eye) #left eye aspect ratio
            
            average_eye_aspect_ratio = (rar + lar) / 2.0
            
            left_eye_hull = cv2.convexHull(left_eye)
            right_eye_hull = cv2.convexHull(right_eye)
            
            cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
            
            if average_eye_aspect_ratio < eye_aspect_ratio_threshold:
                frame_counter += 1
                
                if frame_counter >= min_number_of_consecutive_frames:
                    if not alarm_on:
                        alarm_on = True
                        thread = Thread(target=play_alarm, args=(arguments["alarm"],))
                        thread.deamon = True
                        thread.start()
                        
                    cv2.putText(frame, "DROWSINESS_ALERT", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
            else:
                frame_counter = 0
                alarm_on = False
                
            cv2.putText(frame, "EYE ASPECT RATIO: {:.2f}".format(average_eye_aspect_ratio),
                        (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
            
        cv2.imshow("frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    
video_stream.stop()
cv2.destroyAllWindows()