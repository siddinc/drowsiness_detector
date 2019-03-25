from scipy.spatial import distance as dist
import playsound


def play_alarm(file_path):
    playsound.playsound(file_path)
    
def calculate_eye_aspect_ratio(eye):
    a = dist.euclidean(eye[1], eye[5])
    b = dist.euclidean(eye[2], eye[4])
    c = dist.euclidean(eye[0], eye[3])
    eye_aspect_ratio = (a + b) / (2.0 + c)
    
    return eye_aspect_ratio