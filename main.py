import compute_feature
import Student
import Recognition
import cv2
import numpy as np


# Not available for now
# Use Recognition.py instead
def main():
    # 0 means the default video capture device in OS
    video_capture = cv2.VideoCapture(0)
    # infinite loop, break by key ESC
    the_bodyguard = Recognition.FaceIdentify()
    while True:
        result = 0
        if not video_capture.isOpened():
            sleep(5)
        _, frame = video_capture.read()
        face_imgs, points = the_bodyguard.detect_face(frame)
        list_names, list_scores = the_bodyguard.identify_face(face_imgs)
        the_bodyguard.draw_label(frame=frame, points=points, labels=list_names,
                                scores=list_scores)
        cv2.imshow('Keras Faces', frame)
        if cv2.waitKey(5) == 27 or result > 0:  # ESC key press
            break

if __name__ == "__main__":
    main()
