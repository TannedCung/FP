import compute_feature
import Student
import Recognition2 as Recognition
import cv2
import numpy as np
import dlib


# Problem for now: the tracker doesn't track 
def main():
    # 0 means the default video capture device in OS
    frame_counter = 0
    face_number = 0

    faceTracker = {}
    faceCurrentPos = {}
    faceStatus = {} # 0,1,2,3,4 = "Detecting"
                    # 5 = "Known"
                    # 6 = "Unknown"
    label = {}  # contains name
    score = {}
    print(type(score))
    video_capture = cv2.VideoCapture(0)
    # infinite loop, break by key ESC
    the_bodyguard = Recognition.FaceIdentify()
    while True:
        result = 0
        if not video_capture.isOpened():
            sleep(5)
        _, frame = video_capture.read()
        frame_counter += 1
        if not (frame_counter % 10):
            face_imgs, points = the_bodyguard.detect_face(frame)
		# thru detected faces
            for (_x, _y, _w, _h) in points:
                x = int(_x)
                y = int(_y)
                w = int(_w)
                h = int(_h)

                # Tinh tam diem cua face
                x_center = x + 0.5 * w
                y_center = y + 0.5 * h  

                matchFaceID = None
                # thru tracked faces, check if this i-th face were the new face
                for faceID in faceTracker.keys():
                    trackedPosition = faceTracker[faceID].get_position()

                    t_x = int(trackedPosition.left())
                    t_y = int(trackedPosition.top())
                    t_w = int(trackedPosition.width())
                    t_h = int(trackedPosition.height())   

                    t_x_center = t_x + 0.5 * t_w
                    t_y_center = t_y + 0.5 * t_h
                    # if the center is out of the rects of existing faces, then it is no longer an existing face   
                    if (t_x <= x_center <= (t_x + t_w)) and (t_y <= y_center <= (t_y + t_h)) and (x <= t_x_center <= (x + w)) and (y <= t_y_center <= (y + h)):
                        matchFaceID = faceID

                if matchFaceID is None:
                    tracker = dlib.correlation_tracker()
                    tracker.start_track(frame, dlib.rectangle(x, y, x+w, y+h))

                    faceTracker[face_number] = tracker
                    faceStatus[face_number] = 0
                    face_number += 1

        for faceID in faceTracker.keys():
            trackedPosition = faceTracker[faceID].get_position()

            t_x = int(trackedPosition.left())
            t_y = int(trackedPosition.top())
            t_w = int(trackedPosition.width())
            t_h = int(trackedPosition.height())

            faceCurrentPos[faceID] = (t_x, t_y, t_w, t_h)
            cv2.rectangle(frame, (t_x,t_y), (t_x+t_w, t_y+t_h), (225,225,0))
            # if the face state (status) is Unknown, draw red rect,
            # no more recognition
            if faceStatus[faceID] == 6:
                the_bodyguard.draw_label(frame=frame, point=faceCurrentPos[faceID],
                                        score=0, flag=6)
            # if the face state is Known, draw that person information
            elif faceStatus[faceID] == 5:
                the_bodyguard.draw_label(frame=frame, point=faceCurrentPos[faceID],
                                        score=score[faceID], flag=5, label=label[faceID])
            # if this is the 1st time detected, detect it
            elif faceStatus[faceID] == 0:
                face, pos = the_bodyguard.crop_face(frame, faceCurrentPos[faceID])
                face = np.expand_dims(face, axis=0)
                name, sc = the_bodyguard.identify_face(face)
                label[faceID] = name
                score[faceID] = sc
                the_bodyguard.draw_label(frame=frame, point=faceCurrentPos[faceID],
                                        score=score[faceID], flag=0 )
                faceStatus[faceID] += 1
            # in 5 times of recognize that face, if at any time, the label change
            # that person should be a stranger
            elif faceStatus[faceID] < 5 and faceStatus[faceID] > 0:
                if not (frame_counter % 2):
                    face, pos = the_bodyguard.crop_face(frame, faceCurrentPos[faceID])
                    face = np.expand_dims(face, axis=0)
                    name, sc = the_bodyguard.identify_face(face)
                    if label[faceID] == name:
                        score[faceID] = (score[faceID]*(faceStatus[faceID]-1) + sc)/faceStatus[faceID] 
                        the_bodyguard.draw_label(frame=frame, point=faceCurrentPos[faceID],
                                            score=score[faceID], flag=faceStatus[faceID] )
                        faceStatus[faceID] += 1
                    else:
                        label[faceID] = "Unknown"
                        faceStatus[faceID] = 6
                        the_bodyguard.draw_label(frame=frame, point=faceCurrentPos[faceID],
                                            score=score[faceID], flag=6 )
        cv2.imshow('Hi there !', frame)
        if cv2.waitKey(5) == ord('q'):
            break


if __name__ == "__main__":
    main()
