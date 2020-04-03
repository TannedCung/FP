from keras.engine import Model
from keras import models
from keras import layers
from keras.layers import Input
from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras.models import load_model
import numpy as np
from keras_vggface import utils
import scipy as sp
import cv2
import os
import glob
import pickle
import Student
import compute_feature

FACE_IMAGES_FOLDER = "./data/face_images"


def save_pikle(address, pickleFile):
    file_to_save = open(address, "wb")
    pickle.dump(pickleFile, file_to_save)
    file_to_save.close()

def load_pickle(address):
    if not os.path.exists(address):
        save_pikle(address, {"name": '???', "id": '???'})
    file_to_load = open(address, "rb")
    pickleFile = pickle.load(file_to_load)
    file_to_load.close()
    return pickleFile

class FaceIdentify():

    Cascade_path = "./pretrained_models/haarcascade_frontalface_alt.xml"

    def __init__(self, recog_model_path='./models/model.h5'):
        self.face_size = 224
        print("Loading VGG Face model...")
        self.model = VGGFace(model='resnet50',
                             include_top=False,
                             input_shape=(224, 224, 3),
                             pooling='avg')  # pooling: None, avg or max
        print("Loading VGG Face model done /n Loading recog model")
        self.recog_model = load_model(recog_model_path)
        print("Loading model done")

    @classmethod
    def draw_label(cls, frame, points, labels, scores, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5,
                    thickness=1):
        Stds = load_pickle("./data/pickle/Students.pickle")
        red = (10, 20, 255)
        green = (30, 255, 10)
        black = (0, 0, 0)
        for i,point in enumerate(points):
            (x,y,w,h)= point
            label = "{}".format(labels[i])
            score = "{:.3f}%".format(scores[i]*100)
            if label == "Unknown":
                size = cv2.getTextSize(label, font, font_scale, thickness)[0]
                cv2.rectangle(frame, (x, y - size[1]), (x + size[0], y), red, cv2.FILLED)
                cv2.rectangle(frame, (x,y), (x+w, y+h), red)
                cv2.putText(frame, label, (x, y), font, font_scale, black, thickness)
            else:
                infor = next(item for item in Stds if item["name"] == label)
                size = cv2.getTextSize(infor.get('name') + "   " + score, font, font_scale, thickness)[0]
                cv2.rectangle(frame, (x,y), (x+w, y+h), green)
                cv2.rectangle(frame, (x, y - size[1]), (x + size[0], y+5), green, cv2.FILLED)
                cv2.putText(frame, infor.get('name') + "   " + score, (x, y), font, font_scale, black, thickness)
                cv2.rectangle(frame, (x, y+5), (x + size[0], y + size[1]+10), green, cv2.FILLED)
                cv2.putText(frame, "ID: {}".format(infor.get("ID")), (x, y + size[1]+5), font, font_scale, black, thickness)
                cv2.rectangle(frame, (x, y + size[1]+10), (x + size[0], y + 2*size[1]+15 ), green, cv2.FILLED)
                cv2.putText(frame, "school_year: {}".format(infor.get("school_year")), (x, y + 2*size[1]+10), font, font_scale, black, thickness)

    def crop_face(self, imgarray, section, margin=20, size=224):
        """
        :param imgarray: full image
        :param section: face detected area (x, y, w, h)
        :param margin: add some margin to the face detected area to include a full head
        :param size: the result image resolution with be (size x size)
        :return: resized image in numpy array with shape (size x size x 3)
        """
        img_h, img_w, _ = imgarray.shape
        if section is None:
            section = [0, 0, img_w, img_h]
        (x, y, w, h) = section
        margin = int(min(w, h) * margin / 100)
        x_a = x - margin
        y_a = y - margin
        x_b = x + w + margin
        y_b = y + h + margin
        if x_a < 0:
            x_b = min(x_b - x_a, img_w - 1)
            x_a = 0
        if y_a < 0:
            y_b = min(y_b - y_a, img_h - 1)
            y_a = 0
        if x_b > img_w:
            x_a = max(x_a - (x_b - img_w), 0)
            x_b = img_w
        if y_b > img_h:
            y_a = max(y_a - (y_b - img_h), 0)
            y_b = img_h
        cropped = imgarray[y_a: y_b, x_a: x_b]
        resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
        resized_img = np.array(resized_img)
        return resized_img, (x_a, y_a, x_b - x_a, y_b - y_a)

    def identify_face(self, faces, threshold=0.95):
        list_name = list(os.listdir(FACE_IMAGES_FOLDER))
        list_name.sort()
        list_person_name = []
        list_score = []
        # self.recog_model.summary()
        features = self.model.predict(faces)
        for feature in features:
            feature = np.expand_dims(feature, axis=0)
            scores = self.recog_model.predict(feature)
            person_score = np.max(scores)
            list_score.append(person_score)
            id = np.argmax(scores)
            # print(id)
            person_name = list_name[np.argmax(scores)]
            # print(person_score*100, person_name)

            if person_score >= threshold:
                list_person_name.append(person_name)
            else:
                list_person_name.append("Unknown")
        return list_person_name, np.array(list_score) 

    def detect_face(self, frame):
        """
        need: 1 frame of the stream
        return: a list of cropped images to identify + its (x,y,w,h)
        """
        face_cascade = cv2.CascadeClassifier(self.Cascade_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
                                            gray,
                                            scaleFactor=1.2,
                                            minNeighbors=5,
                                            minSize=(32, 32)
                                            )
        face_imgs = np.empty((len(faces), self.face_size, self.face_size, 3))   
        points = []     
        for i, face in enumerate(faces):
            face_img, cropped = self.crop_face(frame, face, margin=10, size=self.face_size)
            face_imgs[i, :, :, :] = face_img
            points.append(cropped)
        return face_imgs, points
    
        
'''
def main():
    face = FaceIdentify()
    face.detect_face()

if __name__ == "__main__":
    main()
'''