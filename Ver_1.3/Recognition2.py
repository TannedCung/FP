# all funtions in this module take input as a single frame or single face poistion
from keras import models
from keras import layers
from keras.layers import Input
from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras.models import load_model
from tensorflow.keras.models import load_model as lm
import numpy as np
from keras_vggface import utils
import scipy as sp
import cv2
import os
import glob
import pickle
import Student
import dlib
from imutils import face_utils
from keras.initializers import glorot_uniform
from keras.utils import CustomObjectScope
# import compute_feature
import tensorflow as tf
from PIL import ImageFont, ImageDraw, Image

FACE_IMAGES_FOLDER = "./data/face_images"


def save_pikle(address, pickleFile):
    file_to_save = open(address, "wb")
    pickle.dump(pickleFile, file_to_save)
    file_to_save.close()

def load_pickle(address):
    if not os.path.exists(address):
        save_pikle(address, {})
    file_to_load = open(address, "rb")
    pickleFile = pickle.load(file_to_load)
    file_to_load.close()
    return pickleFile

class New_model():

    def __init__(self, path, flag=0):
       self.model = self.loadmodel(path) if flag==0 else self.load_model_2(path)
       self.graph = tf.get_default_graph()

    @staticmethod
    def loadmodel(path):
        return load_model(path)
    @staticmethod
    def load_model_2(path):
        with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
            return lm(path)


    def predict(self, X):
        with self.graph.as_default():
            return self.model.predict(X)

class FaceIdentify():

    Cascade_path = "./pretrained_models/haarcascade_frontalface_alt.xml"
    eye_path = "./pretrained_models/haarcascade_eye.xml"
    feature_model_path = "./models/feature_model.h5"
    model_path = "./models/model.h5"
    landmark_path = "./pretrained_models/shape_predictor_68_face_landmarks.dat"

    def __init__(self, recog_model_path='./models/model.h5'):
        self.face_size = 224
        print("Loading model...")
        self.ft_graph = tf.Graph()
        self.recog_graph = tf.Graph()
        self.landmark_detect = dlib.shape_predictor(self.landmark_path)

        
        self.feature_model = New_model(self.feature_model_path)
        self.recog_model = New_model(self.model_path)
        self.mask_model = New_model('./models/mask_model.h5', flag=1)
        self.recog_model.model.summary()
        print("Loading model done")

    def draw_label(self, frame, point, score, flag, label=None, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5,
                    thickness=1):
        Stds = load_pickle("./data/pickle/Students.pickle")
        fontpath = "./data/Font/Asap-Regular.otf"
        Font = ImageFont.truetype(fontpath, 15, encoding="unic")
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        
        red = (10, 20, 255)
        green = (30, 255, 10)
        black = (0, 0, 0)
        yellow = (10, 255, 225)
        (x,y,w,h)= point
        if flag <= 10:
            label = "Detecting {}%". format(flag*10-1)
            size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            cv2.rectangle(frame, (x, y), (x + w, y + size[1]+5), yellow, cv2.FILLED)
            cv2.rectangle(frame, (x,y), (x+w, y+h), yellow)

            img_pil = Image.fromarray(frame)
            draw = ImageDraw.Draw(img_pil)

            draw.text((x,y-3), label, font=Font, fill=black)
            return np.array(img_pil)
        elif flag == 11:
            label = "Stranger"
            size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            acc_size = cv2.getTextSize(" xx.xx%", font, font_scale, thickness)[0]
            cv2.rectangle(frame, (x, y), (x+size[0], y+size[1]), red, cv2.FILLED)
            cv2.rectangle(frame, (x, y+h-acc_size[1]), (x+acc_size[0], y+h), red, cv2.FILLED)
            cv2.rectangle(frame, (x,y), (x+w, y+h), red)

            img_pil = Image.fromarray(frame)
            draw = ImageDraw.Draw(img_pil)

            draw.text((x,y-4), label , font=Font, fill=black)
            draw.text((x,y+h-acc_size[1]-3), " {:.2f}%".format(score), font=Font, fill=black)
            return np.array(img_pil)
        elif flag >= 12: # an acquaintance
            infor = next(item for item in Stds if item["name"] == label)
            cv2.rectangle(frame, (x,y), (x+w, y+h), green)
            size = cv2.getTextSize(infor.get('name'), font, font_scale, thickness)[0]
            acc_size = cv2.getTextSize("xx.xx%", font, font_scale, thickness)[0]
            cv2.rectangle(frame, (x,y), (x+w, y+size[1]+5), green, cv2.FILLED)
            cv2.rectangle(frame, (x, y+h-acc_size[1]), (x+acc_size[0],y+h), green, cv2.FILLED)

            img_pil = Image.fromarray(frame)
            draw = ImageDraw.Draw(img_pil)

            draw.text((x,y-2), infor.get('name') , font=Font, fill=black)
            draw.text((x, y+h-acc_size[1]-3), "{:.2f}%".format(score) , font=Font, fill=black)
            return np.array(img_pil)
        
    def draw_mask_stt(self, frame, point, state, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5,
                    thickness=1):
        red = (10, 20, 255)
        green = (30, 255, 10)
        black = (0, 0, 0)
        x,y,w,h = point
        x=int(x)
        y=int(y)
        w=int(w)
        h=int(h)
        if state == 1:
            label = 'Mask: On'
            size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            cv2.rectangle(frame, (x+w-size[0], y+h-size[1]), (x+w, y+h), green, cv2.FILLED)
            cv2.putText(frame, label, (x+w-size[0], y+h), font, font_scale, black, thickness)
        else:
            label = 'Mask: Off'
            size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            cv2.rectangle(frame, (x+w-size[0], y+h-size[1]), (x+w, y+h), red, cv2.FILLED)
            cv2.putText(frame, label, (x+w-size[0], y+h), font, font_scale, black, thickness)
    
    def is_mask_on_2(self, frame, face_area):
        x, y, w, h = face_area
        frame = frame[y:y+h, x:x+w]
        frame = cv2.resize(frame, (150,150))
        frame = np.expand_dims(frame, axis=0)
        score = self.mask_model.predict(frame)
        return np.argmax(score)

    def is_mask_on(self,frame, face_area):
        x, y, w, h = face_area
        face = frame[y:y+h, x:x+w]
        if np.array(face).size == 0:
            return 0
        else:
            try:
                gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                rect = dlib.rectangle(0, 0, w, h)
                landmark = self.landmark_detect(gray, rect)
                landmark = face_utils.shape_to_np(landmark)
                # Capture mouth area        		
                (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
                mouth = landmark[mStart:mEnd]
                boundRect = cv2.boundingRect(mouth)
                # Calculate hsv of mouth area
                hsv = cv2.cvtColor(face[int(boundRect[1]):int(boundRect[1] + boundRect[3]), 
                                        int(boundRect[0]):int(boundRect[0] + boundRect[2])], cv2.COLOR_RGB2HSV) 
                area = int(boundRect[2])*int(boundRect[3])

                boundaries = [
                ([0, 0, 0], [360, 255, 25]), # black
                ([0, 0, 166], [360, 39, 255]),  # white
                ([0, 0, 25], [360, 39, 166]), # gray
                ([150, 39, 25], [180, 255, 255]), # bule-green # with cyan included
                ([180, 39, 25], [255, 255, 255]) # blue # also inculded navy
                ]

                for (lower, upper) in boundaries:
                    lower = np.array(lower)
                    upper = np.array(upper)

                    mask = cv2.inRange(hsv, lower, upper)
                    # print(np.sum(mask)/area)
                    if np.sum(mask)/area > 50:
                        return 1
            except:
                return 0

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

    def identify_face(self, face, threshold=0.1):
        list_name = list(os.listdir(FACE_IMAGES_FOLDER))
        list_name.sort()
        # input of feature_model should be of shape (None, 224, 224, 3)
        # face = np.expand_dims(face, axis=0)
        # Extract features from the face

        feature = self.feature_model.predict(face)
        # Expand the dim of the feature as the input of recog_model need shape (None, 2048)
        # feature = np.expand_dims(feature, axis=0)
        # Predict
        scores = self.recog_model.predict(feature)
        name = list_name[np.argmax(scores)]
        score = np.max(scores)*100
        if score < threshold:
            name = "Unknown"
        return name, score 

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

    def predict_face_from_eyes(self, last_face, frame, margin=10):

        eye_detect = cv2.CascadeClassifier(self.eye_path)
        face = cv2.cvtColor(frame[x-margin:x+w+margin, y-margin:y+h+margin], cv2.COLOR_BGR2GRAY)
        center = (0,0)
        eyes = eye_detect.detectMultiScale(face, scaleFactor=1.1, minNeighbors=30, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)
        if len(eyes)>2 :
            matrx = np.zeros(len(eyes), len(eyes))
            for i in len(eyes):
                for j in len(eyes):
                    if i==j:
                        matrx[i][j] = 1000
                    else:
                        matrx[i][j] = (eyes[i][1] - eyes[j][1]) if (eyes[i][1] - eyes[j][1]) >=0 else 1000

            y1 = np.argmin(matrx.min(axis=1))
            y2 = np.argmin(matrx.min(axis=0))  
            Cx = (eyes[y1][0] + eyes[y2][0])/2
            Cy = (eyes[y1][1] + eyes[y2][1])/2
            center = (Cx, Cy)
        elif len(eyes)==2 :
            Cx = (eyes[1][0] + eyes[0][0])/2
            Cy = (eyes[1][1] + eyes[0][1])/2
            center = (Cx, Cy)
        else: 
            return None
        a = sqrt(center[0]**2 + center[1]**2)
        x = int(center[0] - 1.285*a)
        y = int(center[1] -1.05*a)
        w = int(2.67*a)
        h = int(2.67*a)
        return (x,y,w,h)
    
    def reload_model(self):
        self.recog_model = New_model(self.model_path)
        print("loading new model done!")
            
'''
def main():
    cap = cv2.VideoCapture(0)
    face_recog = FaceIdentify()
    while 1:
        _, frame = cap.read()
        faces, points = face_recog.detect_face(frame)
        for i, face in enumerate(faces):
            name, score = face_recog.identify_face(face)
            print("name:{}  score:{}".format(name, score))
        cv2.imshow("face", frame)
        if cv2.waitKey(5) == 27:
            break

if __name__ == "__main__":
    main()
'''