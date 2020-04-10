import cv2
import os
import glob
import pickle
import numpy as np
import scipy as sp
import utlis as ut
from keras_vggface.vggface import VGGFace
from keras.preprocessing import image
from keras_vggface import utils
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import ModelCheckpoint
import Recognition


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

def list_image(image_path, name):
    """
    return a list_imgs of a person with name given
    """
    list_image = []
    list_name = []

    imagePath = os.path.join(image_path, name)
    imagePath = list(glob.iglob(os.path.join(imagePath, '*.*')))

    for (i, imagePath) in enumerate(imagePath):
        name = imagePath.split(os.path.sep)[-2]
        # print(name)
        img = image.load_img(imagePath, target_size=(224, 224))
        img = image.img_to_array(img)
        
        img = np.expand_dims(img, axis=0)
        img = utils.preprocess_input(img)
        
        list_name.append(name)
        list_image.append(img)
        
    return np.vstack(list_image), np.vstack(list_name)

class FaceExtractor():

    Cascade_path = "./pretrained_models/haarcascade_frontalface_alt.xml"

    def __init__(self, face_size=224):
        self.face_size = face_size
        self.resnet50_features = VGGFace(model='resnet50',
                                include_top=False,
                                input_shape=(224, 224, 3),
                                pooling='avg')  # pooling: None, avg or max


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
        margin = int(min(w,h) * margin / 100)
        x_a = x - margin
        y_a = y - margin
        x_b = x + w + margin
        y_b = y + h + margin
        if x_a < 0:
            x_b = min(x_b - x_a, img_w-1)
            x_a = 0
        if y_a < 0:
            y_b = min(y_b - y_a, img_h-1)
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

    def extract_faces(self, name, cap, save_folder=FACE_IMAGES_FOLDER):
        face_cascade = cv2.CascadeClassifier(self.Cascade_path)
        save_folder = os.path.join(save_folder, name)
        os.makedirs(save_folder)
        # 0 means the default video capture device in OS
        # cap = cv2.VideoCapture(0)
        cap = cap

        '''
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)
        print("length: {}, w x h: {} x {}, fps: {}".format(length, width, height, fps))

        '''
        # infinite loop, break by key ESC
        frame_counter = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret and frame_counter / 5 <= 100:
                frame_counter = frame_counter + 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.2,
                    minNeighbors=10,
                    minSize=(64, 64)
                )
                # only keep the biggest face as the main subject
                face = None
                if len(faces) > 1:  # Get the largest face as main face
                    face = max(faces, key=lambda rectangle: (rectangle[2] * rectangle[3]))  # area = w * h
                elif len(faces) == 1:
                    face = faces[0]
                if face is not None:
                    face_img, cropped = self.crop_face(frame, face, margin=40, size=self.face_size)
                    (x, y, w, h) = cropped
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
                    if frame_counter % 5 == 0 :
                        imgfile = name +"_"+ str(frame_counter / 5) + ".png"
                        imgfile = os.path.join(save_folder, imgfile)
                        cv2.imwrite(imgfile, face_img)
                cv2.imshow('Faces', frame)
            if cv2.waitKey(5) == 27:  # ESC key press
                break
        # cap.release()
        # cv2.destroyAllWindows()
    def compute_features(name, image_folder= FACE_IMAGES_FOLDER,):      
        print('Doing a comptation of features')
        images, names = list_image(image_folder, name)
        features = self.resnet50_features.predict(images)
        # caculate mean fearures and save
        save_folder = os.path.join(FACE_IMAGES_FOLDER, name)
        feature_vector = np.array(features).sum(axis=0)/len(features)

        precompute_features = list(load_pickle("./data/pickle/precompute_features.pickle"))
        precompute_features.append({"name": name, "features": feature_vector})
        save_pikle("./data/pickle/precompute_features.pickle", precompute_features)
    
