import cv2
import os
import glob
import pickle
import numpy as np
import scipy as sp
from utlis import *
from keras_vggface.vggface import VGGFace
from keras.preprocessing import image
from keras_vggface import utils
from keras.utils import np_utils, Sequence
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.models import Model
from keras.models import load_model
from keras.optimizers import SGD
import Recognition2
import tensorflow as tf
from imutils import paths
from sklearn.utils import shuffle
from math import hypot, cos, sin, pi

FACE_IMAGES_FOLDER = "./data/face_images"
FEATURE_FOLDER = "./data/pickle/features"
PICKLE_FOLDER = "./data/pickle"

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

def load(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = utils.preprocess_input(img)
    img = np.squeeze(img, axis=0)
    return img

class make_color():
    def __init__(self, frame, center=(320, 240), radius=140 ):
        self.frame = np.empty((center[1]*2, center[0]*2, 3), np.uint8)
        self.mask = np.empty((center[1]*2, center[0]*2, 3), np.uint8)*0
        self.x, self.y = center
        self.radius = radius

    def circle_mask(self):
        rows, cols, _ = self.frame.shape

        for i in range(cols):
            for j in range(rows):
                if hypot(i-self.x, j-self.y) > self.radius:
                    self.frame[j,i] = 0.5
                else: 
                    self.frame[j,i] = 1
        return self.frame

    def draw_percent(self, percent, l=20):
        cv2.circle(self.mask, (self.x, self.y), self.radius, (225,10,225), 3)
        alpha = 2*pi*percent
        # first point
        x1 = int(self.x + (self.radius+15)*sin(alpha))
        y1 = int(self.y - (self.radius+15)*cos(alpha))
        # second point
        x2 = int(x1 + l*sin(alpha))
        y2 = int(y1 - l*cos(alpha))

        cv2.line(self.mask, (x1,y1), (x2,y2), (10,225,10), 2)
        return self.mask

class MY_Generator(Sequence):

    def __init__(self, image_filenames, labels, batch_size):
        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size

    def __len__(self):
        return int(len(self.image_filenames) / float(self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([load(file_name)
               for file_name in batch_x]), np.array(batch_y)

class FaceExtractor():

    Cascade_path = "./pretrained_models/haarcascade_frontalface_alt.xml"
    feature_model_path = "./models/feature_model.h5"
    model_path = "./models/model.h5"

    def __init__(self, face_size=224):
        self.face_size = face_size

        self.ft_graph = tf.Graph()
        self.recog_graph = tf.Graph()
        with self.ft_graph.as_default():
            self.ft_session = tf.Session()
            with self.ft_session.as_default():
                self.feature_model = load_model(self.feature_model_path)
        with self.recog_graph.as_default():
            self.recog_session = tf.Session()
            with self.recog_session.as_default():
                self.recog_model = load_model(self.model_path)

        for layer in self.feature_model.layers[:165]:
            layer.trainable = False
        print("Loading done")

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
        """
        need: capture, name
        return: nothing, but save 100 images of that person in directory /save_folder/name
        """ 
        face_cascade = cv2.CascadeClassifier(self.Cascade_path)
        save_folder = os.path.join(save_folder, name)
        os.makedirs(save_folder)
        # 0 means the default video capture device in OS
        # cap = cv2.VideoCapture(0)
        cap = cap
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)
        
        # infinite loop, break by key ESC
        frame_counter = 0
        face_counter = 0
        max_counter = 50

        while(cap.isOpened()):
            ret, frame = cap.read()
            frame = np.fliplr(frame)
            if ret:
                # make the mask at the first frame
                if frame_counter == 0:
                    colorist = make_color(frame=frame)
                    mask1 = colorist.circle_mask()
                    mask2 = colorist.draw_percent(percent=0)
                
                frame_counter = frame_counter + 1
                fra = frame[int((height/2)-colorist.radius):int((height/2)+colorist.radius), int((width/2)-colorist.radius):int((width/2)+colorist.radius)]
                gray = cv2.cvtColor(fra, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.2,
                    minNeighbors=10,
                    minSize=(64, 64),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                # only keep the biggest face as the main subject
                face = None
                if len(faces) > 1:  # Get the largest face as main face
                    face = max(faces, key=lambda rectangle: (rectangle[2] * rectangle[3]))  # area = w * h
                elif len(faces) == 1:
                    face = faces[0]
                if face is not None:
                    face_img, cropped = self.crop_face(fra, face, margin=20, size=self.face_size)
                    (x, y, w, h) = cropped
                    if frame_counter % 3 == 0 :
                        imgfile = name +"_"+ str(face_counter) + ".png"
                        imgfile = os.path.join(save_folder, imgfile)
                        cv2.imwrite(imgfile, face_img)
                        face_counter += 1
                        mask2 = colorist.draw_percent(percent=(face_counter/max_counter))

                check_img = cv2.multiply(frame, mask1)
                check_back = np.asarray(cv2.multiply(frame,(1-mask1))*0.2, np.uint8)
                check_img = cv2.add(check_img, check_back)
                cv2.imshow('Faces', cv2.add(check_img, mask2))
            if cv2.waitKey(5) == 27 or face_counter >= max_counter:   # ESC key press
                break
        cv2.destroyAllWindows()
        agu_brightness(save_folder)

    def compute_features(self, name, image_folder= FACE_IMAGES_FOLDER):     
        """
        need: a folder of the person with the namen given
        return: nothing, but save feature to features.pickle in format name.pickle
        """ 
        print('Doing a comptation of features')
        images, names = list_image(image_folder, name)
        with self.ft_graph.as_default():
            with self.ft_session.as_default():
                features = self.feature_model.predict(images)

        save_pikle("./data/pickle/features/{}.pickle".format(name), features)
    
    def biuld_new_model(self):
        print ("Building new recog model")

        features = []
        labels = []
        classes = []
        features_paths = glob.iglob(FEATURE_FOLDER+"/*.pickle")
        for path in features_paths:
            file = load_pickle(path)
            if len(features)== 0:
                features = file
            else:
                features = np.concatenate((features, file), axis=0)
            name = os.path.basename(path)[:-7]
            classes.append(name)
            for i in range(len(file)):
                labels.append(name)

        le = LabelEncoder()
        labels = le.fit_transform(labels)
        lb = LabelBinarizer()
        labels = lb.fit_transform(labels)

        shuffle(features, labels)

        trainX, validX, trainY, validY = train_test_split(features,
                                                                labels, test_size=0.2, random_state=24)

        self.model = Sequential()

        self.model.add(Dense(len(classes),input_dim=2048, activation='softmax'))
        opt = SGD(lr=0.01)
        batch_size = 256 
        epochs=2
        self.model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
        print ('Start fitting model')
        check_point = ModelCheckpoint('./models/model.h5', monitor='val_loss', save_best_only=True)
        self.model.summary()
        
        H = self.model.fit(x=trainX, y=trainY, epochs=epochs, verbose=1, callbacks=[check_point], validation_data=(validX, validY),
                            steps_per_epoch=len(trainX//batch_size), validation_steps=len(validX//batch_size))
        print("Build new model successful")
'''  
com = FaceExtractor()
names = os.listdir(FACE_IMAGES_FOLDER)

for name in names:
    com.compute_features(name)

com.biuld_new_model()

'''


