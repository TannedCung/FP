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
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
import Student
import compute_feature_v2 as cf

FACE_IMAGES_FOLDER = "./data/face_images"

def save_pikle(address, pickleFile):
    file_to_save = open(address, "wb")
    pickle.dump(pickleFile, file_to_save)
    file_to_save.close()

def load_pickle(address):
    file_to_load = open(address, "rb")
    pickleFile = pickle.load(file_to_load)
    file_to_load.close()
    return pickleFile
'''
def compute_features(name, image_folder= FACE_IMAGES_FOLDER,):
    print ("Load VGG model")
    # print (name)
    resnet50_features = VGGFace(model='resnet50',
                            include_top=False,
                            input_shape=(224, 224, 3),
                            pooling='avg')  # pooling: None, avg or max

    print('Doing a computation of features')
    images, names = list_image(image_folder, name)
    features = resnet50_features.predict(images)
        
    # extractor = FaceExtractor()
    
    save_folder = os.path.join(FACE_IMAGES_FOLDER, name)
    os.makedirs(save_folder, exist_ok=True)

    # extractor.extract_faces(name, save_folder)
    # ut.agumentate(save_folder)
    precompute_features = []
    person_name = []
    precompute_features = list(load_pickle("./data/pickle/precompute_features.pickle"))
    person_name = list(load_pickle("./data/pickle/names.pickle"))

    for feature in features:
        precompute_features.append(feature)
    for name in names:
        person_name.append(name)
    save_pikle("./data/pickle/precompute_features.pickle", precompute_features)
    save_pikle("./data/pickle/names.pickle", person_name)

# def biuld_new_model():
 resnet50_features = VGGFace(model='resnet50',
                                include_top=False,
                                input_shape=(224, 224, 3),
                                pooling='avg')  # pooling: None, avg or max


    features, id_name = compute_feature(image_folder=FACE_IMAGES_FOLDER, names= names)

    print ("Building new recog model")
    features = np.array(load_pickle('./data/pickle/precompute_features.pickle'))
    print(features.shape)
    # features = np.expand_dims(features, axis=1)
    id_name = load_pickle('./data/pickle/names.pickle')

    number_of_people = len(list(glob.iglob(os.path.join(FACE_IMAGES_FOLDER, '*'))))
    print ('Start spliting')
    le = LabelEncoder()
    id = le.fit_transform(id_name)
    X_train, X_test, y_train, y_test = train_test_split(features, id, test_size=0.2, random_state=0)
    y_train = np_utils.to_categorical(y_train, number_of_people)
    y_test = np_utils.to_categorical(y_test, number_of_people)
    print (X_train[1].shape)

    print('building model')
    model = Sequential()
    model.add(Dense(number_of_people, activation='softmax'))
    checkpoint = ModelCheckpoint('models/model.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='auto')
    model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
    print ('Start fitting model')
    model.fit(X_train, y_train, epochs=10, verbose=1, validation_split=0.2, callbacks=[checkpoint]) 
    preds = model.evaluate(X_test, y_test)
    print("score ={} ".format(preds))  


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

folders = list(glob.iglob(os.path.join(FACE_IMAGES_FOLDER, '*')))
list_names = [os.path.basename(folder) for folder in folders]
print (list_names)
for name in list_names:
    compute_features(name)
'''

# for the 1st run
names = os.listdir(FACE_IMAGES_FOLDER)

first_compute = cf.FaceExtractor()
for i,name in enumerate(names):
    first_compute.compute_features(name)
    print(i)

barrack = Student.Student()
barrack.save_infor(name="barack", id=20160000, school_year=61)
michelle = Student.Student()
michelle.save_infor(name="michelle", id=20160001, school_year=61)

NPT = Student.Student()
NPT.save_infor(name="Nguyen Phu Trong", id=00000000 )
NXP = Student.Student()
NXP.save_infor(name="Nguyen Xuan Phuc")

files = list(load_pickle("./data/pickle/Students.pickle"))
print(files)
precompute_features = list(load_pickle("./data/pickle/precompute_features.pickle"))
print(precompute_features)
