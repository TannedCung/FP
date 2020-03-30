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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import random
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression



'''
# 1.
img = cv2.imread("1.png", 1)
# cv2.imshow('1', img)
# cv2.waitKey(0)
img = np.expand_dims(img, axis=0)
img = utils.preprocess_input(img, version=1)

#a = np.array([[[1]], [[2,3,6,5,99]], [[3,2,5,6,3]], [[3,2,4]]])
#print (a.shape)

img = np.squeeze(img, axis=0)
cv2.imshow('2', img)
cv2.waitKey(0)'''

# 2
'''
img = image.load_img('1.png', target_size=(10, 10))
x = image.img_to_array(img)
print (x[5,:,:])
x = np.expand_dims(x, axis=0)
x = utils.preprocess_input(x, version=1)  # or version=2
x = np.squeeze(x, axis=0)
print (x[5,:,:])
cv2.imshow("x", x)
cv2.waitKey(0)
'''
'''
# 3

FACE_IMAGES_FOLDER = "./data/face_images"

def list_image(image_path, names):
    list_image = []
    list_name = []
    list_path = []

    for (j, name) in enumerate(names):
        imagePath = os.path.join(image_path, name)
        imagePath = list(glob.iglob(os.path.join(imagePath, '*.*')))
        list_path = list_path + imagePath
    random.shuffle(list_path)
    for (i, imagePath) in enumerate(list_path):
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
# list_names.sort()
images, names = list_image(FACE_IMAGES_FOLDER, list_names)

resnet50_features = VGGFace(model='resnet50',
                            include_top=False,
                            input_shape=(224, 224, 3),
                            pooling='avg')

features = resnet50_features.predict(images)


ohe = LabelEncoder()
id = ohe.fit_transform(names)
print(id.shape)
X_train, X_test, y_train, y_test = train_test_split(features, id, test_size=0.2, random_state=0)
print(y_train[0])
y_train = np_utils.to_categorical(y_train, 3)
print(y_train[0])
y_test = np_utils.to_categorical(y_test, 3)

print('building model')

model = Sequential()
# model.add(Dense(2048,activation='sigmoid'))
model.add(Dense(3, activation='softmax'))
checkpoint = ModelCheckpoint('models/model.h5',
                                monitor='val_loss',
                                verbose=0,
                                save_best_only=True,
                                mode='auto')
model.compile(loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
'''
# params = {'C' : [0.1, 1.0, 10.0, 100.0]}
# model = GridSearchCV(LogisticRegression(), params)
# checkpoint = ModelCheckpoint('models/model.h5',
#                                monitor='val_loss',
#                                verbose=0,
#                                save_best_only=True,
 #                               mode='auto')
# print('Best parameter for the model {}'.format(model.best_params_))
'''
print ('Start fitting model')
model.fit(X_train, y_train, epochs=10, verbose=1, validation_split=0.2, callbacks=[checkpoint]) 
preds = model.evaluate(X_test, y_test)
print("score ={} ".format(preds))   

'''

'''
# 4 
resnet50_features = VGGFace(model='resnet50',
                            include_top=False,
                            input_shape=(224, 224, 3),
                            pooling='avg')

model = load_model('./models/model.h5')

img = image.load_img('2.png', target_size=(224, 224))
img = image.img_to_array(img)

img = np.expand_dims(img, axis=0)
img = utils.preprocess_input(img)
print(resnet50_features.predict(img).shape)

score = model.predict(resnet50_features.predict(img))
print(score)



# 5

def sum(numbers):
    sum = []
    ret = []
    for number in numbers:
        sum += number
'''

# 6
dicts = [
    { "name": "Tom", "age": 10 },
    { "name": "Mark", "age": 5 },
    { "name": "Pam", "age": 7 },
    { "name": "Dick", "age": 12 }
]

a = next(item for item in dicts if item["name"] == "Pam")
print(a.get('age'))