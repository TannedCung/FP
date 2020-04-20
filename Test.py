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
import dlib
from keras.engine import  Model
from tkinter import *
from tkinter import ttk


Cascade_path = "./pretrained_models/haarcascade_frontalface_alt.xml"
eye_path = "./pretrained_models/haarcascade_eye.xml"


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
'''

# 7
'''
class FaceIdentify():

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
            faces_imgs[i, :, :, :] = face_img
            points.append(cropped)
        return face_imgs, points
    
    def draw_label(cls, frame, points, labels, scores, font=cv2.FONT_HERSHEY_SCRIPT_COMPLEX, font_scale=0.5,
                    thickness=1):
        Stds = load_pickle("./data/pickle/Students.pickle")
        red = (10, 20, 255)
        green = (30, 255, 10)
        black = (0, 0, 0)
        for i,point in enumerate(points):
            size = cv2.getTextSize(labesl[i], font, font_scale, thickness)[0]
            (x,y,w,h)= point
            label = "{}".format(labels[i])
            score = "{:.3f}%".format(scores[i]*100)
                if label == "Unknown":
                    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), red, cv2.FILLED)
                    cv2.rectangle(frame, (x,y), (x+w, y+h), red)
                    cv2.putText(image, label, (x, y), font, font_scale, black, thickness)
                else:
                    infor = next(item for item in Stds if item["name"] == label[i])
                    cv2.rectangle(frame, (x,y), (x+w, y+h), green)
                    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), green, cv2.FILLED)
                    cv2.putText(image, infor.get('name') + "   " + score, (x, y), font, font_scale, black, thickness)
                    cv2.rectangle(image, (x, y), (x + size[0], y + size[1]), green, cv2.FILLED)
                    cv2.putText(image, "ID: {}".format(infor.get("ID")), (x, y + size[1]), font, font_scale, black, thickness)
                    cv2.rectangle(image, (x, y + size[1]), (x + size[0], y + ), green, cv2.FILLED)
                    cv2.putText(image, "school_year: {}".format(infor.get("school_year")), (x,*size[1] y), font, font_scale, black, thickness)
'''
'''
# 8 
face_detect = cv2.CascadeClassifier(Cascade_path)
video = cv2.VideoCapture(0)
frame_counter = 0
faceTracker = {}
face_number = 0
carCurrentPosition = {}
while True:
    _, frame = video.read()

    if frame is None:
        sleep(5)
    
    if not (frame_counter % 50):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detect.detectMultiScale(frame, scaleFactor=1.1,
                                            minNeighbors=50, minSize=(50,50), flags=cv2.CASCADE_SCALE_IMAGE)
        for (x, y, w, h) in faces :
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)

			# Tinh tam diem cua car
            x_center = x + 0.5 * w
            y_center = y + 0.5 * h


            matchfaceID = None
			# Duyet qua cac car da track
            for faceID in faceTracker.keys():
                # Lay vi tri cua car da track
                trackedPosition = faceTracker[faceID].get_position()
                t_x = int(trackedPosition.left())
                t_y = int(trackedPosition.top())
                t_w = int(trackedPosition.width())
                t_h = int(trackedPosition.height())
                # Tinh tam diem cua car da track
                t_x_center = t_x + 0.5 * t_w
                t_y_center = t_y + 0.5 * t_h

                # Kiem tra xem co phai ca da track hay khong
                if (t_x <= x_center <= (t_x + t_w)) and (t_y <= y_center <= (t_y + t_h)) and (x <= t_x_center <= (x + w)) and (y <= t_y_center <= (y + h)):
                    matchfaceID = faceID

			# Neu khong phai car da track thi tao doi tuong tracking moi
            if matchfaceID is None:

                tracker = dlib.correlation_tracker()
                tracker.start_track(frame, dlib.rectangle(x, y, x + w, y + h))

                faceTracker[face_number] = tracker
                # carStartPosition[face_number] = [x, y, w, h]

                face_number +=1

	# Thuc hien update position cac car
    for faceID in faceTracker.keys():
        trackedPosition = faceTracker[faceID].get_position()

        t_x = int(trackedPosition.left())
        t_y = int(trackedPosition.top())
        t_w = int(trackedPosition.width())
        t_h = int(trackedPosition.height())

        cv2.rectangle(frame, (t_x, t_y), (t_x + t_w, t_y + t_h), (255,0,0), 4)
        carCurrentPosition[faceID] = [t_x, t_y, t_w, t_h]
        faceTracker[faceID].update(frame)
        print (carCurrentPosition[faceID])
    cv2.imshow("vd", frame)
    if cv2.waitKey(6) == 27:
        break
'''
# 9 

'''resnet50_features = VGGFace(model='resnet50',
                            include_top=False,
                            input_shape=(224, 224, 3),
                            pooling='avg')  # pooling: None, avg or max

nb_class = 30
vgg_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3))
# vgg_model.summary()
last_layer = vgg_model.get_layer('activation_49').output
print(type(last_layer))
x = Flatten(name='flatten')(last_layer)
out = Dense(nb_class, activation='softmax', name='classifier')(x)
custom_vgg_model = Model(vgg_model.input, out)
custom_vgg_model.summary()



resnet50_features2 = VGGFace(model='resnet50')
                            # input_shape=(224, 224, 3),
                            # pooling='avg')  # pooling: None, avg or max
print("include_top = True")
resnet50_features2.summary()
'''
'''
# 10
for p, n, f in os.walk('./train_data'):
    for folder in n:
        path = os.path.join(p, folder)
        files = list(glob.iglob(os.path.join(path, "*.*")))
        if len(files) < 20:
            utlis.agumentate(path)
        files1 = list(glob.iglob(os.path.join(path, "*.*")))
        print ('{},  {} -> {}'.format(folder, len(files), len(files1)))\
'''
'''
# 11
def ret(a):
    if a > 1:
        return 1
    return 0

print(ret(2))
print("-----------------------")
print(ret(0))
'''
# 12

tk = Tk()

def submit():
    n = str(name.get())
    i = int(id.get())
    sy = int(school_year.get())
    a = Label(tk, text="{}...{}...{}".format(n,i,sy))
    a.pack()
name = Entry(tk, width=50)
name.pack()
name.insert(0, "Tell me your name: ")
id = Entry(tk, width=50)
id.pack()
id.insert(0, "Your ID: ")
school_year = Entry(tk, width=50)
school_year.pack()
school_year.insert(0, "And your schoolyear: ")

submit = Button(tk, text="Submit", command=submit)
submit.pack()

def click(event):


options = ['1', '2', '3']

cb = ttk.Combobox(tk, value=options)
cb.current(0)
cb.bind("<<ComboboxSelected>>", click)
cb.pack

exit = Button(tk, text="Exit", command=tk.destroy)
exit.pack()

tk.mainloop()
'''

# 13
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
add = "./data/pickle/Students.pickle"

files = load_pickle(add)
print(files)

'''