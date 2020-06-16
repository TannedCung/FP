from tkinter import * 
from tkinter.ttk import *
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from keras.initializers import glorot_uniform
from keras.utils import CustomObjectScope


with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
  model = load_model('./models/mask_model.h5')

# model = keras.models.load_model("./models/model_mask.h5")
Cascade_path = "./pretrained_models/haarcascade_frontalface_alt.xml"
face_cascade = cv2.CascadeClassifier(Cascade_path)

cap = cv2.VideoCapture(0)

def is_mask_on(image):
  image = cv2.resize(image, (150,150))
  image = np.expand_dims(image, axis=0)
  score = model.predict(image)
  return np.argmax(score)

while 1:
  _, frame = cap.read()
  frame = np.fliplr(frame)
  frame = np.array(frame)

  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(
                                      gray,
                                      scaleFactor=1.2,
                                      minNeighbors=5,
                                      minSize=(32, 32))
  for face in faces:
    (x, y, w, h) = face
    cv2.rectangle(frame, (x,y), (x+w, y+h), (225, 225, 10), thickness=1)
    score = is_mask_on(frame[y:y+h, x:x+w])
    print(score)
    # frame = cv2.putText(frame, score, (x,y), cv2.FONT_HERSHEY_SIMPLEX,  
    #              fontScale=1, color=(225,225,10), thickness=1, lineType=cv2.LINE_AA) 
  cv2.imshow("img", frame)
  if cv2.waitKey(5) == 27:
    break

