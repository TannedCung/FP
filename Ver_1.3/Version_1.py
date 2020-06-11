# import compute_feature
# import Recognition
import cv2
import numpy as np
import os
import glob
import pickle
import path
import keras
import tensorflow as tf
from Student import *
import compute_feature

from tkinter import font, Tk
from tkinter import *


# 1st run\

"""
com = compute_feature.FaceExtractor()
names = {"barack", "michelle", "Nguyen Phu Trong", "Nguyen Xuan Phuc", "Unknown"}
for name in names:
    com.compute_features(name)

com.biuld_new_model()


std = Student()
std.save_infor(name="Nguyen Phu Trong")
std.save_infor(name="Nguyen Xuan Phuc")
std.save_infor(name="Unknown")
"""


# 2
"""
FACE_IMAGES_FOLDER = "./data/face_images"
FEATURE_FOLDER = "./data/pickle/features"
PICKLE_FOLDER = "./data/pickle"

path = FACE_IMAGES_FOLDER + "/Unknown/*.*"
save_folder = FACE_IMAGES_FOLDER + "/Unknown_1/"
img_path = glob.iglob(path)

for i, p in enumerate(img_path):
    pic = cv2.imread(p)
    cv2.resize(pic, (224,224))
    cv2.imwrite(save_folder+"Unknown_{}.png".format(i), pic)
"""
#3
"""
t = Tk()

def entry(i, f):
     i = Entry(t, font=(f, 25, "bold"))
     i.insert(0, f)
     i.pack()


fonts = font.families()
sorted(fonts)

for i, f in enumerate(fonts):
    entry(i,f)

t.mainloop()
"""

 #4 
"""
for i in range(4):
    pic = cv2.imread('sb{}.png'.format(i+1))
    pic = cv2.resize(pic, (100,100))
    cv2.imwrite("sb_{}.png".format(i+1), pic)

bt_home = cv2.imread('home.png')
bt_home = cv2.resize(bt_home, (85,33))
cv2.imwrite("home.png", bt_home)

bt_sign_up = cv2.imread('sign_up.png')
bt_sign_up = cv2.resize(bt_sign_up, (85,31))
cv2.imwrite("sign_up.png", bt_sign_up)

bt_quit = cv2.imread('quit.png')
bt_quit = cv2.resize(bt_quit, (85,33))
cv2.imwrite("quit.png", bt_quit)

bt_have_mask = cv2.imread('have_mask.png')
bt_have_mask = cv2.resize(bt_have_mask, (85,33))
cv2.imwrite("have_mask.png", bt_have_mask)

bt_sign_up_2 = cv2.imread('sign_up_2.png')
bt_sign_up_2 = cv2.resize(bt_sign_up_2, (147,40))
cv2.imwrite("sign_up_2.png", bt_sign_up_2)
"""


# 5
"""

print("keras version: {}".format(keras.__version__))
print("tf version: {}".format(tf.__version__))
"""
