# import compute_feature
# import Recognition
import cv2
import numpy as np
import os
import glob
import pickle
import path
import compute_feature


# 1st run\

com = compute_feature.FaceExtractor()
names = {"barack", "michelle", "Nguyen Phu Trong", "Nguyen Xuan Phuc", "Unknown"}
for name in names:
    com.compute_features(name)

com.biuld_new_model()


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