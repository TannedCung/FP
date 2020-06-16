import numpy as np
import cv2
import os
import glob
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

path = './mils/Keras_face_identification_realtime-master/data/face_images/barack'

def load_image(img_path):
    ''' need: an img_path that contains images inside
        return: list of images in shape (1, 224, 224, 3)
    '''
    face_images = list(glob.iglob(os.path.join(img_path, '*.*')))
    # print(face_images)
    list_image = []
    for imagePath in face_images:
        img = load_img(imagePath)
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        # print(img.shape)
        list_image.append(img)
    return np.vstack(list_image)


def save_img(img_path, img, What_to_agu, i, j):
    name = img_path + '/' + os.path.basename(img_path)+ '_' + str(What_to_agu) + '_' + str(i + j) + '.png'
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(name, img)


def agu_brightness(img_path):
    agu_images = load_image(img_path)
    myImageGen = ImageDataGenerator(brightness_range=[0.5,2.0])
    for (j,image) in enumerate(agu_images):
        image = np.expand_dims(image, axis=0)
        gen = myImageGen.flow(image, batch_size=1)
        for i in range(5):
            save_img(img_path, gen.next()[0].astype('uint8'), 'brightness', i, j) # if np.random.rand()<0.4 else 0

            
def agu_shear(img_path):
    agu_images = load_image(img_path)
    myImageGen = ImageDataGenerator(shear_range=45)
    for (j,image) in enumerate(agu_images):
        image = np.expand_dims(image, axis=0)
        gen = myImageGen.flow(image, batch_size=1)
        for i in range(5):
            save_img(img_path, gen.next()[0].astype('uint8'), 'shear', i, j) if np.random.rand() < 0.2 else 0

def agu_rotate(img_path):
    agu_images = load_image(img_path)
    myImageGen = ImageDataGenerator(rotation_range=45)
    for (j, image) in enumerate(agu_images):
        image = np.expand_dims(image, axis=0)
        gen = myImageGen.flow(image, batch_size=1)
        for i in range(5):
            save_img(img_path, gen.next()[0].astype('uint8'),'rotate', i, j) if np.random.rand() < 0.2 else 0

def agu_shift(img_path):
    agu_images = load_image(img_path)
    myImageGen = ImageDataGenerator(width_shift_range=45)
    for (j, image) in enumerate(agu_images):
        image = np.expand_dims(image, axis=0)
        gen = myImageGen.flow(image, batch_size=1)
        for i in range(5):
            save_img(img_path, gen.next()[0].astype('uint8'), 'shift', i, j) if np.random.rand() < 0.1 else 0


def agumentate(img_path):
    agu_brightness(img_path)
    agu_rotate(img_path)
    # agu_shear(img_path)
    # agu_shift(img_path)
