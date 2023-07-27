import cv2
import datetime as dt
import h5py
import numpy as np
import os
import tensorflow as tf
import fnmatch
import matplotlib.pyplot as plt
import IPython.display as display
from PIL import Image
from img_preprocess import resize_img

def inference(image_filename):
    resnet = tf.keras.models.load_model('./ensemble/model_resnet.h5')
    mean_img = cv2.imread('./dataset/mean_image_1000.jpg')

    image_example=cv2.imread(image_filename, cv2.IMREAD_COLOR) #이미지파일

    new_img=resize_img(image_example)
    sub_img = new_img - mean_img
    label2index={
        '원형':0,
        '모래시계형':1,
        '일자형':2,
        '삼각형':3
    }
    index2label={v:k for k, v in label2index.items()}
    pred = np.argmax(model_resnet.predict(np.expand_dims(new_img, axis=0),verbose=0), axis=-1)[0]
    plt.imshow(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))

    print("당신의 체형은 ?\n")
    print(index2label[pred])