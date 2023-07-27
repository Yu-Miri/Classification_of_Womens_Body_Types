!pip install opencv-python
!pip install split-folders

import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def resize_img(img_path, img_size=224):    
    img = cv2.imread(img_path) # 행렬 변환

    if(img.shape[1] > img.shape[0]) : # 한 변의 길이를 비교하여 더 긴 변의 길이에 맞게 비율 변환 
        ratio = img_size/img.shape[1]
    else :
        ratio = img_size/img.shape[0]

    img = cv2.resize(img, dsize=(0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)

    # 그림 주변에 검은색으로 패딩 처라
    w, h = img.shape[1], img.shape[0]

    dw = (img_size-w)/2 # img_size와 w의 차이
    dh = (img_size-h)/2 # img_size와 h의 차이

    M = np.float32([[1,0,dw], [0,1,dh]])  #(2*3 이차원 행렬)
    img_re = cv2.warpAffine(img, M, (224, 224)) #이동변환  
    img_re = cv2.cvtColor(img_re, cv2.COLOR_BGR2GRAY).reshape(224,224,1) #그레이스케일

    return img_re

def compute_mean(imgs):
    return np.mean(imgs, axis=0)

def mean_img(train_dataset): # mean_img 생성
    train_data = []

    for label, filenames in train_dataset.items():
        for filename in filenames:
            image = resize_img(filename)
            train_data.append(image)
    train_data = np.array(train_data)
    mean_img = compute_mean(train_data)

    with open('mean_img.pickle', 'wb') as f:
        pickle.dump(mean_img, f)
    
    return mean_img

def zero_centering(image):
    sub_mean_img = image.astype('int8') - mean_img.astype('int8')
    return sub_mean_img

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).reshape(224, 224, 1)
    # img = img - mean_img
    return img

def split_dataset(my_dataset): # 학습을 위한 데이터셋 split
    splitfolders.ratio(my_dataset, output = 'datasets', seed=77, ratio=(0.8, 0.1, 0.1))
    # train, validation, test

def train_data_generator(data_dic):
    data_dic = ImageDataGenerator(
        rotation_range=30,
        # width_shift_range=0.2,
        height_shift_range=0.3,
        # shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        # vertical_flip=True,
        preprocessing_function=grayscale # Gray-sclaing 적용
    )
    return data_dic

def val_test_data_generator(data_dic):
    data_dic = ImageDataGenerator(
        preprocessing_function=grayscale # Gray-sclaing 적용
    )