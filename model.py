from tensorflow.keras import datasets, layers
from tensorflow.keras.layers import Dense, Flatten, MaxPool2D
from tensorflow.keras import Input
from tensorflow.keras.layers import Dropout, BatchNormalization
import tensorflow as tf
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications.resnet import ResNet101
import torch
import torch.nn

def resnet101():
    base_model = ResNet101(include_top=False, weights = 'imagenet') # imagenet으로 pretrained model을 분류기만 학습
    base_model.trainable = True

    inputs = tf.keras.Input(shape = (224, 224, 1)) #input tensor
    inputs = tf.repeat(inputs, 3, axis=-1)

    # x = tf.keras.layers.Conv2D(3, (3, 3), padding='same')(inputs)

    x = base_model(inputs, training = False)
    x = tf.keras.layers.Flatten(input_shape=base_model.output_shape[1:])(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x= tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x= tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(4, activation='softmax')(x)

    return inputs, outputs

def vgg16():
    inputs = Input(shape=(224, 224, 3))
    x = Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same', activation='relu')(inputs)
    x = Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=(2,2), strides=2)(x)

    x = Conv2D(filters=128, kernel_size=(3,3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(filters=128, kernel_size=(3,3), strides=1, padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=(2,2), strides=2)(x)

    x = Conv2D(filters=256, kernel_size=(3,3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(filters=256, kernel_size=(3,3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(filters=256, kernel_size=(3,3), strides=1, padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=(2,2), strides=2)(x)

    x = Conv2D(filters=512, kernel_size=(3,3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(filters=512, kernel_size=(3,3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(filters=512, kernel_size=(3,3), strides=1, padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=(2,2), strides=2)(x)

    x = Conv2D(filters=512, kernel_size=(3,3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(filters=512, kernel_size=(3,3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(filters=512, kernel_size=(3,3), strides=1, padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=(2,2), strides=2)(x)

    x = Flatten()(x)

    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)
    outputs = Dense(4, activation='softmax')(x)
    
    return inputs, outputs

def googlenet():
    googlenet = models.googlenet(pretrained=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    googlenet.fc=nn.Sequential(
        nn.Linear(in_features=1024,out_features=512,bias=True),
        nn.ReLU(),
        nn.Linear(in_features=512,out_features=128,bias=True),
        nn.ReLU(),
        nn.Linear(in_features=128,out_features=32,bias=True),
        nn.ReLU(),
        nn.Linear(in_features=32,out_features=4,bias=True),

        ) 
    googlenet.to(device)

    def alexnet():
        model=Sequential([
        tensorflow.keras.layers.Conv2D(96,kernel_size=11,input_shape=(224,224,3),strides=4,activation='relu'),    
        tensorflow.keras.layers.MaxPooling2D(3,2),
        tensorflow.keras.layers.BatchNormalization(),
        tensorflow.keras.layers.Conv2D(256,kernel_size=5,padding='same',activation='relu'), 
        tensorflow.keras.layers.MaxPooling2D(3,2),
        tensorflow.keras.layers.BatchNormalization(),
        tensorflow.keras.layers.Conv2D(384,kernel_size=3,padding='same',activation='relu'),
        tensorflow.keras.layers.Conv2D(384,kernel_size=3,padding='same',activation='relu'),
        tensorflow.keras.layers.Conv2D(256,kernel_size=3,padding='same',activation='relu'),
        tensorflow.keras.layers.MaxPooling2D(3,2),
        tensorflow.keras.layers.BatchNormalization(),
        tensorflow.keras.layers.Flatten(),
        tensorflow.keras.layers.Dense(4096,activation='relu'),
        tensorflow.keras.layers.Dense(4096,activation='relu'),
        tensorflow.keras.layers.Dense(4,activation='softmax')])
        
        model.summary()