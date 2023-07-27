from matplotlib import pyplot as plt
import tensorflow
from tensorflow import keras
from tensorflow.keras import datasets, layers
from tensorflow.keras.layers import Dense, Flatten, MaxPool2D
from tensorflow.keras import Input
from tensorflow.keras.layers import Dropout, BatchNormalization
import tensorflow as tf
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications.resnet import ResNet101
from img_preprocess import train_data_generator, val_test_data_generator
from model import resnet101, vgg16

def training(train_data_dic, val_data_dic, test_data_dic):
    batch_size = 64
    train_datagen = train_data_generator(train_data_dic)
    val_datagen = val_test_data_generator(val_data_dic)
    test_datagen = val_test_data_generator(test_data_dic)

    train_generator = train_datagen.flow_from_directory(train_data_dic,batch_size=batch_size, target_size=(224, 224), class_mode = 'sparse')
    val_generator = val_datagen.flow_from_directory(val_data_dic,batch_size=batch_size, target_size=(224, 224), class_mode = 'sparse')
    test_generator = test_datagen.flow_from_directory(test_data_dic,batch_size=batch_size, target_size=(224, 224), class_mode = 'sparse')

    # model
    inputs, outputs = resnet101()

    resnet = tf.keras.Model(inputs, outputs)
    resnet.compile(optimizer = tf.keras.optimizers.Adam(learning_rate= 0.00001),
                    loss = 'sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    resnet.summary()
    es = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 15)
    lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', patience = 7, factor = 0.1, verbose = 0)
    
    history = resnet.fit(train_generator, validation_data = val_generator, epochs = 100, callbacks = [es, lr])
    plt.plot(history.history['val_loss'])

    print(resnet.evaluate(test_generator))