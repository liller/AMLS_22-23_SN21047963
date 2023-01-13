import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from Modules.plot_result import plot_history, plot_confusion_matrix


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.svm import SVC
import itertools

import tensorflow as tf

from keras.preprocessing import image

from keras.preprocessing.image import array_to_img, img_to_array, load_img
# import scipy.misc
from keras import optimizers
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau,EarlyStopping
# from keras.callbacks import EarlyStopping


class A1_CNN:
    def __init__(self):
        # Set the CNN model
        # my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out
        print("Construct CNN model =====")
        self.model = Sequential([
            Conv2D(filters=32, kernel_size=(5, 5), padding='Same',activation='relu', input_shape=(218, 178, 3)),
            Conv2D(filters=32, kernel_size=(5, 5), padding='Same',activation='relu'),
            MaxPool2D(pool_size=(2, 2)),
            # 每一个神经元被丢弃的概率
            Dropout(0.25),
            Conv2D(filters=64, kernel_size=(3, 3), padding='Same',activation='relu'),
            Conv2D(filters=64, kernel_size=(3, 3), padding='Same',activation='relu'),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(256, activation="relu"),
            Dropout(0.5),
            Dense(2, activation='softmax')
        ])
        print("Summary of the CNN model")
        self.model.summary()
        optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        self.model.model.compile(optimizer = optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    def train(self, x_train, y_train, x_val, y_val, path, epochs= 30, batch_size= 32, plot= True):
         print("Training CNN model =====")
         history = self.model.fit(x_train, y_train, epochs = epochs,batch_size=batch_size,
                                  validation_data = (x_val, y_val), verbose = 1)
         if plot:
             plot_history(history,path)
         return history


    def test(self, x_test, y_test, confusion_mat=False):
        print("Test CNN model on test set=====")
        pred=self.model.predict(x_test,verbose=1)
        pred = np.round(pred)
        predicted_labels = np.array(np.argmax(pred, axis=1))
        true_labels = np.array(np.argmax(y_test, axis=1))
        score = accuracy_score(true_labels, predicted_labels)
        if confusion_mat:
            plot_confusion_matrix(np.argmax(y_test,axis = 1),np.argmax(pred,axis = 1))
        return score

class A1_SVM:
    def __init__(self, kernal, degree=3, gamma=0.7, C=1,):
        print("===== Construct SVM model with different kernal =====")
        self.model = SVC(kernel=kernal, degree=degree, gamma=gamma, C=C)

    def train(self, x_train,y_train):
        print("===== Training SVM model =====")
        self.model.fit(x_train, y_train)
        train_accuracy = self.model.score(x_train,y_train)

        return train_accuracy

    def test(self, x_test,y_test, confusion_mat=False):
        print("===== Test SVM model on test set=====")
        y_predict = self.model.predict(x_test)
        test_accuracy = accuracy_score(y_test,y_predict)
        if confusion_mat:
            plot_confusion_matrix(y_test,y_predict)
        return test_accuracy


























