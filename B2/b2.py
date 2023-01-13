import numpy as np
import os
import random
random.seed(10)
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from Modules.plot_result import plot_history, plot_confusion_matrix
from keras.callbacks import ReduceLROnPlateau


class B2_CNN:
    def __init__(self):
        # Set the CNN model
        print("Construct CNN model =====")
        self.model = Sequential([
            Conv2D(filters=32, kernel_size=(5, 5), padding='Same',activation='relu', input_shape=(500, 500, 3)),
            Conv2D(filters=32, kernel_size=(5, 5), padding='Same',activation='relu'),
            MaxPool2D(pool_size=(2, 2)),
            #  Probability of each neuron being discarded
            Dropout(0.25),
            Conv2D(filters=64, kernel_size=(3, 3), padding='Same',activation='relu'),
            Conv2D(filters=64, kernel_size=(3, 3), padding='Same',activation='relu'),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(256, activation="relu"),
            Dropout(0.5),
            Dense(5, activation='softmax')
        ])
        print("Summary of the CNN model")
        self.model.summary()
        optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        self.model.compile(optimizer = optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    def train(self, train_batch, valid_batch, path, epochs= 30, plot= True):
         print("Training CNN model =====")
         learning_rate_reduction = ReduceLROnPlateau(monitor="val_loss",
                                                     patience=3,
                                                     verbose=1,
                                                     factor=0.5,
                                                     min_lr=0.00001)
         history = self.model.fit_generator(train_batch, steps_per_epoch=len(train_batch), validation_data = valid_batch,
                                  validation_steps=len(valid_batch), epochs=epochs, verbose=1,callbacks=[learning_rate_reduction])
         if plot:
             plot_history(history, path)
         return history


    def test(self, test_batch, confusion_mat=False):
        print("Test CNN model on test set=====")
        pred=self.model.predict_generator(test_batch, steps = 32, verbose=1)
        pred = np.round(pred)
        predicted_labels = np.array(np.argmax(pred, axis=1))
        true_labels = np.array(test_batch.classes)
        score = accuracy_score(true_labels, predicted_labels)
        if confusion_mat:
            plot_confusion_matrix(np.argmax(test_batch,axis = 1),np.argmax(pred,axis = 1))
        return score


class B2_SVM:
    def __init__(self, kernal, degree=3, gamma=0.7, C=1, ):
        print("===== Construct SVM model with different kernal =====")
        self.model = SVC(kernel=kernal, degree=degree, gamma=gamma, C=C)

    def train(self, x_train, y_train):
        print("===== Training SVM model =====")
        self.model.fit(x_train, y_train)
        train_accuracy = self.model.score(x_train, y_train)
        return train_accuracy


    def test(self, x_test,y_test, confusion_mat=False):
        print("===== Test SVM model on test set=====")
        y_predict = self.model.predict(x_test)
        test_accuracy = accuracy_score(y_test,y_predict)
        print(confusion_matrix(y_test, y_predict))
        print(classification_report(y_test, y_predict))
        if confusion_mat:
            plot_confusion_matrix(y_test,y_predict)
        return test_accuracy