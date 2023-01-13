import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix,accuracy_score
import pandas as pd


def plot_history(history,path):
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(history.history['loss'], color='b', label="Training loss")
    ax[0].plot(history.history['val_loss'], color='r', label="Validation loss", axes=ax[0])
    legend = ax[0].legend(loc='best', shadow=True)
    ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
    ax[1].plot(history.history['val_accuracy'], color='r', label="Validation accuracy")
    legend = ax[1].legend(loc='best', shadow=True)
    if os.path.isfile(path):
        plt.savefig(path)


def plot_confusion_matrix(y_test_array,y_predict):
    disp = confusion_matrix(y_test_array, y_predict, normalize='true')
    plt.figure(figsize=(4, 4))
    sns.heatmap(disp,annot=True,cmap='Blues')
    plt.ylim(0, 2)
    plt.title('Confusion matrix')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()




