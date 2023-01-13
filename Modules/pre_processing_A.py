import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import os
import cv2
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding




file_path = './Datasets/celeba/img/'
csv_path = './Datasets/celeba/labels.csv'


def data_preprocessing(target_column, train_ratio = 0.80,validation_ratio = 0.10, test_ratio = 0.10, random_seed = 2):
    print('# Loading and transforming dataset =====')
    labels = pd.read_csv(csv_path,sep='\t',usecols=[1,2,3],header=0)
    samples_genderorsmile = labels[target_column]
    # samples_smile = labels['smiling']
    f_names = []
    imgs = []
    for i in range(len(labels)):
        f_name = file_path + labels.loc[i]['img_name']
        f_names.append(f_name)
    for i in range(len(f_names)):
        img = image.load_img(f_names[i])
        # Read the image and convert it to an arry
        img2arr = image.img_to_array(img)
        # Adjust the dimensionality of the image to add a dimension representing the number of samples
        img_di = np.expand_dims(img2arr, axis=0)
        imgs.append(img2arr)


    samples = np.array(imgs)
    # Check for null and missing values
    num_null = np.isnan(samples).sum()
    print(f"Checking for null and missing values, sum of NaN is :{num_null}")
    samples = samples / 255
    for i in range(len(samples_genderorsmile)):
        if samples_genderorsmile[i] == -1:
            samples_genderorsmile[i] = 0

    # if gray:
    #     samples = np.array(samples).reshape(len(samples), -1)
    #     samples_genderorsmile = np.array(samples_genderorsmile)
    samples_genderorsmile = to_categorical(samples_genderorsmile, num_classes=2)
    # print(samples_genderorsmile)


    # Split the train and the validation set for the fitting
    # As long as the random_state is the same, the random result of the division is the same
    # train is now 80% of the entire data set
    # the _junk suffix means that we drop that variable completely
    print('# Divide dataset =====')
    x_train, x_test, y_train, y_test = train_test_split(samples, samples_genderorsmile, test_size=1 - train_ratio,
                                                        random_state=random_seed)

    # test is now 10% of the initial data set
    # validation is now 10% of the initial data set
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test,
                                                    test_size=test_ratio / (test_ratio + validation_ratio),
                                                    random_state=random_seed)
    print(f"The shape of training set is :{x_train.shape}")
    print(y_train.shape)
    print(f"The shape of validation set is :{x_val.shape}")
    print(y_val.shape)
    print(f"The shape of test set is :{x_test.shape}")
    print(y_test.shape)
    return x_train, y_train, x_val, y_val, x_test, y_test


""







































#
#
#
# # Set the CNN model
# # my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out
# #softmax
#
# model = Sequential()
#
# model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',
#                  activation ='relu', input_shape = (218,178,3)))
# model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',
#                  activation ='relu'))
# model.add(MaxPool2D(pool_size=(2,2)))
# #每一个神经元被丢弃的概率
# model.add(Dropout(0.25))
#
#
# model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
#                  activation ='relu'))
# model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
#                  activation ='relu'))
# model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(256, activation = "relu"))
# model.add(Dropout(0.5))
# model.add(Dense(2, activation = "softmax"))
#
#
#
# model.summary()
# optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
#
# metrics=[]
# model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
#
# epochs = 30 # Turn epochs to 30 to get 0.9320 accuracy
# batch_size = 86
#
# history = model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs,
#           validation_data = (x_val, y_val), verbose = 1)
#
#
# # predict results
# # Model = load_model('cnn_model_gender.h5')
# y_predict = model.predict(x_test)