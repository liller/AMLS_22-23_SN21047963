
import pandas as pd
import os
import random
random.seed(10)
import shutil
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from keras.preprocessing.image import ImageDataGenerator



img_path = './Datasets/cartoon_set/img/'
csv_path = './Datasets/cartoon_set/labels.csv'
img_path_test = './Datasets/cartoon_set_test/img/'

def mkdir():
    folder = os.path.exists(img_path_test)
    if not folder:  # Determine if a folder exists if not then create as a folder
        os.makedirs(img_path_test)  # makedirs creates the path if it does not exist when creating the file
        print("Successfully make a new directory")
    else:
        print('The cartoon_set_test folder exists')


def data_preprocessing_generator(target_column, target_size, train_ratio = 0.80,validation_ratio = 0.10, test_ratio = 0.10, random_seed = 2):
    mkdir()
    print('# Loading and transforming dataset ========')
    print('# Randomly dividing data into two directory ====')
    # Store the names of the images in the original dataset in a list
    files = sorted(os.listdir(img_path), key=lambda x: int(x.split(".")[0]))
    print(f"length of original files is : {len(files)}")
    # A random selection from the files is used as the test set
    files_test = sorted(random.sample(files ,round(test_ratio*len(files))),key=lambda x: int(x.split(".")[0]))
    print(f"length of test files is : {len(files_test)}")
    # #Move the randomly selected test to img_path_test
    for f in  files_test:
        shutil.move(img_path+f,img_path_test)
    print(len(os.listdir(img_path_test)))
    print(len(os.listdir(img_path)))

    print("loading label file and clean the data ====")
    # # Read the label file, get the face_shape in the sample, and convert it to a list
    labels = pd.read_csv(csv_path, sep='\t', usecols=[1, 2, 3], header=0)
    # print(labels)
    # The data in the column where the label is located is not a string type and needs to be converted
    labels['face_shape'] = labels['face_shape'].astype(str)
    labels['eye_color'] = labels['eye_color'].astype(str)
    # List of all the images within the training and testing folders
    # Iterate through the list of addresses in the training set, the test set, and obtain a list of images in the order of the paths, respectively
    # labels['file_name'] == i return the corresponding index, so after traversing the list of addresses you can return the index of each address in the label (df)
    test_index = [labels[labels['file_name'] == i].index[0]
                  for i in os.listdir(img_path_test)]
    training_index = [labels[labels['file_name'] == i].index[0]
                      for i in os.listdir(img_path)]

    train_labels = labels.iloc[[i for i in training_index]]
    test_labels = labels.iloc[[i for i in test_index]]

    print('files and labels order')
    print(train_labels)
    print(os.listdir(img_path))
    print(test_labels)
    print(os.listdir(img_path_test))

    print('Dividing dataset into trainingset, validation set, test set ====')
    # ,validation_split = 1/9
    datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=1 / 9)
    train_batch = datagen.flow_from_dataframe(
        dataframe=train_labels,
        directory=img_path,
        x_col='file_name',
        y_col=target_column,
        subset='training',
        target_size=target_size,
        color_mode='rgb',
        batch_size=32)

    valid_batch = datagen.flow_from_dataframe(
        dataframe=train_labels,
        directory=img_path,
        x_col='file_name',
        y_col=target_column,
        subset='validation',
        target_size=target_size,
        color_mode='rgb',
        batch_size=32)

    ##Use shuffle = false here or the generator will break up and cause an order exception.
    # Set this to False(For Test generator only, for others set True), because you need to yield the images in “order”, to predict the outputs and match them with their unique ids or filenames.
    testdatagen = ImageDataGenerator(rescale=1.0 / 255)
    test_batch = testdatagen.flow_from_dataframe(
        dataframe=test_labels,
        directory=img_path_test,
        x_col='file_name',
        y_col=target_column,
        shuffle=False,
        target_size=target_size,
        color_mode='rgb',
        batch_size=32)


    return train_batch, valid_batch, test_batch


















