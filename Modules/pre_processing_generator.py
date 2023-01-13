
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
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(img_path_test)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("Successfully make a new directory")
    else:
        print('The cartoon_set_test folder exists')


def data_preprocessing_generator(target_column, target_size, train_ratio = 0.80,validation_ratio = 0.10, test_ratio = 0.10, random_seed = 2):
    mkdir()
    print('# Loading and transforming dataset ========')
    print('# Randomly dividing data into two directory ====')
    # 将原始数据集中的图片名称存入一个list
    files = sorted(os.listdir(img_path), key=lambda x: int(x.split(".")[0]))
    print(f"length of original files is : {len(files)}")
    # 从files中随机抽取一部分作为test集
    files_test = sorted(random.sample(files ,round(test_ratio*len(files))),key=lambda x: int(x.split(".")[0]))
    print(f"length of test files is : {len(files_test)}")
    # #将随机抽取的test移动至img_path_test
    for f in  files_test:
        shutil.move(img_path+f,img_path_test)
    print(len(os.listdir(img_path_test)))
    print(len(os.listdir(img_path)))

    print("loading label file and clean the data ====")
    # # 读取label文件,获取样本中face_shape,将其转换为list
    labels = pd.read_csv(csv_path, sep='\t', usecols=[1, 2, 3], header=0)
    # print(labels)
    # label所在的列中的数据不是 string类型，需要进行转换
    labels['face_shape'] = labels['face_shape'].astype(str)
    labels['eye_color'] = labels['eye_color'].astype(str)
    # List of all the images within the training and testing folders
    # 将训练集 测试集中的地址列表遍历,分别得到按照路径下图片顺序序号的一个列表
    # labels['file_name'] == i返回对应的index,因此遍历地址列表后可以返回对应的每个地址在label（df）中的index
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

    ##这个地方要用shuffle = false 否则generator会打乱 导致顺序异常
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


















