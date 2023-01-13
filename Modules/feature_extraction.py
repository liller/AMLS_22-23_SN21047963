import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import os
import cv2
import dlib
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


from sklearn.model_selection import train_test_split





file_path = './Datasets/celeba/img/'
csv_path = './Datasets/celeba/labels.csv'

img_path = './Datasets/cartoon_set/img/'
img_path_test = './Datasets/cartoon_set_test/img/'
csv_path_cartoon = './Datasets/cartoon_set/labels.csv'
predictor_path = './shape_predictor_68_face_landmarks.dat'
#人脸识别模型
face_rec_model_path = './dlib_face_recognition_resnet_model_v1.dat'

detector = dlib.get_frontal_face_detector() #a detector to find the faces
shape_pre = dlib.shape_predictor(predictor_path) #shape predictor to find face landmarks
face_rec = dlib.face_recognition_model_v1(face_rec_model_path) #face recognition model



def feature_extract(target_column, train_ratio = 0.90, random_seed = 3, predictor = False):
    print('# ===== Loading and extracting features =====')

    labels = pd.read_csv(csv_path,sep='\t',usecols=[1,2,3],header=0)

    f_names = []
    imgs = []
    face_descriptor = []
    landmarks = []
    face_smile = []
    face_gender = []

    for i in range(len(labels)):
        f_name = file_path + labels.loc[i]['img_name']
        f_names.append(f_name)

    for i in range(len(f_names)):
        img = cv2.imread(f_names[i])
        det_sample = detector(img, 0)
        #   det_sample:  rectangles[[(45, 93) (131, 179)], [(-27, 5) (53, 77)]]
        # 有的样本未能识别出脸 需要先过滤
        if len(det_sample) != 0:
            # 如果识别出一个脸，生成一组x1y1x2y2，两个脸则生成两组，需要剔除掉脏数据
            if len(det_sample) != 2:
                for k, d in enumerate(det_sample):
                    # Get the landmarks/parts for the face in box d
                    shape = shape_pre(img, d)
                    face_des = face_rec.compute_face_descriptor(img, shape)
                    face_descriptor.append(face_des)
                    landmark = np.matrix([[p.x, p.y] for p in shape.parts()])  # 可以把关键点转换成shape为(68,2)的矩阵
                    #             print(landmark)
                    landmarks.append(landmark)
            else:
                print('exist')
                # 2522 有两个脸
                for k, d in enumerate(det_sample):
                    shape = shape_pre(img, d)
                    face_des = face_rec.compute_face_descriptor(img, shape)
                    face_descriptor.append(face_des)
                    landmark = np.matrix([[p.x, p.y] for p in shape.parts()])  # 可以把关键点转换成shape为(68,2)的矩阵
                    #             print(landmark)
                    landmarks.append(landmark)
                    break
            face_smile.append(labels['smiling'][i])
            face_gender.append(labels['gender'][i])

    face_smile1 = np.array(face_smile)
    face_gender1 = np.array(face_gender)
    print(face_smile1.shape)
    print(face_gender1.shape)

    face_descriptor1 = np.array(face_descriptor)
    print(face_descriptor1.shape)

    landmarks1 = np.array(landmarks)
    landmarks1 = np.array(landmarks1).reshape(len(landmarks1), -1)
    print(landmarks1.shape)

    if target_column == 'gender':
        x_train, x_test, y_train, y_test = train_test_split(face_descriptor1, face_gender1, test_size=1 - train_ratio,
                                                        random_state=random_seed)
    if target_column == 'smiling':
        x_train, x_test, y_train, y_test = train_test_split(face_descriptor1, face_smile1, test_size=1 - train_ratio,
                                                            random_state=random_seed)
    if predictor == True:
        x_train, x_test, y_train, y_test = train_test_split(landmarks1, face_smile1, test_size=1 - train_ratio,
                                                            random_state=random_seed)
    print(len(x_train))
    print(len(x_test))
    print(len(y_train))
    return x_train, y_train, x_test, y_test


def feature_extract_dir(predictor = False):
    print('# ===== Loading and extracting features =====')
    labels = pd.read_csv(csv_path_cartoon,sep='\t',usecols=[1,2,3],header=0)
    # List of all the images within the training and testing folders
    # 将训练集 测试集中的地址列表遍历,分别得到按照路径下图片顺序序号的一个列表
    # labels['file_name'] == i返回对应的index,因此遍历地址列表后可以返回对应的每个地址在label（df）中的index
    test_index = [labels[labels['file_name'] == i].index[0]
                  for i in os.listdir(img_path_test)]
    training_index = [labels[labels['file_name'] == i].index[0]
                      for i in os.listdir(img_path)]

    ##train
    num = []
    f_names = []
    imgs = []
    face_descriptor_train = []
    face_shape_train = []
    eye_color_train = []

    # 读df
    for i in training_index:
        f_name = img_path + labels.loc[i]['file_name']
        f_names.append(f_name)
        num.append(labels[labels.file_name == labels.loc[i]['file_name']].index.tolist()[0])
    print(len(f_names))

    for i in range(len(f_names)):
        img = cv2.imread(f_names[i])
        # 先检测脸 返回一个/多个脸
        det_sample = detector(img, 0)
        #   det_sample:  rectangles[[(45, 93) (131, 179)], [(-27, 5) (53, 77)]]
        # 有的样本未能识别出脸 需要先过滤
        if len(det_sample) != 0:
            # 如果识别出一个脸，生成一组x1y1x2y2，两个脸则生成两组，需要剔除掉脏数据
            for k, d in enumerate(det_sample):
                # Get the landmarks/parts for the face in box d
                shape = shape_pre(img, d)
                face_des = face_rec.compute_face_descriptor(img, shape)
                face_descriptor_train.append(face_des)
            eye_color_train.append(labels['eye_color'][num[i]])
            face_shape_train.append(labels['face_shape'][num[i]])

    ##test
    # 清空之前列表内容 放在函数中成为局部变量 不需要考虑
    num = []
    f_names = []
    imgs = []
    face_descriptor_test = []
    face_shape_test = []
    eye_color_test = []

    # 读df
    for i in test_index:
        f_name = img_path_test + labels.loc[i]['file_name']
        f_names.append(f_name)
        num.append(labels[labels.file_name == labels.loc[i]['file_name']].index.tolist()[0])
    print(len(f_names))

    for i in range(len(f_names)):
        img = cv2.imread(f_names[i])
        det_sample = detector(img, 0)
        #   det_sample:  rectangles[[(45, 93) (131, 179)], [(-27, 5) (53, 77)]]
        # 有的样本未能识别出脸 需要先过滤
        if len(det_sample) != 0:
            # 如果识别出一个脸，生成一组x1y1x2y2，两个脸则生成两组，需要剔除掉脏数据
            for k, d in enumerate(det_sample):
                # Get the landmarks/parts for the face in box d
                shape = shape_pre(img, d)
                face_des = face_rec.compute_face_descriptor(img, shape)
                face_descriptor_test.append(face_des)
            eye_color_test.append(labels['eye_color'][num[i]])
            face_shape_test.append(labels['face_shape'][num[i]])

    eye_color_test1 = np.array(eye_color_test)
    # print(face_smile1[:50])
    print(eye_color_test1.shape)

    eye_color_train1 = np.array(eye_color_train)
    # print(eye_color1)
    print(eye_color_train1.shape)

    face_shape_test1 = np.array(face_shape_test)
    print(face_shape_test1.shape)

    face_shape_train1 = np.array(face_shape_train)
    print(face_shape_train1.shape)

    face_descriptor_train1 = np.array(face_descriptor_train)
    print(face_descriptor_train1.shape)

    face_descriptor_test1 = np.array(face_descriptor_test)
    print(face_descriptor_test1.shape)

    print(len(face_descriptor_train1[0]))
    print(type(face_descriptor_train1[0]))

# # Split the train and the validation set for the fitting
    x_train = face_descriptor_train1
    x_test = face_descriptor_test1

    y_train_eye = eye_color_train1
    y_test_eye = eye_color_test1

    y_train_face = face_shape_train1
    y_test_face = face_shape_test1


    return x_train, x_test, y_train_eye, y_test_eye, y_train_face, y_test_face
