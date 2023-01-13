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
#Face recognition models
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
        #det_sample:  rectangles[[(45, 93) (131, 179)], [(-27, 5) (53, 77)]]
        # Some samples fail to recognise faces and need to be filtered first
        if len(det_sample) != 0:
            # If one face is recognized, a set of x1y1x2y2 is generated, and two faces generate two sets, and dirty data needs to be eliminated
            if len(det_sample) != 2:
                for k, d in enumerate(det_sample):
                    # Get the landmarks/parts for the face in box d
                    shape = shape_pre(img, d)
                    face_des = face_rec.compute_face_descriptor(img, shape)
                    face_descriptor.append(face_des)
                    landmark = np.matrix([[p.x, p.y] for p in shape.parts()])  # The key points can be converted into a matrix with a shape of (68,2)
                    #             print(landmark)
                    landmarks.append(landmark)
            else:
                # print('exist')
                for k, d in enumerate(det_sample):
                    shape = shape_pre(img, d)
                    face_des = face_rec.compute_face_descriptor(img, shape)
                    face_descriptor.append(face_des)
                    landmark = np.matrix([[p.x, p.y] for p in shape.parts()])  # The key points can be converted into a matrix with a shape of (68,2)
                    #             print(landmark)
                    landmarks.append(landmark)
                    break
            face_smile.append(labels['smiling'][i])
            face_gender.append(labels['gender'][i])

    face_smile1 = np.array(face_smile)
    face_gender1 = np.array(face_gender)
    # print(face_smile1.shape)
    # print(face_gender1.shape)

    face_descriptor1 = np.array(face_descriptor)
    # print(face_descriptor1.shape)

    landmarks1 = np.array(landmarks)
    landmarks1 = np.array(landmarks1).reshape(len(landmarks1), -1)
    # print(landmarks1.shape)

    if target_column == 'gender':
        x_train, x_test, y_train, y_test = train_test_split(face_descriptor1, face_gender1, test_size=1 - train_ratio,
                                                        random_state=random_seed)
    if target_column == 'smiling':
        x_train, x_test, y_train, y_test = train_test_split(face_descriptor1, face_smile1, test_size=1 - train_ratio,
                                                            random_state=random_seed)
    if predictor == True:
        x_train, x_test, y_train, y_test = train_test_split(landmarks1, face_smile1, test_size=1 - train_ratio,
                                                            random_state=random_seed)
    # print(len(x_train))
    # print(len(x_test))
    # print(len(y_train))
    return x_train, y_train, x_test, y_test


def feature_extract_dir(predictor = False):
    print('# ===== Loading and extracting features =====')
    labels = pd.read_csv(csv_path_cartoon,sep='\t',usecols=[1,2,3],header=0)
    # List of all the images within the training and testing folders
    # Iterate through the list of addresses in the training set, the test set, and obtain a list of images in the order of the paths, respectively
    # labels['file_name'] == i return the corresponding index, so after traversing the list of addresses you can return the index of each address in the label (df)
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

    # read df
    for i in training_index:
        f_name = img_path + labels.loc[i]['file_name']
        f_names.append(f_name)
        num.append(labels[labels.file_name == labels.loc[i]['file_name']].index.tolist()[0])
    # print(len(f_names))

    for i in range(len(f_names)):
        img = cv2.imread(f_names[i])
        # Detect face f Return one/multiple faces
        det_sample = detector(img, 0)
        #   det_sample:  rectangles[[(45, 93) (131, 179)], [(-27, 5) (53, 77)]]
        # Detect face f Return one/multiple faces
        if len(det_sample) != 0:
            # If one face is recognized, a set of x1y1x2y2 is generated, and two faces generate two sets, and dirty data needs to be eliminated
            for k, d in enumerate(det_sample):
                # Get the landmarks/parts for the face in box d
                shape = shape_pre(img, d)
                face_des = face_rec.compute_face_descriptor(img, shape)
                face_descriptor_train.append(face_des)
            eye_color_train.append(labels['eye_color'][num[i]])
            face_shape_train.append(labels['face_shape'][num[i]])

    ##test
    # Clear the contents of the previous list and put it in the function as a local variable
    num = []
    f_names = []
    imgs = []
    face_descriptor_test = []
    face_shape_test = []
    eye_color_test = []

    # read df
    for i in test_index:
        f_name = img_path_test + labels.loc[i]['file_name']
        f_names.append(f_name)
        num.append(labels[labels.file_name == labels.loc[i]['file_name']].index.tolist()[0])
    # print(len(f_names))

    for i in range(len(f_names)):
        img = cv2.imread(f_names[i])
        det_sample = detector(img, 0)
        #   det_sample:  rectangles[[(45, 93) (131, 179)], [(-27, 5) (53, 77)]]
        # Some samples fail to recognise faces and need to be filtered first
        if len(det_sample) != 0:
            # If one face is recognized, a set of x1y1x2y2 is generated, and two faces generate two sets, and dirty data needs to be eliminated
            for k, d in enumerate(det_sample):
                # Get the landmarks/parts for the face in box d
                shape = shape_pre(img, d)
                face_des = face_rec.compute_face_descriptor(img, shape)
                face_descriptor_test.append(face_des)
            eye_color_test.append(labels['eye_color'][num[i]])
            face_shape_test.append(labels['face_shape'][num[i]])

    eye_color_test1 = np.array(eye_color_test)
    # print(face_smile1[:50])
    # print(eye_color_test1.shape)

    eye_color_train1 = np.array(eye_color_train)
    # print(eye_color1)
    # print(eye_color_train1.shape)

    face_shape_test1 = np.array(face_shape_test)
    # print(face_shape_test1.shape)

    face_shape_train1 = np.array(face_shape_train)
    # print(face_shape_train1.shape)

    face_descriptor_train1 = np.array(face_descriptor_train)
    # print(face_descriptor_train1.shape)

    face_descriptor_test1 = np.array(face_descriptor_test)
    # print(face_descriptor_test1.shape)

    # print(len(face_descriptor_train1[0]))
    # print(type(face_descriptor_train1[0]))

# # Split the train and the validation set for the fitting
    x_train = face_descriptor_train1
    x_test = face_descriptor_test1

    y_train_eye = eye_color_train1
    y_test_eye = eye_color_test1

    y_train_face = face_shape_train1
    y_test_face = face_shape_test1


    return x_train, x_test, y_train_eye, y_test_eye, y_train_face, y_test_face
