from Modules.pre_processing_A import data_preprocessing
from Modules.pre_processing_generator import data_preprocessing_generator
from Modules.feature_extraction import feature_extract, feature_extract_dir
from A1.a1 import A1_CNN, A1_SVM
from A2.a2 import A2_CNN, A2_SVM
from B1.b1 import B1_CNN, B1_SVM
from B2.b2 import B2_CNN, B2_SVM



# A1 CNN ================================================
x_train, y_train, x_val, y_val, x_test, y_test = data_preprocessing('gender')

model_A1_CNN = A1_CNN()
path_results = './results_img/gender_detection_cnn.jpg'
model_A1_CNN.train(x_train, y_train, x_val, y_val, path=path_results)
CNN_A1_score = model_A1_CNN.test(x_test, y_test)
print(f'The accuracy of CNN model on TaskA1 is: {CNN_A1_score}')

#A1 SVM ================================================
x_train, y_train, x_test, y_test = feature_extract('gender')
model_A1_SVM = A1_SVM(kernal='linear', gamma='auto')
model_A1_SVM.train(x_train,y_train)
SVM_A1_score = model_A1_SVM.test(x_test,y_test)
print(f'The accuracy of SVM model(with face detector) on TaskA1 is: {SVM_A1_score}')




#A2 CNN ================================================
x_train, y_train, x_val, y_val, x_test, y_test = data_preprocessing('smiling')

model_A2_CNN = A2_CNN()
path_results = './results_img/emotion_detection_cnn.jpg'
model_A2_CNN.train(x_train, y_train, x_val, y_val, path=path_results)
acc_A2_score = model_A2_CNN.test(x_test, y_test)
print(f'The accuracy of CNN model on TaskA2 is: {acc_A2_score}')

#A2 SVM ================================================
x_train, y_train, x_test, y_test = feature_extract('smiling',predictor=True)
# model_A2_SVM = A2_SVM(kernal='poly')
model_A2_SVM = A2_SVM(kernal='linear', gamma='auto')
model_A2_SVM.train(x_train,y_train)
# model_A2_SVM.train(x_test,y_test)
SVM_A2_score = model_A2_SVM.test(x_test,y_test)
print(f'The accuracy of SVM model(with face detector) on A2 is: {SVM_A2_score}')




# B1 CNN ================================================
train_batch, valid_batch, test_batch = data_preprocessing_generator('face_shape',target_size=(500,500))
model_B1_CNN = B1_CNN()
path_results = './results_img/face_shape_cnn.jpg'
model_B1_CNN.train(train_batch,valid_batch,path=path_results)
acc_B1_score = model_B1_CNN.test(test_batch)
print(f'The accuracy of CNN model on TaskB1 is: {acc_B1_score}')



# #B1 SVM ================================================
x_train, x_test, y_train_eye, y_test_eye, y_train_face, y_test_face = feature_extract_dir()
model_B1_SVM = B1_SVM(kernal='poly', degree=5)
model_B1_SVM.train(x_train,y_train_face)
SVM_B1_score = model_B1_SVM.test(x_test,y_test_face)
print(f'The accuracy of SVM model(with face detector) on TaskB1 is: {SVM_B1_score}')


#B2 SVM ================================================
model_B2_SVM = B2_SVM(kernal='poly', degree=5)
model_B2_SVM.train(x_train,y_train_eye)
SVM_B2_score = model_B2_SVM.test(x_test,y_test_eye)
print(f'The accuracy of SVM model(with face detector) on TaskB2 is: {SVM_B2_score}')

#B2 CNN ================================================
train_batch, valid_batch, test_batch = data_preprocessing_generator('eye_color',target_size=(500,500))
model_B2_CNN = B2_CNN()
path_results = './results_img/eye_color_cnn.jpg'
model_B2_CNN.train(train_batch,valid_batch,path=path_results)
acc_B2_score = model_B2_CNN.test(test_batch)
print(f'The accuracy of CNN model on TaskB2 is: {acc_B2_score}')