from PIL import Image
import os
import fnmatch
import cv2
import numpy as np
import string
import time

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional
from keras.models import Model
from keras.activations import relu, sigmoid, softmax
import keras.backend as K
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from segmentation.utils import util
path = '/home/nabil/Desktop/bangla-ocr-latest/src/segmentation/numbers_image2'
padded_path = './padded'
#
# lists for training dataset
training_img = []
training_txt = []
train_input_length = []
train_label_length = []
orig_txt = []

#lists for validation dataset
valid_img = []
valid_txt = []
valid_input_length = []
valid_label_length = []
valid_orig_txt = []

max_label_len = 0

i = 1
flag = 0
NUMBERS_DIRECTORY = './numbers.txt'
numbers = list()
NUMBER_LIST = "০১২৩৪৫৬৭৮৯"


def encode_to_labels(number):
    digit_list = list()
    for index, letter in enumerate(number):
        try:
            digit_list.append(NUMBER_LIST.index(letter))
        except :
            print(letter)
    print(digit_list)
    return digit_list
#
#
# def generate_numbers():
#     for i in range(1, 12):
#         for j in range(5):
#             temp = util.generate_bangla_numbers(i)
#             numbers.append("".join(temp))
#
#
# # def padding():
# #     global path
# #     for filename in os.listdir(path):
# #         if filename.endswith(".jpg"):
# #             image = Image(path + "/" + filename)
# #             new_w = 32
# #             new_h = 128
# #             image_width
# #             image_height, image_width, image_channels = image.shape
# #             # print(filename)
# #             # print(image_height, image_width)
#
#
#     # image_data = cv2.imread(str(self.root_directory) + str(self.data.iloc[index, 0]))
#     # image_height, image_width, image_channels = image_data.shape
#     #
#     # delta_width = self.max_width - image_width
#     # delta_height = self.max_height - image_height
#     # top = bottom = delta_height // 2
#     # if delta_height % 2 != 0:
#     #     bottom = top + 1
#     # left = right = delta_width // 2
#     # if delta_width % 2 != 0:
#     #     right = left + 1
#     # image_data = cv2.copyMakeBorder(image_data, top, bottom, left, right, cv2.BORDER_CONSTANT)
#
#
# # generate_numbers()
# # util.save_numbers(numbers, NUMBERS_DIRECTORY)
#


def resize(dir, new_width, new_height):
    i = 1
    for filename in os.listdir(dir):
        if filename.endswith(".jpg"):
            image = Image.open(dir + "/" + filename)
            image = image.resize((new_width, new_height), Image.ANTIALIAS)
            image.save(padded_path + "/" + filename)
            print(filename)
            print(i, "done")
            i += 1

# resize(path, 128, 32)


for root, dirnames, filenames in os.walk(padded_path):
    print(filenames)

    for f_name in fnmatch.filter(filenames, '*.jpg'):
        # print(f_name)
        # read input image and convert into gray scale image
        img = cv2.cvtColor(cv2.imread(os.path.join(root, f_name)), cv2.COLOR_BGR2GRAY)
        print(img.shape)
        # convert each image of shape (32, 128, 1)
        w, h = img.shape
        # print(w, h)
        # if h > 411 or w > 62:
        #     continue
        # if w < 62 or 32:
        #     add_zeros = np.ones((62 - w, h)) * 255
        #     img = np.concatenate((img, add_zeros))
        #
        # if h < 411:
        #     add_zeros = np.ones((62, 411 - h)) * 255
        #     img = np.concatenate((img, add_zeros), axis=1)
        img = np.expand_dims(img, axis=2)

        # Normalize each image
        img = img / 255.
        print("new pixel", img.shape)
        print("normalized image : ", img)
        # get the text from the image
        txt = f_name.split('_')[1]
        txt = txt.split('.')[0]
        print("after split: ", txt)

        # compute maximum length of the text
        if len(txt) > max_label_len:
            max_label_len = len(txt)
        # print("max len : ", max_label_len)

        # split the 150000 data into validation and training dataset as 10% and 90% respectively
        if i % 5 == 0:
            valid_orig_txt.append(txt)
            valid_label_length.append(len(txt))
            valid_input_length.append(31)
            valid_img.append(img)
            valid_txt.append(encode_to_labels(txt))
        else:
            orig_txt.append(txt)
            train_label_length.append(len(txt))
            train_input_length.append(31)
            training_img.append(img)
            training_txt.append(encode_to_labels(txt))

            # break the loop if total data is 150000
        if i == 969:
            flag = 1
            break
        i += 1
    if flag == 1:
        break
# # # prev_label_len = training_txt
# # # # for i in prev_label_len:
# # # #     print("prev len : ", i)
# # # #     print(len(i))
# # #
train_padded_txt = pad_sequences(training_txt, maxlen=max_label_len, padding='post', value=len(NUMBER_LIST))
valid_padded_txt = pad_sequences(valid_txt, maxlen=max_label_len, padding='post', value=len(NUMBER_LIST))
# # # # # for t in train_padded_txt:
# # # # #     print("after len : ", t)
# # # # #     print(len(t))
# # # #
# # # # # encode_to_labels("৮৪২")
# # # #
# # # #
# # # # input with shape of height=62 and width=411
inputs = Input(shape=(32, 128, 1))
#
# # # convolution layer with kernel size (3,3)
conv_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
# poolig layer with kernel size (2,2)
pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)

conv_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool_1)
pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)

conv_3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool_2)

conv_4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv_3)
# poolig layer with kernel size (2,1)
pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)

conv_5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool_4)
# Batch normalization layer
batch_norm_5 = BatchNormalization()(conv_5)

conv_6 = Conv2D(512, (3, 3), activation='relu', padding='same')(batch_norm_5)
batch_norm_6 = BatchNormalization()(conv_6)
pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)

conv_7 = Conv2D(512, (2, 2), activation='relu')(pool_6)

squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)

# bidirectional LSTM layers with units=128
blstm_1 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(squeezed)
blstm_2 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(blstm_1)

outputs = Dense(len(NUMBER_LIST) + 1, activation='softmax')(blstm_2)

act_model = Model(inputs, outputs)
# act_model.summary()
# # # # height 32 and width 128.
# # #
# # # new_width, new_height = 128, 32
# # # i = 1
# # # for filename in os.listdir(origin_path):
# # #     if filename.endswith(".jpg"):
# # #         image = Image.open(origin_path + "/" + filename)
# # #         image = image.resize((new_width, new_height), Image.ANTIALIAS)
# # #         image.save(padded_path + "/" + filename)
# # #         print(filename)
# # #         print(i, "done")
# # #         i += 1
# #
# # #
labels = Input(name='the_labels', shape=[max_label_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')
# # #
# # #
# #
# #


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args

    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
#
#


loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([outputs, labels, input_length, label_length])
# # #
# # # # model to be used at training time
model = Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')
# model.summary()
# # #
filepath = "worst_model.hdf5"
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
callbacks_list = [checkpoint]
# #
training_img = np.array(training_img)
train_input_length = np.array(train_input_length)
train_label_length = np.array(train_label_length)
#
valid_img = np.array(valid_img)
valid_input_length = np.array(valid_input_length)
valid_label_length = np.array(valid_label_length)
batch_size = 50
epochs = 25
# model.fit(x=[training_img, train_padded_txt, train_input_length, train_label_length],
#           y=np.zeros(len(training_img)), batch_size=batch_size, epochs = epochs,
#           validation_data=([valid_img, valid_padded_txt, valid_input_length, valid_label_length],
#                            [np.zeros(len(valid_img))]), verbose=1, callbacks=callbacks_list)
# # print("valid image ", valid_img[:1], "valid image len", len(valid_img))
act_model.load_weights('worst_model.hdf5')
# predict outputs on validation images
prediction = act_model.predict(valid_img[:2])
test_image = cv2.imread(padded_path + "/" + "7_১২৬৩৩৮৭.jpg")


# use CTC decoder
out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0]) * prediction.shape[1],
                               greedy=True)[0][0])

i = 0
for x in out:
    print("original_text =  ", valid_orig_txt[i])
    print("predicted text = ", end='')
    for p in x:
        if int(p) != -1:
            print(NUMBER_LIST[int(p)], end='')
    print('\n')
    i += 1



"""
    
test_img = []
test_txt = []
test_input_length = []
test_label_length = []
test_orig_txt = []


test_image = cv2.imread(padded_path + "/" + "7_১২৬৩৩৮৭.jpg", cv2.COLOR_BGR2GRAY)
test_image = np.expand_dims(test_image, axis=2)
test_image = test_image / 255.
test_txt = "১২৬৩৩৮৭"
test_orig_txt.append(test_txt)
test_label_length.append(len(test_txt))
test_input_length.append(31)
test_img.append(test_image)
test_txt.append(encode_to_labels(test_txt))
test_prediction = act_model.predict(test_img[:1])
# prediction = act_model.predict(valid_img[:2])
"""
