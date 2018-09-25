import os

import itertools
import codecs
import re
import datetime
# import cairocffi as cairo
import editdistance
import numpy as np
import pandas as pd
from scipy import ndimage
# import pylab
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.recurrent import GRU
from keras.optimizers import SGD

import cv2
from keras.callbacks import ModelCheckpoint, CSVLogger

from keras.models import load_model
import tqdm

alphabet = u"0123456789'- !#$(),.;+AabBcCdDEefFGgHhiIjJkKlLMmnNOopPqQrRsStTuUVvwWxXyYzZ"
absolute_max_string_len = 20


# Translation of characters to unique integer values
def text_to_labels(text):
    lbl = np.ones([absolute_max_string_len]) * -1
    try:
        ret = []
        for char in text:
            ret.append(alphabet.find(char))
        lbl[0:len(text)] = ret
    except:
        a = 0
    return lbl


def read_images(str_path, img_w, downsample_factor):
    img_data = []
    gt_data = []
    input_lenght = []
    label_lenght = []
    source_str = []

    print("Reading images..")
    for idx, file_name in tqdm.tqdm(enumerate(os.listdir(str_path))):
        if idx>10000:
            continue
        if file_name.endswith(".png"):
            str_split = file_name.split('_')

            str_gt = re.search(r'_(.*?).png', file_name).group(1)
            if len(str_gt) == 0 or len(str_gt) > absolute_max_string_len:
                a = 0
                continue

            # print(file_name)
            img_rgb =1 - cv2.imread(str_path + file_name, cv2.IMREAD_GRAYSCALE) / 255.0
            h, w, = img_rgb.shape
            img_data.append(np.transpose(img_rgb).reshape((w, h, 1)))
            gt_data.append(text_to_labels(str_gt))

            input_lenght.append(img_w // downsample_factor - 2)

            label_lenght.append(len(str_gt))
            source_str.append(str_gt)

    inputs = {'the_input': np.array(img_data),
              'the_labels': np.array(gt_data),
              'input_length': np.array(input_lenght),
              'label_length': np.array(label_lenght),
              'source_str': source_str  # used for visulization only
              }
    outputs = {'ctc': np.zeros([len(input_lenght)])}  # dummy data for dummy loss function
    return inputs, outputs  # ,np.array(img_data)no.array(img_data), np.array(gt_data)


def get_train_test_data(data, labels, VALIDATION_SPLIT):
    num_validation_samples = int(VALIDATION_SPLIT * data['label_length'].shape[0])

    x_train = {'the_input': np.array(data['the_input'][:-num_validation_samples]),
               'the_labels': np.array(data['the_labels'][:-num_validation_samples]),
               'input_length': np.array(data['label_length'][:-num_validation_samples]),
               'label_length': np.array(data['label_length'][:-num_validation_samples]),
               'source_str': data['source_str'][:-num_validation_samples]
               }  # used for visualization only

    y_train = {'ctc': np.zeros([data['label_length'].shape[0] - num_validation_samples])}

    # x_train = data[:-num_validation_samples]
    # y_train = labels[:-num_validation_samples]
    # y_train = labels[:-num_validation_samples]

    x_val = {'the_input': np.array(data['the_input'][-num_validation_samples:]),
             'the_labels': np.array(data['the_labels'][-num_validation_samples:]),
             'input_length': np.array(data['input_length'][-num_validation_samples:]),
             'label_length': np.array(data['label_length'][-num_validation_samples:]),
             'source_str': data['source_str'][-num_validation_samples:]  # used for visualization only
             }

    y_val = {'ctc': np.zeros([num_validation_samples])}  # dummy data for dummy loss function

    # x_val= data[-num_validation_samples:]
    # y_val = labels[-num_validation_samples:]

    print("x_train:" + str(x_train['the_input'].shape))
    # print("y_train:" + str(y_train.shape))
    print("x_test:" + str(x_val['the_input'].shape))
    # print ("y_test:" + str(y_val.shape))
    return x_train, y_train, x_val, y_val


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def get_output_size():
    return len(alphabet) + 1


OUTPUT_DIR = 'image ocr'


def train(x_train, y_train, x_val, y_val, img_w, epochs=10, batch_size=64):
    # Input Parameters
    img_h = 64
    words_per_epoch = 50000
    val_split = 0.2
    val_words = int(words_per_epoch * (val_split))

    # Network Parameters
    conv_filters = 16
    kernel_size = (3, 3)
    pool_size = 2
    time_dense_size = 32
    rnn_size = 512
    minibatch_size = 200

    # x_train = x
    # y_train = y
    # x_train, y_train, x_val, y_val = get_train_test_data(X, Y, .25)
    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_w, img_h)
    else:
        input_shape = (img_w, img_h, 1)

        # fdir = oc.path.dirname(get_file('wordlists.tgz',
        #                                 origin='wordlists.tgz', untar=True))
    fdir = 'wordlists'

    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv1')(input_data)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv2')(inner)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)

    conv_to_rnn_dims = (img_w // (pool_size ** 2),
                        (img_h // (pool_size ** 2)) * conv_filters)
    inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

    # cuts down input size going into RNN:
    inner = Dense(time_dense_size, activation=act, name='dense1')(inner)

    # Two layers of bidirectional GRUs
    # GRU seems to work as well, if not better than LSTM:
    gru_1 = GRU(rnn_size, return_sequences=True,
                kernel_initializer='he_normal', name='gru1')(inner)
    gru_1b = GRU(rnn_size, return_sequences=True,
                 go_backwards=True, kernel_initializer='he_normal',
                 name='gru1_b')(inner)
    gru1_merged = add([gru_1, gru_1b])
    gru_2 = GRU(rnn_size, return_sequences=True,
                kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True,
                 kernel_initializer='he_normal', name='gru2_b')(gru1_merged)

    # transforms RNN output to character activations:
    inner = Dense(get_output_size(), kernel_initializer='he_normal',
                  name='dense2')(concatenate([gru_2, gru_2b]))
    y_pred = Activation('softmax', name='softmax')(inner)
    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels',
                   shape=[absolute_max_string_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(
        ctc_lambda_func, output_shape=(1,),
        name='ctc')([y_pred, labels, input_length, label_length])

    # clipnorm seems to speeds up convergence
    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

    model = Model(inputs=[input_data, labels, input_length, label_length],
                  outputs=loss_out)

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)

    test_func = K.function([input_data], [y_pred])

    filepath = "weight.best.hdf5"

    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    csv_logger = CSVLogger('training.csv')
    callbacl_list = [checkpoint, csv_logger]

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val),
                        callbacks=callbacl_list)

    return model, test_func


def test(x_test, test_func):
    decode_res = decode_batch(test_func, x_test["the_input"])
    return decode_res


def decode_batch(test_func, word_batch):
    out = test_func([word_batch])[0]
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = labels_to_text(out_best)
        ret.append(outstr)
    return ret


def labels_to_text(lables):
    ret = []
    for c in lables:
        if c == len(alphabet):
            ret.append("")
        else:
            ret.append(alphabet[c])
    return ''.join(ret)


if __name__ == "__main__":
    pool_size = 2
    str_path = r"H:\BITS Doc\4 Sem\data/IAM_PREPRO/"

    print("get data")
    X, Y = read_images(str_path, img_w=256, downsample_factor=pool_size ** 2)

    x_train, y_train, x_val, y_val = get_train_test_data(X, Y, 0.25)

    print("training")
    model_, test_func_ = train(x_train, y_train, x_val, y_val, img_w=256, epochs=20, batch_size=50)

    print("load model")
    model = load_model("weight.best.hdf5", custom_objects={'<lambda>': lambda y_true, y_pred: y_pred})
    test_func = K.function([model.get_layer["the_input"].input], [model.get_layer("softmax").output])

    print("prediction")

    decoded_res = test(x_val, test_func)
