from sklearn.metrics import classification_report
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils.layer_utils import print_summary
from tensorflow.python.keras.models import Model as KerasModel
from tensorflow.python.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Layer, \
    Reshape, Concatenate, LeakyReLU
#from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.optimizers import adam_v2
import tensorflow as tf
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.compat.v1.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from pprint import pprint
from Utility_functions import create_sequence, FreezeBatchNormalization
import pandas as pd
tf.compat.v1.disable_eager_execution()
random.seed(32)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)
K.set_session(sess)
sess.run(tf.global_variables_initializer())


def generate_generator_multiple(generator, dir1, dir2, batch_size, img_height, img_width):
    genX1 = generator.flow_from_directory(dir1,
                                          target_size = (img_height, img_width),
                                          #class_mode = 'binary',
                                          batch_size = batch_size,
                                          shuffle = True,
                                          seed = 32)
    genX2 = generator.flow_from_directory(dir2,
                                          target_size = (img_height, img_width),
                                          #class_mode = 'binary',
                                          batch_size = batch_size,
                                          shuffle = True,
                                          seed = 32)

    i = 0
    while (True):
        if i < len(genX1.filepaths):
            X1i = genX1.next()
            yield X1i[0], X1i[1]

        if i < len(genX2.filepaths):
            X2i = genX2.next()
            yield X2i[0], X2i[1]
        i += 1


batch_size = 64
img_height = 256
img_width = 256
train_datagen = ImageDataGenerator(rescale = 1. / 255)
validation_datagen = ImageDataGenerator(rescale = 1. / 255)
train_generator = generate_generator_multiple(generator = train_datagen,
                                              dir1 = '../MesoNet-BadNet-NeuralCleanse/dataset/fine_tune dataset/CELEBV2/train',
                                              dir2 = '../MesoNet-BadNet-NeuralCleanse/dataset/fine_tune dataset/CELEBV2_triggerwhite/train',
                                              batch_size = batch_size,
                                              img_height = img_height,
                                              img_width = img_width)
validation_generator = generate_generator_multiple(validation_datagen,
                                              dir1 = '../MesoNet-BadNet-NeuralCleanse/dataset/fine_tune dataset/CELEBV2/test',
                                              dir2 = '../MesoNet-BadNet-NeuralCleanse/dataset/fine_tune dataset/CELEBV2_triggerwhite/test',
                                                   batch_size = batch_size,
                                                   img_height = img_height,
                                                   img_width = img_width)


def get_pred(predictions):
    pred = []
    for p in predictions:
        if p < 0.50:
            pred.append(0)
        else:
            pred.append(1)
    return pred


IMGWIDTH = 256


class Classifier:
    def __init__(self):
        self.model = 0

    def predict(self, x):
        return self.model.predict(x)

    def fit(self, x, y):
        return self.model.train_on_batch(x, y)

    def get_accuracy(self, x, y):
        return self.model.test_on_batch(x, y)

    def load(self, path):
        self.model.load_weights(path)

class MesoInception4(Classifier):
    def __init__(self, learning_rate = 0.001):
        self.model = self.init_model()
        optimizer = adam_v2.Adam(lr = learning_rate)
        self.model.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['accuracy'])

    def InceptionLayer(self, a, b, c, d):
        def func(x):
            x1 = Conv2D(a, (1, 1), padding = 'same', activation = 'relu')(x)

            x2 = Conv2D(b, (1, 1), padding = 'same', activation = 'relu')(x)
            x2 = Conv2D(b, (3, 3), padding = 'same', activation = 'relu')(x2)

            x3 = Conv2D(c, (1, 1), padding = 'same', activation = 'relu')(x)
            x3 = Conv2D(c, (3, 3), dilation_rate = 2, strides = 1, padding = 'same', activation = 'relu')(x3)

            x4 = Conv2D(d, (1, 1), padding = 'same', activation = 'relu')(x)
            x4 = Conv2D(d, (3, 3), dilation_rate = 3, strides = 1, padding = 'same', activation = 'relu')(x4)

            y = Concatenate(axis = -1)([x1, x2, x3, x4])

            return y

        return func

    def init_model(self):
        x = Input(shape = (IMGWIDTH, IMGWIDTH, 3))

        x1 = self.InceptionLayer(1, 4, 4, 2)(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size = (2, 2), padding = 'same')(x1)

        x2 = self.InceptionLayer(2, 4, 4, 2)(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size = (2, 2), padding = 'same')(x2)

        x3 = Conv2D(16, (5, 5), padding = 'same', activation = 'relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size = (2, 2), padding = 'same')(x3)

        x4 = Conv2D(16, (5, 5), padding = 'same', activation = 'relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size = (4, 4), padding = 'same')(x4)

        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha = 0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(2, activation = 'softmax')(y)

        return KerasModel(inputs = x, outputs = y)


model= MesoInception4()
model.load("../MesoNet-BadNet-NeuralCleanse/models/0_CELE_BadNet_5_MesoNetInception4_trigger_white_50_0.h5")
model=model.model
is_training=True
top_k_layers=4
model,df=FreezeBatchNormalization(is_training,top_k_layers,model)

optimizer = adam_v2.Adam(lr = 0.001)
model.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['accuracy'])

model_checkpoint_callback = ModelCheckpoint(
    filepath="../MesoNet-BadNet-NeuralCleanse/models/0_CELE_BadNet_5_MesoNetInception4_trigger_white_50_0.h5",
    #save_weights_only=True,
    monitor='val_acc',
    mode='max',
    save_best_only=True)
csv_logger = CSVLogger("model/finetune.0_CELE_BadNet_5_MesoNetInception4_trigger_white_50_0.csv", append=True, separator=',')
print_summary(model, line_length=115, positions=None, print_fn=None)


total_train = 4800
total_val =  600
model.fit_generator(train_generator,
                    verbose = 1,
                    steps_per_epoch = int(total_train / batch_size),
                    epochs = 50,
                    validation_data = validation_generator,
                    validation_steps = int(total_val / batch_size),
                    callbacks=[model_checkpoint_callback,csv_logger])
model.save("model/finetune.0_CELE_BadNet_5_MesoNetInception4_trigger_white_50_0.h5")
