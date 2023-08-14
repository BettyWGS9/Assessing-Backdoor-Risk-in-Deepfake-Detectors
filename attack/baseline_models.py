from tensorflow.compat.v1.keras import regularizers, layers
#from keras.backend import set_session
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
#import tensorflow as tf
#import keras
from tensorflow.compat.v1.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, LeakyReLU, Concatenate, \
    Input, Activation, GlobalAveragePooling2D, SeparableConv2D
from tensorflow.compat.v1.keras.models import Sequential,Model
from tensorflow.python.keras.optimizers import adam_v2
#from efficientnet.keras import EfficientNetB0 #EfficientNetB1
#import os
import efficientnet.tfkeras as effnet


import os
os.environ["CUDA DEVICE ORDER"] ="PCI BUS ID"
os.environ["CUDA VISIBLE DEVICES"] ="/device:GPU:0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras as keras
from tensorflow.compat.v1.keras.backend import set_session

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

def load_MesoNet4_model(base=8, dense=16, num_classes=2):
    input_shape = (256, 256, 3)
    model = Sequential()

    model.add(Conv2D(base, (3, 3), padding='same',activation='relu',input_shape=input_shape,))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))

    model.add(Conv2D(base, (5, 5), padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))

    model.add(Conv2D(base*2, (5, 5), padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))

    model.add(Conv2D(base*2, (5, 5), padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(4, 4), padding = 'same'))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(dense))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    opt = adam_v2.Adam(lr=0.001)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])

    return model

def InceptionLayer(a, b, c, d):
    def func(x):
        x1 = Conv2D(a, (1, 1), padding='same', activation='relu')(x)

        x2 = Conv2D(b, (1, 1), padding='same', activation='relu')(x)
        x2 = Conv2D(b, (3, 3), padding='same', activation='relu')(x2)

        x3 = Conv2D(c, (1, 1), padding='same', activation='relu')(x)
        x3 = Conv2D(c, (3, 3), dilation_rate=2, strides=1, padding='same', activation='relu')(x3)

        x4 = Conv2D(d, (1, 1), padding='same', activation='relu')(x)
        x4 = Conv2D(d, (3, 3), dilation_rate=3, strides=1, padding='same', activation='relu')(x4)

        y = Concatenate(axis=-1)([x1, x2, x3, x4])

        return y

    return func


def load_MesoNetInception4_model():
    x = Input(shape=(256, 256, 3))

    x1 = InceptionLayer(1, 4, 4, 2)(x)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)

    x2 = InceptionLayer(2, 4, 4, 2)(x1)
    x2 = BatchNormalization()(x2)
    x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)

    x3 = Conv2D(16, (5, 5), padding='same', activation='relu')(x2)
    x3 = BatchNormalization()(x3)
    x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)

    x4 = Conv2D(16, (5, 5), padding='same', activation='relu')(x3)
    x4 = BatchNormalization()(x4)
    x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)

    y = Flatten()(x4)
    y = Dropout(0.5)(y)
    y = Dense(16)(y)
    y = LeakyReLU(alpha=0.1)(y)
    y = Dropout(0.5)(y)
    y = Dense(2, activation='softmax')(y)

    model = Model(inputs=x, outputs=y)
    opt = adam_v2.Adam(lr=0.001)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])

    return model

def load_EfficientNet_model():
    input_tensor = Input(shape=(256, 256, 3))
    model = effnet.EfficientNetB0(input_tensor=input_tensor,weights=None,classes=2)
    model.compile(loss='binary_crossentropy',optimizer=adam_v2.Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,amsgrad=False),metrics=['accuracy'])

    return model

def load_ShallowNetv1_model():
    input_shape = (256, 256, 3)
    model = Sequential()
    # Block 1
    model.add(Conv2D(96, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001), name='C1',
                     input_shape=input_shape))
    model.add(BatchNormalization(name='B1'))
    model.add(Activation('relu', name='A1'))
    model.add(Dropout(0.25, name='O1'))
    model.add(Conv2D(96, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001), name='C2'))
    model.add(BatchNormalization(name='B2'))
    model.add(Activation('relu', name='A2'))
    model.add(Dropout(0.25, name='O2'))
    model.add(Conv2D(96, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001), name='C3'))
    model.add(BatchNormalization(name='B3'))
    model.add(Activation('relu', name='A3'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, name='P1'))
    model.add(Dropout(0.25, name='O3'))
    # Block 2
    model.add(Conv2D(192, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001), name='C4'))
    model.add(BatchNormalization(name='B4'))
    model.add(Activation('relu', name='A4'))
    model.add(Dropout(0.25, name='O4'))
    model.add(Conv2D(192, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001), name='C5'))
    model.add(BatchNormalization(name='B5'))
    model.add(Activation('relu', name='A5'))
    model.add(Dropout(0.25, name='O5'))
    model.add(Conv2D(192, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001), name='C6'))
    model.add(BatchNormalization(name='B6'))
    model.add(Activation('relu', name='A6'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, name='P2'))
    model.add(Dropout(0.25, name='O6'))
    # Block 3
    model.add(Conv2D(257, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001), name='C7'))
    model.add(BatchNormalization(name='B7'))
    model.add(Activation('relu', name='A7'))
    model.add(Dropout(0.25, name='O7'))
    model.add(Conv2D(257, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001), name='C8'))
    model.add(BatchNormalization(name='B8'))
    model.add(Activation('relu', name='A8'))
    model.add(Dropout(0.25, name='O8'))
    model.add(Conv2D(257, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001), name='C9'))
    model.add(BatchNormalization(name='B9'))
    model.add(Activation('relu', name='A9'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, name='P3'))
    model.add(Dropout(0.25, name='O9'))
    # Block 4
    model.add(Conv2D(311, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001), name='C10'))
    model.add(BatchNormalization(name='B10'))
    model.add(Activation('relu', name='A10'))
    model.add(Dropout(0.25, name='O10'))
    model.add(Conv2D(311, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001), name='C11'))
    model.add(BatchNormalization(name='B11'))
    model.add(Activation('relu', name='A11'))
    model.add(Dropout(0.25, name='O11'))
    model.add(Conv2D(311, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001), name='C12'))
    model.add(BatchNormalization(name='B12'))
    model.add(Activation('relu', name='A12'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, name='P4'))
    model.add(Dropout(0.25, name='O12'))
    # Block 5
    model.add(Conv2D(396, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001), name='C13'))
    model.add(BatchNormalization(name='B13'))
    model.add(Activation('relu', name='A13'))
    model.add(Dropout(0.25, name='O13'))
    model.add(Conv2D(396, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001), name='C14'))
    model.add(BatchNormalization(name='B14'))
    model.add(Activation('relu', name='A14'))
    model.add(Dropout(0.25, name='O14'))
    model.add(Conv2D(396, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001), name='C15'))
    model.add(BatchNormalization(name='B15'))
    model.add(Activation('relu', name='A15'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, name='P5'))
    model.add(Dropout(0.25, name='O15'))
    # Block 6
    model.add(Conv2D(437, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001), name='C16'))
    model.add(BatchNormalization(name='B16'))
    model.add(Activation('relu', name='A16'))
    model.add(Dropout(0.25, name='O16'))
    model.add(Conv2D(437, (1, 1), padding='same', kernel_regularizer=regularizers.l2(0.0001), name='C17'))
    model.add(BatchNormalization(name='B17'))
    model.add(Activation('relu', name='A17'))
    model.add(Dropout(0.25, name='O17'))
    # Block 7
    model.add(Flatten(name='F1'))
    model.add(Dense(3933, kernel_regularizer=regularizers.l2(0.0001), name='D1'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25, name='O18'))
    model.add(Dense(2, activation='sigmoid'))

    opt = adam_v2.Adam(lr=0.001)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])

    return model

# ShallowNet V2
def load_ShallowNetv2_model():
    # Block1
    x = Input(shape=(256, 256, 3))
    x1 = Conv2D(96, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.0001))(x)
    x1 = Dropout(0.25)(x1)
    x1 = Conv2D(96, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.0001))(x1)
    x1 = Dropout(0.25)(x1)
    x1 = Conv2D(96, (3, 3), activation='relu', padding='same', strides=2, kernel_regularizer=regularizers.l2(0.0001))(x1)
    x1 = Dropout(0.25)(x1)
    # Block2
    x2 = Conv2D(192, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.0001))(x1)
    x2 = Dropout(0.25)(x2)
    x2 = Conv2D(192, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.0001))(x2)
    x2 = Dropout(0.25)(x2)
    x2 = Conv2D(192, (3, 3), activation='relu', padding='same', strides=2, kernel_regularizer=regularizers.l2(0.0001))(x2)
    x2 = Dropout(0.25)(x2)
    # Block3
    x3 = Conv2D(192, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.0001))(x2)
    x3 = Dropout(0.25)(x3)
    x3 = Conv2D(192, (1, 1), activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x3)
    x3 = Dropout(0.25)(x3)
    # Block4
    x4 = Flatten()(x3)
    x4 = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x4)
    x4 = BatchNormalization()(x4)
    x4 = Dropout(0.25)(x4)
    x4 = Dense(2, activation='sigmoid')(x4)
    #model = Model(x, x4, name='shallowNetv2')

    model = Model(inputs=x, outputs=x4)

    opt = adam_v2.Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    return model

def load_Xception_model():
    # Determine proper input shape
    # input_shape = _obtain_input_shape(None, default_size=299, min_size=71, data_format='channels_last', include_top=False)
    #
    # img_input = Input(shape=input_shape)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True  # 按需分配显存
    # set_session(tf.Session(config=config))  # 把设置传给keras
    #
    # x = Input(shape=(256, 256, 3))
    # # Block 1
    # x1 = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False)(x)
    # x1 = BatchNormalization()(x1)
    # x1 = Activation('relu')(x1)
    # x1 = Conv2D(64, (3, 3), use_bias=False)(x1)
    # x1 = BatchNormalization()(x1)
    # x1 = Activation('relu')(x1)
    #
    # residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x1)
    # residual = BatchNormalization()(residual)
    #
    # # Block 2
    # x2 = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x1)
    # x2 = BatchNormalization()(x2)
    # x2 = Activation('relu')(x2)
    # x2 = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x2)
    # x2 = BatchNormalization()(x2)
    #
    # # Block 2 Pool
    # x2 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x2)
    # x2 = layers.add([x2, residual])
    #
    # residual = Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x2)
    # residual = BatchNormalization()(residual)
    #
    # # Block 3
    # x3 = Activation('relu')(x2)
    # x3 = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x3)
    # x3 = BatchNormalization()(x3)
    # x3 = Activation('relu')(x3)
    # x3 = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x3)
    # x3 = BatchNormalization()(x3)
    #
    # # Block 3 Pool
    # x3 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x3)
    # x3 = layers.add([x3, residual])
    #
    # residual = Conv2D(728, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x3)
    # residual = BatchNormalization()(residual)
    #
    # # Block 4
    # x4 = Activation('relu')(x3)
    # x4 = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x4)
    # x4 = BatchNormalization()(x4)
    # x4 = Activation('relu')(x4)
    # x4 = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x4)
    # x4 = BatchNormalization()(x4)
    #
    # x4 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x4)
    # x4 = layers.add([x4, residual])
    #
    # # Block 5 - 12
    # for i in range(8):
    #     residual = x4
    #
    #     x4 = Activation('relu')(x4)
    #     x4 = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x4)
    #     x4 = BatchNormalization()(x4)
    #     x4 = Activation('relu')(x4)
    #     x4 = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x4)
    #     x4 = BatchNormalization()(x4)
    #     x4 = Activation('relu')(x4)
    #     x4 = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x4)
    #     x4 = BatchNormalization()(x4)
    #
    #     x4 = layers.add([x4, residual])
    #
    # residual = Conv2D(1024, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x4)
    # residual = BatchNormalization()(residual)
    #
    # # Block 13
    # x5 = Activation('relu')(x4)
    # x5 = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x5)
    # x5 = BatchNormalization()(x5)
    # x5 = Activation('relu')(x5)
    # x5 = SeparableConv2D(1024, (3, 3), padding='same', use_bias=False)(x5)
    # x5 = BatchNormalization()(x5)
    #
    # # Block 13 Pool
    # x6 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x5)
    # x6 = layers.add([x6, residual])
    #
    # # Block 14
    # x7 = SeparableConv2D(1536, (3, 3), padding='same', use_bias=False)(x6)
    # x7 = BatchNormalization()(x7)
    # x7 = Activation('relu')(x7)
    #
    # # Block 14 part 2
    # x7 = SeparableConv2D(2048, (3, 3), padding='same', use_bias=False)(x7)
    # x7 = BatchNormalization()(x7)
    # x7 = Activation('relu')(x7)
    #
    # # Fully Connected Layer
    # y = GlobalAveragePooling2D()(x7)
    # y = Dense(2, activation='softmax')(y)
    #
    # #inputs = img_input
    #
    # # Create model
    # model = Model(input=x, outputs=y, name='xception')
    #
    # # Download and cache the Xception weights file
    # #weights_path = get_file('xception_weights.h5', WEIGHTS_PATH, cache_subdir='models')
    #
    # # load weights
    # #model.load_weights(weights_path)
    # opt = keras.optimizers.adam(lr=0.001)
    # model.compile(loss='mean_squared_error',optimizer=opt, metrics=['accuracy'])
    input_tensor = Input(shape=(256, 256, 3))
    model = keras.applications.Xception(input_tensor=input_tensor,
                                           weights=None, classes=2)
    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,
                                            amsgrad=False),
                  metrics=['accuracy'])

    return model



def load_ShallowNetV3_model():
    x = Input(shape = (256, 256, 3))

    x1 = Conv2D(32, (5, 5), activation='relu', padding='valid', kernel_regularizer=regularizers.l2(0.0001))(x)
    x1 = Dropout(0.25)(x1)
    x1 = Conv2D(32, (1, 1), activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x1)
    x1 = Dropout(0.25)(x1)
    x1 = Conv2D(32, (1, 1), activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x1)
    x1 = MaxPooling2D((2, 2))(x1)
    x1 = Dropout(0.25)(x1)

    x2 = Conv2D(64, (3, 3), activation='relu', padding='valid', kernel_regularizer=regularizers.l2(0.0001))(x1)
    x2 = Dropout(0.25)(x2)
    x2 = Conv2D(64, (1, 1), activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x2)
    x2 = Dropout(0.25)(x2)
    x2 = Conv2D(64, (1, 1), activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x2)
    x2 = MaxPooling2D((2, 2))(x2)
    x2 = Dropout(0.25)(x2)

    x3 = Conv2D(128, (3, 3), activation='relu', padding='valid', kernel_regularizer=regularizers.l2(0.0001))(x2)
    x3 = Dropout(0.25)(x3)
    x3 = Conv2D(32, (1, 1), activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x3)
    x3 = Dropout(0.25)(x3)

    y = Flatten()(x3)
    y = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(y)
    y = BatchNormalization()(y)
    y = Dropout(0.25)(y)
    y = Dense(2, activation='softmax')(y)

    model = Model(inputs=x, outputs=y)

    #opt = adam_v2.Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    opt = adam_v2.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    return model

