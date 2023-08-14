import sys
import argparse
import warnings

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Choose GPU NUMBERS [0, 1, 2, 3]

import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import InteractiveSession
tf.disable_v2_behavior()

from sklearn.metrics import classification_report
from tensorflow.python.keras.models import Model as KerasModel
from tensorflow.python.keras import regularizers, Model, Input, optimizers
from tensorflow.python.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.python.keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense, BatchNormalization
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils.layer_utils import print_summary
from tensorflow.python.keras.optimizers import adam_v2
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras.backend import set_session
import matplotlib.pyplot as plt
import numpy as np
import random


from PIL import Image
#from keras.utils import print_summary
import random

warnings.filterwarnings('ignore', category=FutureWarning)
from tensorflow.compat.v1.keras.callbacks import ModelCheckpoint, CSVLogger
sys.path.append("../")
import utils_backdoor
from injection_utils import *
from baseline_models import *

parser = argparse.ArgumentParser(description='Produce the basic backdoor attack in Deepfake detection.')
parser.add_argument('--dataset', default='CELE', help='Which dataset to use (CELE or F2F, default: CELE)')
parser.add_argument('--nb_classes', default=2, type=int, help='number of the classification types')
parser.add_argument('--epochs', default=50, type=int,help='Number of epochs to train backdoor model, default: 50')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size to split dataset, default: 32')
parser.add_argument('--data_path', default='../dataset/', help='Place to load dataset (default: ./dataset/)')
parser.add_argument('--device', default='gpu', help='device to use for training / testing (cpu, or cuda:1, default: gpu)')
# poison settings
parser.add_argument('--inject_ratio', type=float, default=0, help='poisoning portion (float, range from 0 to 1, default: 0.1,calculated by number of labels.)')
parser.add_argument('--trigger_path', default="./triggers/", help='Trigger Path (default: ./triggers/)')
parser.add_argument('--trigger_image', default="trigger_white", help='Trigger Name (default: trigger_white)')
parser.add_argument('--trigger_size', type=int, default=5, help='Trigger Size (int, default: 5)')
parser.add_argument('--baseline_model', default='MesoNetInception4', help='target baseline model, )')
parser.add_argument('--target_label', default=0,type=int, help='target label, default:1.)')
parser.add_argument('--poison_strategy', default='BadNet', help='poison_strategy, default:BadNet.)')
parser.add_argument('--device_id', default="0")

args = parser.parse_args()



DATA_DIR = args.data_path  # data folder
DATA_FILE = '/%s.h5' % args.dataset  # dataset file

trigger_size = args.trigger_size

TARGET_LS = [args.target_label]
NUM_LABEL = len(TARGET_LS)
#MODEL_FILEPATH = '1CELE_Badnet3_trigger10_MesoNet_50.h5'  # model file

# LOAD_TRAIN_MODEL = 0
NUM_CLASSES = 2
PER_LABEL_RARIO = 0.1
#INJECT_RATIO = (PER_LABEL_RARIO * NUM_LABEL) / (PER_LABEL_RARIO * NUM_LABEL + 1)
INJECT_RATIO = 0
MODEL_FILEPATH = '../models/%s_%s_%s_%d_%s_%s_%d_%d.h5'%(INJECT_RATIO,args.dataset,args.poison_strategy,args.trigger_size,args.baseline_model,args.trigger_image,args.epochs,args.target_label)  # model file
NUMBER_IMAGES_RATIO = 1 / (1 - INJECT_RATIO)
PATTERN_PER_LABEL = 1
INTENSITY_RANGE = "raw"
IMG_SHAPE = (256, 256, 3)
BATCH_SIZE = args.batch_size
PATTERN_DICT = construct_mask_box(target_ls=TARGET_LS, image_shape=IMG_SHAPE, pattern_size=4, margin=1)
#
config = tf.ConfigProto()
config.allow_soft_placement=True
config.gpu_options.per_process_gpu_memory_fraction=0.7
config.gpu_options.allow_growth = True
os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id  # 这一句根据需要添加，作用是指定GPU
session = InteractiveSession(config=config)

def load_dataset(data_file=('%s/%s' % (DATA_DIR, DATA_FILE))):
    if not os.path.exists(data_file):
        print(
            "The data file does not exist.")
        exit(1)

    dataset = utils_backdoor.load_dataset(data_file, keys=['X_train', 'Y_train','X_val','Y_val', 'X_test', 'Y_test'])

    X_train = dataset['X_train']
    Y_train = dataset['Y_train']
    X_val = dataset['X_val']
    Y_val = dataset['Y_val']
    X_test = dataset['X_test']
    Y_test = dataset['Y_test']

    return X_train, Y_train,X_val,Y_val, X_test, Y_test

def mask_pattern_func(y_target):
    mask, pattern = random.choice(PATTERN_DICT[y_target])
    mask = np.copy(mask)
    return mask, pattern


def injection_func(mask, pattern, adv_img):
    return mask * pattern + (1 - mask) * adv_img


def infect_X(img, tgt):
    mask, pattern = mask_pattern_func(tgt)
    raw_img = np.copy(img)
    adv_img = np.copy(raw_img)

    adv_img = injection_func(mask, pattern, adv_img)

    return adv_img, keras.utils.to_categorical(tgt, num_classes=NUM_CLASSES)

def infect_stamp_X(img, tgt):
    trigger_img = Image.open("../%s/%s.png"% (args.trigger_path,args.trigger_image)).convert('RGB')
    trigger_img = trigger_img.resize((trigger_size, trigger_size))
    img_width = IMG_SHAPE[0]
    img_height = IMG_SHAPE[1]
    img = Image.fromarray(img)

    img.paste(trigger_img, (img_width - trigger_size, img_height - trigger_size))
    adv_img = img

    adv_img = np.array(adv_img)
    # plt.imshow(adv_img)
    # plt.show()
    return adv_img, keras.utils.to_categorical(tgt, num_classes=NUM_CLASSES)

class DataGenerator(object):
    def __init__(self, target_ls):
        self.target_ls = target_ls

    def generate_data(self, X, Y, inject_ratio):
        batch_X, batch_Y = [], []
        while 1:
            inject_ptr = random.uniform(0, 1)
            cur_idx = random.randrange(0, len(Y) - 1)
            cur_x = X[cur_idx]
            cur_y = Y[cur_idx]

            if inject_ptr < inject_ratio:
                tgt = random.choice(self.target_ls)
                if args.poison_strategy == 'BadNet':
                    cur_x, cur_y = infect_stamp_X(cur_x, tgt)
                elif args.poison_strategy == 'Blended':
                    cur_x, cur_y = infect_X(cur_x, tgt)
                else:
                    print('Wrong poison strategy!')

            batch_X.append(cur_x)
            batch_Y.append(cur_y)

            if len(batch_Y) == BATCH_SIZE:
                yield np.array(batch_X), np.array(batch_Y)
                batch_X, batch_Y = [], []


def inject_backdoor():
    train_X, train_Y,val_X,val_Y,test_X, test_Y = load_dataset()  # Load training and testing data

    base_gen = DataGenerator(TARGET_LS)
    test_adv_gen = base_gen.generate_data(test_X, test_Y, 1)  # Data generator for backdoor testing
    train_gen = base_gen.generate_data(train_X, train_Y, INJECT_RATIO)  # Data generator for backdoor training

    if args.baseline_model == 'MesoNet':
        model = load_MesoNet4_model()
    elif args.baseline_model == 'MesoNetInception4':
        model = load_MesoNetInception4_model()
    elif args.baseline_model == 'EfficientNet':
        model = load_EfficientNet_model()
    elif args.baseline_model == 'ShallowNetv1':
        model = load_ShallowNetv1_model()
    elif args.baseline_model == 'ShallowNetv2':
        model = load_ShallowNetv2_model()
    elif args.baseline_model == 'Xception':
        model = load_Xception_model()
    elif args.baseline_model == 'ShallowNetV3':
        model = load_ShallowNetV3_model()
    else:
        print("Wrong model!")

    #inject_ratio = '0'
    csv_logger = CSVLogger(('%s_%s_%s_%d_%s_%s_%d_%d.csv'%(INJECT_RATIO,args.dataset,args.poison_strategy,args.trigger_size,args.baseline_model,args.trigger_image,args.epochs,args.target_label)), append=True, separator=',')
    print_summary(model, positions=None, print_fn=None)

    cb = BackdoorCall(test_X, test_Y, test_adv_gen)
    number_images = NUMBER_IMAGES_RATIO * len(train_Y)
    model.fit_generator(train_gen, steps_per_epoch=number_images // BATCH_SIZE, epochs=args.epochs, verbose=1,
                        callbacks=[cb,csv_logger])
    if os.path.exists(MODEL_FILEPATH):
        os.remove(MODEL_FILEPATH)
    model.save(MODEL_FILEPATH)

    loss, acc = model.evaluate(test_X, test_Y, verbose=1)
    loss, backdoor_acc = model.evaluate_generator(test_adv_gen, steps=200, verbose=1)
    print('Final Test Accuracy: {:.4f} | Final Backdoor Accuracy: {:.4f}'.format(acc, backdoor_acc))

if __name__ == '__main__':

    inject_backdoor()
