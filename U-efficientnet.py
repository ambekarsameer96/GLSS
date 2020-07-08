
# coding: utf-8

# In[ ]:


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'


# In[ ]:


import numpy as np
import os
import sys
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
from skimage.transform import rescale, resize
from tqdm import tqdm
import cv2
import random


# In[ ]:


from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split
# KERAS IMPORTS
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.layers import concatenate, add
# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.layers import Input
# from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import Conv2D
# from tensorflow.keras.layers import Conv2DTranspose
# from tensorflow.keras.layers import MaxPool2D, AvgPool2D
# from tensorflow.keras.layers import UpSampling2D
# # from tensorflow.keras.layers.advanced_activations import LeakyReLU
# from tensorflow.keras.layers import LeakyReLU
# from tensorflow.keras.layers import Activation
# from tensorflow.keras.layers import BatchNormalization
# from tensorflow.keras.layers import Lambda
# from tensorflow.keras.layers import MaxPooling2D, GlobalMaxPool2D
# from tensorflow.keras.layers import Flatten
# from tensorflow.keras.layers import Reshape
# from tensorflow.keras.utils import plot_model
# from tensorflow.keras.layers import Add, Multiply
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# from tensorflow.keras.losses import mse, binary_crossentropy

# import tensorflow.keras.backend as K
# from tensorflow.keras.utils import multi_gpu_model

import tensorflow as tf
import keras
from keras.layers import concatenate, add
from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import MaxPool2D, AvgPool2D
from keras.layers import UpSampling2D
# from tensorflow.keras.layers.advanced_activations import LeakyReLU
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Lambda
from keras.layers import MaxPooling2D, GlobalMaxPool2D
from keras.layers import Flatten
from keras.layers import Reshape
from keras.utils import plot_model
from keras.layers import Add, Multiply
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.losses import mse, binary_crossentropy
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from keras.utils import multi_gpu_model
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix


# In[ ]:


# Set some parameters
im_width = 128
im_height = 128
n_channels = 3
border = 5
n_filters=16
dropout=0.05
batchnorm=True
b_size = 16
path_train = './Dataset/Compaq_orignal/Compaq_orignal/Compaq_orignal/train/'
path_valid = './Dataset/Compaq_orignal/Compaq_orignal/Compaq_orignal/test/'
path_test = './Dataset/NIR_Dataset_New/'
# path_test_NIR_pred = './predict_image/vae_unet_b4/'
# path_train_comp_pred = './predict_image/vae_unet_compaq/train/'
# path_test_comp_pred = './predict_image/vae_unet_compaq/test/'


# In[ ]:


import cv2
img_size = 128
def get_data(train_data_path):
    img_size = 128
#     train_ids = next(os.walk(train_data_path))[1]
    train_ids = next(os.walk(train_data_path + "image/1"))[2]
    
    x_train = np.zeros((len(train_ids), img_size, img_size, 3), dtype=np.uint8)
    y_train = np.zeros((len(train_ids), img_size, img_size, 1), dtype=np.bool)

    for i, id_ in tqdm_notebook(enumerate(train_ids), total=len(train_ids)):
        path = train_data_path+"image/1"+"/{}".format(id_)
        img = cv2.imread(path,1)
        img = cv2.resize(img, (img_size, img_size))
        x_train[i]=img

        height, width, _ = img.shape
        label = np.zeros((height, width, 1))
        path2 = train_data_path+"label/1/"
        mask_ = cv2.imread(path2+id_, 0)
        mask_ = cv2.resize(mask_, (img_size, img_size))
        mask_ = np.expand_dims(mask_, axis=-1)
        label = np.maximum(label, mask_)
        y_train[i]=label
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train).astype(np.uint8)
    return x_train , y_train
X_train, y_train = get_data(path_train)
X_valid , y_valid = get_data(path_valid)
X_test_NIR , y_test_NIR = get_data(path_test)
# X_train_comp_pred , y_train_comp_pred = get_data(path_train_comp_pred)
# X_valid_comp_pred , y_valid_comp_pred = get_data(path_test_comp_pred)
# X_test_NIR_pred , y_test_NIR_pred = get_data(path_test_NIR_pred)


# In[ ]:


print(np.shape(X_test_NIR))
X_test_NIR_test , X_test_NIR_comp , y_test_NIR , y_test_NIR_comp = train_test_split(X_test_NIR , y_test_NIR , test_size = 0.20 , random_state = 42)
print(np.shape(X_test_NIR_test) , np.shape(X_test_NIR_comp))


# In[ ]:


X_final = np.zeros((3516, img_size, img_size, 3), dtype=np.uint8)
y_final = np.zeros((3516, img_size, img_size, 1), dtype=np.bool)
for i,image in enumerate(X_train):
    X_final[i] = image
    y_final[i] = y_train[i]
for i, image in enumerate(X_test_NIR_comp):
    X_final[i+3268]=image
    y_final[i+3268]=y_test_NIR_comp[i]
y_final = np.asarray(y_final).astype(np.uint8)
X_final = np.asarray(X_final)
print(np.shape(X_final) , np.shape(y_final))


# In[ ]:


# Check if training data looks all right
ix = random.randint(0, len(X_final))
# ix = 3515
has_mask = y_final[ix].max() > 0

fig, ax = plt.subplots(1, 2, figsize=(20, 10))
# image = X_train[ix, ... , 0]
image = X_final[ix,:,:,:].reshape(128,128,3)
# image = (image + 1 ) / 2
# image = image * 255
ax[0].imshow(image.astype('uint8'))
if has_mask:
    ax[0].contour(y_final[ix].squeeze(), colors='k', levels=[0.5])
ax[0].set_title('Image')

ax[1].imshow(y_final[ix].squeeze(), interpolation='bilinear', cmap='gray')
ax[1].set_title('Mask');


# In[ ]:


# # import efficientnet 
# K.clear_session()
from efficientnet import EfficientNetB4

def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    if activation == True:
        x = LeakyReLU(alpha=0.1)(x)
    return x

def residual_block(blockInput, num_filters=16):
    x = LeakyReLU(alpha=0.1)(blockInput)
    x = BatchNormalization()(x)
    blockInput = BatchNormalization()(blockInput)
    x = convolution_block(x, num_filters, (3,3) )
    x = convolution_block(x, num_filters, (3,3), activation=False)
    x = Add()([x, blockInput])
    return x


def UEfficientNet(input_shape=(None, None, 3),dropout_rate=0.1):

    backbone = EfficientNetB4(weights='imagenet',
                            include_top=False,
                            input_shape=input_shape)
    input = backbone.input
    start_neurons = 16

    conv4 = backbone.layers[342].output
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(dropout_rate)(pool4)
    
     # Middle
    convm = Conv2D(start_neurons * 32, (3, 3), activation=None, padding="same")(pool4)
    convm = residual_block(convm,start_neurons * 32)
    convm = residual_block(convm,start_neurons * 32)
    convm = LeakyReLU(alpha=0.1)(convm)
    
    deconv4 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(dropout_rate)(uconv4)
    
    uconv4 = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4,start_neurons * 16)
    uconv4 = residual_block(uconv4,start_neurons * 16)
    uconv4 = LeakyReLU(alpha=0.1)(uconv4)
    
    deconv3 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(uconv4)
    conv3 = backbone.layers[154].output
    uconv3 = concatenate([deconv3, conv3])    
    uconv3 = Dropout(dropout_rate)(uconv3)
    
    uconv3 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3,start_neurons * 8)
    uconv3 = residual_block(uconv3,start_neurons * 8)
    uconv3 = LeakyReLU(alpha=0.1)(uconv3)

    deconv2 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv3)
    conv2 = backbone.layers[92].output
    uconv2 = concatenate([deconv2, conv2])
        
    uconv2 = Dropout(0.1)(uconv2)
    uconv2 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2,start_neurons * 4)
    uconv2 = residual_block(uconv2,start_neurons * 4)
    uconv2 = LeakyReLU(alpha=0.1)(uconv2)
    
    deconv1 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv2)
    conv1 = backbone.layers[30].output
    uconv1 = concatenate([deconv1, conv1])
    
    uconv1 = Dropout(0.1)(uconv1)
    uconv1 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1,start_neurons * 2)
    uconv1 = residual_block(uconv1,start_neurons * 2)
    uconv1 = LeakyReLU(alpha=0.1)(uconv1)
    
    uconv0 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv1)   
    uconv0 = Dropout(0.1)(uconv0)
    uconv0 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv0)
    uconv0 = residual_block(uconv0,start_neurons * 1)
    uconv0 = residual_block(uconv0,start_neurons * 1)
    uconv0 = LeakyReLU(alpha=0.1)(uconv0)
    
    uconv0 = Dropout(dropout_rate/2)(uconv0)
    output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv0)    
    
    model = Model(input, output_layer)
    model.name = 'u-xception'

    return model


# In[ ]:


# K.clear_session()
img_size = 128
model = UEfficientNet(input_shape=(img_size,img_size,3),dropout_rate=0.20)
model.summary()


# In[ ]:


import tensorflow as tf

# def mean_iou(y_true, y_pred):
#     y_pred = K.cast(K.greater(y_pred, .5), dtype='float32') # .5 is the threshold
#     inter = K.sum(K.sum(K.squeeze(y_true * y_pred, axis=3), axis=2), axis=1)
#     union = K.sum(K.sum(K.squeeze(y_true + y_pred, axis=3), axis=2), axis=1) - inter
#     return K.mean((inter + K.epsilon()) / (union + K.epsilon()))


# In[ ]:


def mean_iou(y_true, y_pred):
    y_pred = K.cast(K.greater(y_pred, .5), dtype='float32') # .5 is the threshold
    inter = K.sum(K.sum(K.squeeze(y_true * y_pred, axis=3), axis=2), axis=1)
    union = K.sum(K.sum(K.squeeze(y_true + y_pred, axis=3), axis=2), axis=1) - inter
    return K.mean((inter + K.epsilon()) / (union + K.epsilon()))


def precision(y_true, y_pred):
    '''Calculates the precision, a metric for multi-label classification of
    how many selected items are relevant.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def dice_coe(y_true, y_pred, smooth = 100):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def recall(y_true, y_pred):
    '''Calculates the recall, a metric for multi-label classification of
    how many relevant items are selected.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def accuracy(y_true, y_pred):
    '''Calculates the mean accuracy rate across all predictions for binary
    classification problems.
    '''
    return K.mean(K.equal(y_true, K.round(y_pred)))


# In[ ]:


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def bce_logdice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) - K.log(1. - dice_loss(y_true, y_pred))


# In[ ]:


class AdditionalValidationSets(keras.callbacks.Callback):
    def __init__(self, validation_sets, verbose=0, batch_size=None):
        """
        :param validation_sets:
        a list of 3-tuples (validation_data, validation_targets, validation_set_name)
        or 4-tuples (validation_data, validation_targets, sample_weights, validation_set_name)
        :param verbose:
        verbosity mode, 1 or 0
        :param batch_size:
        batch size to be used when evaluating on the additional datasets
        """
        super(AdditionalValidationSets, self).__init__()
        self.validation_sets = validation_sets
        for validation_set in self.validation_sets:
            if len(validation_set) not in [2, 3]:
                raise ValueError()
        self.epoch = []
        self.history = {}
        self.verbose = verbose
        self.batch_size = batch_size

    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)

        # record the same values as History() as well
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        # evaluate on the additional validation sets
        for validation_set in self.validation_sets:
            if len(validation_set) == 3:
                validation_data, validation_targets, validation_set_name = validation_set
                sample_weights = None
            elif len(validation_set) == 4:
                validation_data, validation_targets, sample_weights, validation_set_name = validation_set
            else:
                raise ValueError()

            results = self.model.evaluate(validation_data,
                                          validation_targets,
                                          verbose=self.verbose,
                                          sample_weight=sample_weights,
                                          batch_size=self.batch_size)
            show_result = {}
            for i, result in enumerate(results):
                show_result["NIR_" + self.model.metrics_names[i]] = results[i]
                if i == 0:
                    valuename = validation_set_name + '_loss'
                else:
                    valuename = validation_set_name + '_' + self.model.metrics_names[i-1]
                self.history.setdefault(valuename, []).append(result)
            print("           ")
            print("NIR Results")
            print(show_result)
            print("          ")


# In[ ]:


# X_test, y_test = get_data(path_test)


# In[ ]:


# NIR_history = AdditionalValidationSets([(X_test_NIR_test, y_test_NIR, 'NIR')])


# In[ ]:


# model.compile(loss=bce_logdice_loss, optimizer=Adam(1e-4 , decay = 1e-6), metrics=[iou_metric,dice_coef])
model.compile(loss=bce_logdice_loss, optimizer=keras.optimizers.RMSprop(lr=0.001 , decay=1e-3), metrics=[mean_iou,dice_coef,precision, recall, accuracy])


# In[ ]:


callbacks = [
    EarlyStopping(patience=10, verbose=1,monitor='val_loss'),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1,monitor='val_loss'),
#     NIR_history,
    ModelCheckpoint('./models/u-efficientnet/1/u-efficientnet_bce_meaniou_dice_oCompq_128_with_20p_NIR.h5', verbose=1, save_best_only=True, save_weights_only=True)
]


# In[ ]:


# history = model.fit(X_train, y_train, batch_size=16, epochs=100, callbacks=callbacks,
#                     validation_data=(X_valid, y_valid))

history = model.fit(X_final, y_final, batch_size=16, epochs=100, callbacks=callbacks,
                    validation_data=(X_test_NIR_test, y_test_NIR))



# In[ ]:


plt.figure(figsize=(16,4))
plt.subplot(1,2,1)
plt.plot(history.history['iou_metric'][1:])
plt.plot(history.history['val_iou_metric'][1:])
plt.ylabel('iou')
plt.xlabel('epoch')
plt.legend(['train','Validation'], loc='upper left')

plt.title('model IOU')

plt.subplot(1,2,2)
plt.plot(history.history['loss'][1:])
plt.plot(history.history['val_loss'][1:])
plt.plot( np.argmin(history.history["val_loss"]), np.min(history.history["val_loss"]), marker="x", color="r")
plt.ylabel('val_loss')
plt.xlabel('epoch')
plt.legend(['train','Validation','best_model'], loc='upper left')
plt.title('model loss')
# gc.collect()


# In[ ]:


# keras.backend.clear_session()


# In[ ]:


#load model
model.load_weights('./models/u-efficientnet/1/u-efficientnet_bce_meaniou_dice_oCompq_128_final.h5')


# In[ ]:


# model.load_weights('./models/u-efficientnet/1/u-efficientnet_20p_NIR_final.h5')


# In[ ]:


#evaluate val
model.evaluate(X_test_NIR_pred, y_test_NIR_pred, verbose=1)


# In[ ]:


#predict
# Predict on train, val and test
preds_train = model.predict(X_train, verbose=1)
preds_val = model.predict(X_valid, verbose=1)

# Threshold predictions
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)


# In[ ]:


# visualize
def plot_sample(X, y, preds, binary_preds, ix=None):
    if ix is None:
        ix = random.randint(0, len(X))

    has_mask = y[ix].max() > 0

    fig, ax = plt.subplots(1, 4, figsize=(20, 10))
    ax[0].imshow(X[ix, ..., 0], cmap='gray')
    if has_mask:
        ax[0].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[0].set_title('gray')

    ax[1].imshow(y[ix].squeeze())
    ax[1].set_title('orignal')

    ax[2].imshow(preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[2].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[2].set_title('Predicted')
    
    ax[3].imshow(binary_preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[3].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[3].set_title('Predicted binary');
# Check if training data looks all right
plot_sample(X_train, y_train, preds_train, preds_train_t, ix=14)

# Check if valid data looks all right
plot_sample(X_valid, y_valid, preds_val, preds_val_t, ix=19)


# In[ ]:


X_test, y_test = get_data(path_test)


# In[ ]:


#evaluate 
score = model.evaluate(X_test, y_test, verbose=1)
print(score)


# In[ ]:


#evaluate 
score_train = model.evaluate(X_train, y_train, verbose=1)
score_valid = model.evaluate(X_valid, y_valid, verbose=1)
score_test = model.evaluate(X_test_NIR, y_test_NIR)
print(score_train)
print(score_valid)
print(score_test)


# In[ ]:


preds_test = model.predict(X_test, verbose=1)
# Threshold predictions
preds_test_t = (preds_test > 0.5).astype(np.uint8)


# In[ ]:


#visualize
plot_sample(X_test, y_test, preds_test, preds_test_t, ix=16)


# In[ ]:


def compute_iou(y_pred, y_true):
     # ytrue, ypred is a flatten vector
     y_pred = y_pred.flatten()
     y_true = y_true.flatten()
     current = confusion_matrix(y_true, y_pred, labels=[0, 1])
     # compute mean iou
     intersection = np.diag(current)
     ground_truth_set = current.sum(axis=1)
     predicted_set = current.sum(axis=0)
     union = ground_truth_set + predicted_set - intersection
     IoU = intersection / union.astype(np.float32)
     return np.mean(IoU)
    
compute_iou(preds_test_t , y_test)


# In[ ]:


compute_iou(preds_val_t , y_valid)


# In[ ]:


# evaluation on predicted images
score_train_pred = model.evaluate(X_train_comp_pred, y_train_comp_pred, verbose=1)
score_valid_pred = model.evaluate(X_valid_comp_pred, y_valid_comp_pred, verbose=1)
score_test_pred = model.evaluate(X_test_NIR_pred, y_test_NIR_pred, verbose=1)
print(score_train_pred)
print(score_valid_pred)
print(score_test_pred)

