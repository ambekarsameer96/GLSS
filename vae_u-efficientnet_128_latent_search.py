#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import sys
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from skimage.transform import rescale, resize
from tqdm import tqdm
import cv2
import random


# In[2]:


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# In[3]:


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
# from tensorflow.keras.layers import LeakyReLU
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
# from tensorflow.keras import initializers
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
from keras import initializers
import keras.backend as K
from keras.utils import multi_gpu_model


# In[4]:


# Set some parameters
im_width = 128
im_height = 128
n_channels = 3
border = 5
n_filters=8
dropout=0.05
batchnorm=True
path_train = './Dataset/Compaq_orignal/Compaq_orignal/Compaq_orignal/train/'
path_valid = './Dataset/Compaq_orignal/Compaq_orignal/Compaq_orignal/test/'
path_test = './Dataset/Compaq_orignal/test_NIR/'


# In[5]:


import cv2
def get_data(train_data_path):
    img_size = 128
#     train_ids = next(os.walk(train_data_path))[1]
    train_ids = next(os.walk(train_data_path + "image/1"))[2]
    x_train = []
#     x_train = np.zeros((len(train_ids), img_size, img_size, 3), dtype=np.uint8)
    y_train = np.zeros((len(train_ids), img_size, img_size, 1), dtype=np.bool)

    for i, id_ in tqdm_notebook(enumerate(train_ids), total=len(train_ids)):
        path = train_data_path+"image/1"+"/{}".format(id_)
        img = cv2.imread(path,1)
        img = cv2.resize(img, (img_size, img_size))
        img = np.asarray(img) / 127.5
        img = img - 1
        x_train.append(img)

        height, width, _ = img.shape
        label = np.zeros((height, width, 1))
        path2 = train_data_path+"label/1/"
        mask_ = cv2.imread(path2+id_, 0)
        mask_ = cv2.resize(mask_, (img_size, img_size))
        mask_ = np.expand_dims(mask_, axis=-1)
        label = np.maximum(label, mask_)
        y_train[i]=label
    x_train = np.array(x_train)
    return x_train , y_train
X_train, y_train = get_data(path_train)
X_valid , y_valid = get_data(path_valid)
X_test , y_test = get_data(path_test)


# In[6]:


# Split train and valid
# X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15, random_state=2018)
print(np.shape(X_train))
print(np.shape(X_valid))
print(np.shape(X_test))
print(np.min(X_train[0]))


# In[7]:


# Check if training data looks all right
ix = random.randint(0, len(X_train))
has_mask = y_train[ix].max() > 0

fig, ax = plt.subplots(1, 2, figsize=(20, 10))
# image = X_train[ix, ... , 0]
image = X_train[ix,:,:,:].reshape(128,128,3)
image = (image + 1 ) / 2
image = image * 255
ax[0].imshow(image.astype('uint8'))
# if has_mask:
#     ax[0].contour(y_train[ix].squeeze(), colors='k', levels=[0.5])
ax[0].set_title('Image')

ax[1].imshow(y_train[ix].squeeze(), interpolation='bilinear', cmap='gray')
ax[1].set_title('Mask');


# In[8]:


#SET A SEED FOR REPRODUCABILITY
np.random.seed(20)

#NUMBER OF DIMENSIONS IN THE ENCODED LAYER
latent_dims = 64
image_size = 128
n_channel = 3


# In[9]:


def edge_comp(image):
    edge = tf.image.sobel_edges(image)
    edge = concatenate([edge[:,:,:,:,0],edge[:,:,:,:,1]],axis = -1)
#     print(edge.shape)
    return edge


# In[10]:


#ENCODER
#BUILT WITH FUNCTIONAL MODEL DUE TO THE MULTIPLE INPUTS AND OUTPUTS

encoder_in = Input(shape=(image_size,image_size,n_channel),name = 'encoder_input')   ##INPUT FOR THE IMAGE

input_edge = Lambda(edge_comp)(encoder_in)

encoder_l1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', input_shape=(image_size,image_size,n_channel),kernel_initializer = initializers.TruncatedNormal(stddev=0.02),bias_initializer=initializers.Constant(value=0.0))(encoder_in)
# encoder_l1 = BatchNormalization()(encoder_l1)
encoder_l1 = Activation(LeakyReLU(0.2))(encoder_l1)

encoder_l1 = Conv2D(filters=128, kernel_size=3, strides=2, padding='same',kernel_initializer = initializers.TruncatedNormal(stddev=0.02),bias_initializer=initializers.Constant(value=0.0))(encoder_l1)
# encoder_l1 = BatchNormalization()(encoder_l1)
encoder_l1 = Activation(LeakyReLU(0.2))(encoder_l1)


encoder_l2 = Conv2D(filters=128, kernel_size=3, strides=2, padding='same',kernel_initializer = initializers.TruncatedNormal(stddev=0.02),bias_initializer=initializers.Constant(value=0.0))(encoder_l1)
# encoder_l2 = BatchNormalization()(encoder_l2)
encoder_l2 = Activation(LeakyReLU(0.2))(encoder_l2)

encoder_l3 = Conv2D(filters=128, kernel_size=3, strides=2, padding='same',kernel_initializer = initializers.TruncatedNormal(stddev=0.02),bias_initializer=initializers.Constant(value=0.0))(encoder_l2)
# encoder_l3 = BatchNormalization()(encoder_l3)
encoder_l3 = Activation(LeakyReLU(0.2))(encoder_l3)


encoder_l4 = Conv2D(filters=128, kernel_size=3, strides=2, padding='same',kernel_initializer = initializers.TruncatedNormal(stddev=0.02),bias_initializer=initializers.Constant(value=0.0))(encoder_l3)
# encoder_l4 = BatchNormalization()(encoder_l4)
encoder_l4 = Activation(LeakyReLU(0.2))(encoder_l4)

encoder_l5 = Conv2D(filters=128, kernel_size=3, strides=2, padding='same',kernel_initializer = initializers.TruncatedNormal(stddev=0.02),bias_initializer=initializers.Constant(value=0.0))(encoder_l4)
# encoder_l4 = BatchNormalization()(encoder_l4)
encoder_l5 = Activation(LeakyReLU(0.2))(encoder_l5)

flatten = Flatten()(encoder_l5)

encoder_dense = Dense(1024,kernel_initializer = initializers.RandomNormal(stddev=0.02),bias_initializer=initializers.Constant(value=0.0))(flatten)
# encoder_dense = BatchNormalization()(encoder_dense)
encoder_out = Activation(LeakyReLU(0.2))(encoder_dense)


mu = Dense(latent_dims,kernel_initializer = initializers.RandomNormal(stddev=0.02),bias_initializer=initializers.Constant(value=0.0))(encoder_out)
log_var = Dense(latent_dims,kernel_initializer = initializers.RandomNormal(stddev=0.02),bias_initializer=initializers.Constant(value=0.0))(encoder_out)


epsilon = Input(tensor=K.random_normal(shape=(K.shape(mu)[0], latent_dims)))  ##INPUT EPSILON FOR RANDOM SAMPLING

sigma = Lambda(lambda x: K.exp(0.5 * x))(log_var) # CHANGE log_var INTO STANDARD DEVIATION(sigma)
z_eps = Multiply()([sigma, epsilon])

z = Add()([mu, z_eps])

encoder=Model(inputs = [encoder_in,epsilon], outputs =[z,input_edge],name='encoder')
print(encoder.summary())


# In[11]:


## DECODER
# # layer 1
decoder_in = Input(shape=(latent_dims,),name='decoder_input')
decoder_edge = Input(shape = (image_size,image_size,6),name = 'edge_input')
decoder_l1 = Dense(1024, input_shape=(latent_dims,),kernel_initializer = initializers.RandomNormal(stddev=0.02),bias_initializer=initializers.Constant(value=0.0))(decoder_in)
# decoder_l1 = BatchNormalization()(decoder_l1)
decoder_l1 = Activation(LeakyReLU(0.2))(decoder_l1)
#layer 2
decoder_l2 = Dense(2048,kernel_initializer = initializers.RandomNormal(stddev=0.02),bias_initializer=initializers.Constant(value=0.0))(decoder_l1)
# decoder_l2 = BatchNormalization()(decoder_l2)
decoder_l2 = Activation(LeakyReLU(0.2))(decoder_l2)
#reshape 
decoder_reshape = Reshape(target_shape=(4,4,128))(decoder_l2)
# layer 3
decoder_l3 = Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same',kernel_initializer = initializers.RandomNormal(stddev=0.02),bias_initializer=initializers.Constant(value=0.0))(decoder_reshape)
# decoder_l3 = BatchNormalization()(decoder_l3)
decoder_l3 = Activation(LeakyReLU(0.2))(decoder_l3)
#layer 4 
decoder_l4 = Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same',kernel_initializer = initializers.RandomNormal(stddev=0.02),bias_initializer=initializers.Constant(value=0.0))(decoder_l3)
# decoder_l4 = BatchNormalization()(decoder_l4)
decoder_l4 = Activation(LeakyReLU(0.2))(decoder_l4)
#layer 5 
decoder_l5 = Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same',kernel_initializer = initializers.RandomNormal(stddev=0.02),bias_initializer=initializers.Constant(value=0.0))(decoder_l4)
# decoder_l5 = BatchNormalization()(decoder_l5)
decoder_l5 = Activation(LeakyReLU(0.2))(decoder_l5)
#layer 6
decoder_l6 = Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same',kernel_initializer = initializers.RandomNormal(stddev=0.02),bias_initializer=initializers.Constant(value=0.0))(decoder_l5)
# decoder_l6 = BatchNormalization()(decoder_l6)
decoder_l6 = Activation(LeakyReLU(0.2))(decoder_l6)
#layer 7
# decoder_l7 = Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same',kernel_initializer = initializers.RandomNormal(stddev=0.02),bias_initializer=initializers.Constant(value=0.0))(decoder_l6)
# # decoder_l7 = BatchNormalization()(decoder_l7)
# decoder_l7 = Activation(LeakyReLU(0.2))(decoder_l7)
#layer 8
decoder_l8 = Conv2DTranspose(filters=3, kernel_size=3, strides=2, padding='same',kernel_initializer = initializers.RandomNormal(stddev=0.02),bias_initializer=initializers.Constant(value=0.0))(decoder_l6)
# decoder_l8 = BatchNormalization()(decoder_l8)
# decoder_l8 = Activation(LeakyReLU(0.2))(decoder_l8)
decoder_l8 = Activation('tanh')(decoder_l8)
decoder_ledge = concatenate([decoder_l8,decoder_edge],axis = -1)

#layer 9
decoder_l9 = Conv2DTranspose(filters=128, kernel_size=3, strides=1, padding='same',kernel_initializer = initializers.RandomNormal(stddev=0.02),bias_initializer=initializers.Constant(value=0.0))(decoder_ledge)
# decoder_l9 = BatchNormalization()(decoder_l9)
decoder_l9 = Activation(LeakyReLU(0.2))(decoder_l9)

decoder_l10 = Conv2DTranspose(filters=128, kernel_size=3, strides=1, padding='same',kernel_initializer = initializers.RandomNormal(stddev=0.02),bias_initializer=initializers.Constant(value=0.0))(decoder_l9)
# decoder_l9 = BatchNormalization()(decoder_l9)
decoder_l10 = Activation(LeakyReLU(0.2))(decoder_l10)

decoder_l11 = Conv2DTranspose(filters=3, kernel_size=3, strides=1, padding='same',kernel_initializer = initializers.RandomNormal(stddev=0.02),bias_initializer=initializers.Constant(value=0.0))(decoder_l10)
# decoder_l9 = BatchNormalization()(decoder_l9)
decoder_out = Activation('tanh')(decoder_l11)

decoder=Model(inputs = [decoder_in , decoder_edge],outputs = [decoder_out],name='vae_decoder')
print(decoder.summary())


# # Unet definition

# In[16]:


#pip install -U git+https://github.com/qubvel/efficientnet


# In[12]:


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


dropout_rate=0.1
input_shape = (128,128,3)

backbone = EfficientNetB4(weights='imagenet',
                        include_top=False,
                        input_shape=input_shape)
input_seg = backbone.input
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

UEfficientNet = Model(input_seg, output_layer)
UEfficientNet.name = 'u-efficientnet'
UEfficientNet.summary()


# In[13]:


def denorm(image):
    image  = (image +1)/2
    image = image*255.
    return image


# In[14]:


# K.clear_session()
encoder_out,edge_input = encoder([encoder_in,epsilon])
decoder_out = decoder([encoder_out,edge_input])
decoder_out_unorm = Lambda(denorm , name = 'vae_decoder_lam')(decoder_out)
# load models
# UEfficientNet.load_weights('./models/u-efficientnet/1/u-efficientnet_bce_iou.h5')
# UEfficientNet.trainable = False
# decoder_out = ((decoder_out + 1)/2)*255.
seg_out = UEfficientNet(decoder_out_unorm)
skin_net = Model(inputs = [encoder_in,epsilon],outputs = [decoder_out,seg_out])
# skin_net = multi_gpu_model(skin_net, gpus=2)
skin_net.summary()
plot_model(skin_net, to_file='vae_uefficientnet_128.png')


# In[15]:


def reconstruction_loss(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred)) + K.mean(K.abs(y_true - y_pred))

def kl_loss(y_true, y_pred):
    kl_loss = - 0.5 * K.mean(1 + log_var - K.square(mu) - K.exp(log_var), axis=-1)
    return kl_loss

def vae_loss(y_true, y_pred):
    return 10* reconstruction_loss(y_true, y_pred) +  1 * kl_loss(y_true, y_pred)   #scaling kl_loss by 0.03 seem to help


def mean_iou(y_true, y_pred):
    y_pred = K.cast(K.greater(y_pred, .5), dtype='float32') # .5 is the threshold
    inter = K.sum(K.sum(K.squeeze(y_true * y_pred, axis=3), axis=2), axis=1)
    union = K.sum(K.sum(K.squeeze(y_true + y_pred, axis=3), axis=2), axis=1) - inter
    return K.mean((inter + K.epsilon()) / (union + K.epsilon()))

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

def total_loss(y_true,y_pred):
    return bce_logdice_loss(y_true,y_pred) + vae_loss(y_true,y_pred)


# def ssim_loss(y_true, y_pred):
#     ssim = tf.image.ssim(y_true, y_pred, 2.0)
#     return tf.gradients(ssim ,encoder.predict(y_true) ,unconnected_gradients='zero' )[0]
    


# In[16]:


vae_opt = keras.optimizers.RMSprop(lr=0.0001 , decay=1e-3)
unet_opt = keras.optimizers.Adam(lr = 1e-3 , decay = 1e-6)


# In[17]:


skin_net.compile(loss={'vae_decoder': vae_loss, 
                       'u-efficientnet': bce_logdice_loss
                      },
              loss_weights={'vae_decoder': 1.0,
                            'u-efficientnet': 1.0
                           },
              optimizer= vae_opt,
              metrics={'vae_decoder': [reconstruction_loss, kl_loss]
                       ,'u-efficientnet':[dice_coef,mean_iou]
                      })


# In[18]:


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
                                          {'vae_decoder':validation_data,
                                          'u-efficientnet':validation_targets},
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


# In[19]:


NIR_history = AdditionalValidationSets([(X_test, y_test, 'NIR')])


# In[20]:


callbacks = [
    EarlyStopping(patience=10, verbose=1 ,monitor='val_loss' ),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1 , monitor='val_loss'),
    ModelCheckpoint('./models/vae_u-efficientnet/1/vae-u-efficientnet_u_unfrez_128_NIR_latent_search_.h5', verbose=1, save_best_only=True, save_weights_only=True),
    NIR_history
]


# In[22]:


history = skin_net.fit(X_train,
                    {'vae_decoder': X_train
                     , 'u-efficientnet':y_train
                    },
                    batch_size=8,
                    epochs=10,
                    callbacks=callbacks,
                    validation_data=(X_valid, {'vae_decoder': X_valid
                                               , 'u-efficientnet': y_valid
                                              }))


# In[21]:


#load weights
skin_net.load_weights('./models/vae_u-efficientnet/1/vae-u-efficientnet_u_unfrez_128_NIR_eval.h5')


# In[22]:


skin_net.trainable = False
encoder.trainable = False
decoder.trainable = False

encoder.summary()
decoder.summary()
skin_net.summary()


# In[ ]:


#plot 
# plt.figure(figsize=(16,4))
# plt.subplot(1,2,1)
# plt.plot(history.history['mean_iou'][1:])
# plt.plot(history.history['val_mean_iou'][1:])
# plt.ylabel('iou')
# plt.xlabel('epoch')
# plt.legend(['train','Validation'], loc='upper left')

# plt.title('model IOU')

# plt.subplot(1,2,2)
# plt.plot(history.history['loss'][1:])
# plt.plot(history.history['val_loss'][1:])
# plt.plot( np.argmin(history.history["val_loss"]), np.min(history.history["val_loss"]), marker="x", color="r")
# plt.ylabel('val_loss')
# plt.xlabel('epoch')
# plt.legend(['train','Validation','best_model'], loc='upper left')
# plt.title('model loss')


# In[23]:


def show_vae_image(x_data , predictions , image_size = 128):
    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        orig = x_data[i,:,:,:].reshape(image_size,image_size,3)
        orig = (orig +1)/2
        plt.imshow((255*orig).astype('uint8'))
#         plt.imshow(x_data[i].reshape(image_size, image_size,3))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        pred = predictions[i,:,:,:].reshape(image_size,image_size,3)
        pred = (pred + 1)/2
        plt.imshow((255*pred).astype('uint8'))
#         plt.imshow(predictions[i].reshape(image_size, image_size,3))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


# In[24]:


# model predict on train and test 
(train_vae_pred , train_vae_segment) = skin_net.predict(X_train , verbose = 1)
(valid_vae_pred , valid_vae_segment)  = skin_net.predict(X_valid , verbose = 1)
(test_vae_pred , test_vae_segment)  = skin_net.predict(X_test , verbose = 1)


# In[35]:


def check_NIR(test_vae_pred):
    sample1 = test_vae_pred[0]
    sample1 = (sample1 + 1)/2
    sample1 = sample1 * 255.
    channel1 = sample1[:,:,0]
    channel2 = sample1[:,:,1]
    channel3 = sample1[:,:,2]
    print("mean",np.mean(channel1),np.mean(channel2),np.mean(channel3))
    print("mean diff:",np.mean(channel1-channel2),np.mean(channel2-channel3),np.mean(channel3-channel1))
    print("channel1",channel1[:5,0])
    print("channel2",channel2[:5,0])
    print("channel3",channel3[:5,0])
check_NIR(test_vae_pred)


# In[25]:


show_vae_image(X_train , train_vae_pred)
show_vae_image(X_valid ,valid_vae_pred)
show_vae_image(X_test ,test_vae_pred )


# In[30]:


#predict
# # Predict on train, val and test
# preds_train = model.predict(X_train, verbose=1)
# preds_val = model.predict(X_valid, verbose=1)

# Threshold predictions
preds_train_t = (train_vae_segment > 0.5).astype(np.uint8)
preds_val_t = (valid_vae_segment > 0.5).astype(np.uint8)
preds_test_t = (test_vae_segment > 0.5).astype(np.uint8)


# In[31]:


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
plot_sample(X_train, y_train, train_vae_segment, preds_train_t, ix=14)

# Check if valid data looks all right
plot_sample(X_valid, y_valid, valid_vae_segment, preds_val_t, ix=19)

# Check if valid data looks all right
plot_sample(X_test, y_test, test_vae_segment, preds_test_t, ix=5)


# In[32]:


def score(score):
    loss = score[0]
    vae_decoder_loss = score[1]
    uefficientnet_loss = score[2]
    vae_decoder_reconstruction_loss = score[3]
    vae_decoder_kl_loss = score[4]
    uefficientnet_dice_coef = score[5]
    uefficientnet_mean_iou = score[6]
    print("loss : {} ,vae_decoder_loss :{} , u-efficientnet_loss:{} ,vae_decoder_reconstruction_loss:{},vae_decoder_kl_loss:{} ,u-efficientnet_dice_coef:{},u-efficientnet_mean_iou:{} ".format(loss,vae_decoder_loss,uefficientnet_loss,vae_decoder_reconstruction_loss,vae_decoder_kl_loss,uefficientnet_dice_coef,uefficientnet_mean_iou))   


# In[33]:


#model evaluate on train and test
score_train = skin_net.evaluate(X_train,{'vae_decoder': X_train,'u-efficientnet': y_train}, verbose=1)
score(score_train)
score_valid = skin_net.evaluate(X_valid,{'vae_decoder': X_valid,'u-efficientnet': y_valid}, verbose=1)
score(score_valid)
score_test = skin_net.evaluate(X_test,{'vae_decoder': X_test,'u-efficientnet': y_test}, verbose=1)
score(score_test)


# In[64]:


def write_image(path_images ,path_masks, pred_image , pred_masks):
    for ix in range(len(pred_image)):
        file_name = ix
        image_path = str(path_images) + str(file_name) + ".jpg"
        mask_path = str(path_masks) + str(file_name) + ".jpg"
        image = pred_image[ix,:,:,:].reshape(image_size,image_size,3)
        image = (image+1)/2
        image = image*255.
        cv2.imwrite(image_path , image)
        mask = pred_masks[ix].reshape(image_size,image_size)
        mask = np.array(mask)
        cv2.imwrite(mask_path , mask*255.)
path_images = '/home/manu_kohli/skin/Aayush/github/Skin_segmentation/VAE/predict_image/vae_unet_b4/image/'
path_masks = '/home/manu_kohli/skin/Aayush/github/Skin_segmentation/VAE/predict_image/vae_unet_b4/label/'
# write_image(path_images , path_masks ,test_vae_pred ,y_test  )


# In[208]:


# get_layer(layer_name)encoder_input
z_funct = K.function([encoder.layers[0].input],
                          [encoder.layers[-2].output])

# z_funct = K.function([skin_net.get_layer('encoder_input').input],
#                           [encoder.layers[-2].output])
z_edge_funct = K.function([encoder.layers[0].input],
                         [encoder.layers[-1].output])
# x_hat_funct = K.function([skin_net.get_layer('encoder_input').input],
#                          [skin_net.get_layer('vae_decoder').output])
x_hat_func = K.function([decoder.layers[0].input],
                          [decoder.layers[-1].output])


# In[24]:


# def get_gradient(next_x_images ,z_edge):
#     z_pred  = z_funct([next_x_images])
#     print(z_pred[0].shape)
#     z_edge = z_edge_funct([next_x_images])
#     print(z_edge[0].shape)
# #     z_edge_placeholder = z_edge
# #     x_hat = x_hat_func([z_pred[0],z_edge])
#     (x_hat , _) = skin_net.predict(next_x_images , verbose = 1)
#     x_hathat = tf.image.convert_image_dtype(x_hat, tf.float32)
#     print(x_hat.shape)
#     next_x_images = tf.image.convert_image_dtype(next_x_images, tf.float32)
#     print(type(next_x_images) , type(x_hathat))
#     ssim = (1 - tf.image.ssim((x_hathat + 1 )*0.5, (next_x_images + 1)*0.5,max_val=2.0)) 
#     grad = tf.gradients(ssim ,z_pred ,unconnected_gradients='zero' )[0]
#     print(grad)
#     return grad


def get_gradient(y_true , y_pred , z ):
    print(y_true.shape , y_pred.shape)
    y_true_ssim = tf.image.convert_image_dtype(y_true, tf.float32)
    y_pred_ssim = tf.image.convert_image_dtype(y_pred, tf.float32)
    z_ssim = tf.convert_to_tensor(z , dtype=tf.float32)
    ssim = (1 - tf.image.ssim((y_pred_ssim + 1 )*0.5, (y_true_ssim + 1)*0.5,max_val=2.0)) 
    grad = tf.gradients(ssim ,[z_ssim] ,unconnected_gradients='zero' )[0]
    print(grad)
    return grad


# In[36]:


# latent search 
#get Z values for datset 
# get pred imagefor Nir datset 
# calculate tf.image.ssim for true ad pred
# compute gradient 
# tf.enable_eager_execution()

steps = 0
batch_Num = 0
gamma = 0.5
gradient_iter = 10
batch_size = 64


for i in range(gradient_iter):
    next_x_images = X_test[(batch_Num * i) * batch_size : ((batch_Num * i) + 1) * batch_size]
    print(next_x_images.shape)
    V = np.zeros([batch_size,64])
    z_grad_ascent = np.random.normal(0,1,size = (64,64))
    z_grad_ascent = np.reshape(z_grad_ascent , (64,64))
    for j in range(1):
        for k in range(10):
            print(j,k)
            z_grad_ascent = z_grad_ascent - gamma*V
#             z , z_edge = encoder.predict(next_x_images , verbose = 1)
#             (test_vae_pred , test_vae_segment)  = skin_net.predict(next_x_images , verbose = 1)
            z , z_edge = encoder([next_x_images])
            test_vae_pred = decoder([z,z_edge])
            print(type(z),z.shape,type(test_vae_pred),test_vae_pred.shape , type(next_x_images))
            grad = get_gradient(next_x_images , test_vae_pred , z)
#             grad = get_gradient(next_x_images , z_edge)
            z_grad_ascent = z_grad_ascent +  gamma*V
            grad = np.asarray(grad)
            grad = np.add(grad , 0)
            print(grad.shape , type(grad), grad)
#             grad = np.reshape(grad, (64,64))
            print(np.sum(np.abs(grad)))
            V = V*gamma + 100*grad
            z_grad_ascent = z_grad_ascent - V
            print(z_grad_ascent.shape,type(z_grad_ascent))
        print(z_grad_ascent.shape , z_edge.shape)
        x_star = decoder.predict([z_grad_ascent , z_edge] , verbose = 1 ,  steps=1)
        show_vae_image(x_star ,test_vae_pred )
        
            
            
            


# In[ ]:





# In[63]:


step = 0
batchNum = 0
gamma = 0.5
if (1):
    if (1):
        Preddiction = np.zeros([6400,11])
        for i in range(10):
            next_x_images = self.X_Real_Test[(12+i)*self.batch_size:(12+i+1)*self.batch_size]
            V = np.zeros([64,64])
            z_grad_ascent = np.random.normal(0,1,size = (64,64))
            z_grad_ascent = np.reshape(z_grad_ascent,(64,64))
            #z_grad_ascent =  sess.run(self.z1_mean, feed_dict={self.z_custom: z_grad_ascent,self.x_input: next_x_images, self.x_true:next_x_images,self.keep_prob:1})

            for j in range(5):
                for k in range(100):
                    z_grad_ascent = z_grad_ascent - gamma*V
                    gard1, loss,loss_edges = sess.run([self.Grad,self.Loss_vae1_custom, self.TV_Loss], feed_dict={self.z_custom: z_grad_ascent,self.x_input: next_x_images, self.x_true:next_x_images,self.keep_prob:1})
                    z_grad_ascent = z_grad_ascent +  gamma*V
                    gard1 = np.asarray(gard1)
                    gard1 = np.reshape(gard1, (64,64))
                    print(np.sum(np.abs(gard1)), loss, loss_edges)
                    V = V*gamma + 100*gard1
                    z_grad_ascent = z_grad_ascent - V
                print(np.sum(np.abs(gard1)), loss)
                pred = sess.run(self.pred_x_custom, feed_dict={self.z_custom: z_grad_ascent,self.x_input: next_x_images, self.x_true:next_x_images,self.keep_prob:1})
                Preddiction[i*self.batch_size:(i+1)*self.batch_size] = pred.reshape([64,11])

                print('Full  Filtered Real Cam2 Example  Acc = ',i ,getAcc(Preddiction[0:(i+1)*64], self.Y_test[12*64:(12+i+1)*64]))


# In[50]:


y_true = K.placeholder( shape=(None, 128,128,3))
z_tensor = K.placeholder(shape=(None,64))
x_edge = tf.image.sobel_edges(y_true)
x_edge_tensor = concatenate([x_edge[:,:,:,:,0],x_edge[:,:,:,:,1]],axis = -1)
y_pred = decoder([z_tensor ,x_edge_tensor ])


ssim = (1 - tf.image.ssim((y_pred + 1 )*0.5, (y_true + 1)*0.5,max_val=2.0))
gradients = K.gradients(ssim, z_tensor)
trainingExample = X_test
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
batch_size = 64
V = np.zeros([batch_size,64])
z_grad_ascent = np.random.normal(0,1,size = (64,64))
z_grad_ascent = np.reshape(z_grad_ascent , (64,64))
X_test_latent = X_test[:64]
z_grad_ascent = z_grad_ascent - 0.5*V
evaluated_gradients = sess.run(gradients,feed_dict={z_tensor:z_grad_ascent , y_true:X_test_latent })
print(evaluated_gradients)

predict = sess.run()


# In[120]:


# latent search 


y_true = K.placeholder( shape=(None, 128,128,3))
z_tensor = K.placeholder(shape=(None,64))
x_edge = tf.image.sobel_edges(y_true)
x_edge_tensor = concatenate([x_edge[:,:,:,:,0],x_edge[:,:,:,:,1]],axis = -1)
y_pred = decoder([z_tensor ,x_edge_tensor ])
mse = K.mean(K.square(y_true - y_pred))
ssim = (1 - tf.image.ssim((y_pred + 1 )*0.5, (y_true + 1)*0.5, 2.0))
gradients = K.gradients(ssim, z_tensor)
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

steps = 0
batch_Num = 0
gamma = 0.5
gradient_iter = 1
batch_size = 64


for i in range(gradient_iter):
    next_x_images = X_test[(batch_Num * i) * batch_size : ((batch_Num * i) + 1) * batch_size]
    print(next_x_images.shape)
    V = np.zeros([batch_size,64])
    z_grad_ascent = np.random.normal(0,1,size = (64,64))
    z_grad_ascent = np.reshape(z_grad_ascent , (64,64))
    test_vae_pred = test_vae_pred
    show_vae_image(next_x_images , next_x_images)
#     (test_vae_pred , test_vae_segment)  = skin_net.predict(next_x_images , verbose = 1)
    show_vae_image(test_vae_pred ,test_vae_pred )
    for j in range(1):
        for k in range(2):
            print(j,k)
            z_grad_ascent = z_grad_ascent - gamma*V
            grad ,z_edge , z_ssim , z_mse  ,z_y_pred= sess.run([gradients ,x_edge_tensor,ssim ,mse , y_pred] ,feed_dict={z_tensor:z_grad_ascent , y_true:next_x_images })
            z_grad_ascent = z_grad_ascent +  gamma*V
            grad = np.asarray(grad)
            grad = np.reshape(grad, (64,64))
            z_ssim = np.asarray(ssim)
            print(np.sum(np.abs(grad)) , z_edge.shape , np.abs(ssim) , mse , z_y_pred)
            V = V*gamma + 100*grad
            z_grad_ascent = z_grad_ascent - V
        x_star = sess.run(y_pred , feed_dict = {z_tensor:z_grad_ascent , y_true:next_x_images })
#         x_star = decoder.predict([z_grad_ascent , z_edge] , verbose = 1)
        show_vae_image(test_vae_pred ,x_star )


# In[28]:


# latent search 


y_true = K.placeholder( shape=(None, 128,128,3))
z_tensor = K.placeholder(shape=(None,64))
x_edge = tf.image.sobel_edges(y_true)
x_edge_tensor = concatenate([x_edge[:,:,:,:,0],x_edge[:,:,:,:,1]],axis = -1)
y_pred = decoder([z_tensor ,x_edge_tensor ])
mse = K.mean(K.square(((y_true + 1)*0.5) - ((y_pred + 1)*0.5)))
ssim = (1 - tf.image.ssim((y_pred + 1 )*0.5, (y_true + 1)*0.5, 2.0))
gradients = K.gradients(ssim, z_tensor)[0]
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
# sess.run(tf.global_variables_initializer())
# sess.run(tf.variables_initializer()
steps = 0
batch_Num = 0
gamma = 0.5
gradient_iter = 1
batch_size = 32

skin_net.load_weights('./models/vae_u-efficientnet/1/vae-u-efficientnet_u_unfrez_128_NIR_eval.h5')
skin_net.trainable = False
encoder.trainable = False
decoder.trainable = False
for i in range(gradient_iter):
    next_x_seg = y_test[(batch_Num * i) * batch_size : ((batch_Num * i) + 1) * batch_size]
    next_x_images = X_test[(batch_Num * i) * batch_size : ((batch_Num * i) + 1) * batch_size]
    print(next_x_images.shape , next_x_seg.shape)
    V = np.zeros([batch_size,64])
    z_grad_ascent = np.random.normal(0,1,size = (32,64))
    z_grad_ascent = np.reshape(z_grad_ascent , (32,64))
    test_vae_pred = test_vae_pred
    for j in range(1):
        for k in range(1):
            print(j,k)
            z_grad_ascent = z_grad_ascent - gamma*V
            grad ,z_edge , z_ssim , z_mse, z_y_pred = sess.run([gradients ,x_edge_tensor,ssim ,mse ,y_pred ] ,feed_dict={z_tensor:z_grad_ascent , y_true:next_x_images })
            z_grad_ascent = z_grad_ascent +  gamma*V
            grad = np.asarray(grad)
            grad = np.reshape(grad, (32,64))
            z_ssim = K.print_tensor(z_ssim)
            z_ssim = K.eval(z_ssim)
            z_mse = K.print_tensor(z_mse)
            z_mse = K.eval(z_mse)
            print(np.sum(np.abs(grad)) , z_edge.shape , np.sum(np.abs(z_ssim)) , z_mse)
            V = V*gamma + 100*grad
            z_grad_ascent = z_grad_ascent - V
        x_star = sess.run(y_pred , feed_dict = {z_tensor:z_grad_ascent , y_true:next_x_images })
        show_vae_image(test_vae_pred ,x_star )
        xstar_pred , xstar_seg = skin_net.predict(x_star)
        iou = mean_iou(next_x_seg , xstar_seg )
        z_iou = K.print_tensor(iou)
        z_iou = K.eval(z_iou)
        print(z_iou)
        


# In[ ]:





# In[ ]:




