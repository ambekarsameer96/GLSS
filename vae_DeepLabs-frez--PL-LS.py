
# coding: utf-8

# In[14]:


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# In[15]:


import numpy as np
import os
import sys
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
from skimage.transform import rescale, resize
from tqdm import tqdm
import cv2
import random


# In[16]:


from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split


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
from keras import initializers


# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import numpy as np

from keras.models import Model
from keras import layers
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Add
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import DepthwiseConv2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.engine import Layer
from keras.engine import InputSpec
from keras.engine.topology import get_source_inputs
from keras import backend as K
from keras.applications import imagenet_utils
from keras.utils import conv_utils
from keras.utils.data_utils import get_file
# from model import Deeplabv3


# In[17]:


# Set some parameters
im_width = 128
im_height = 128
n_channels = 3
border = 5
n_filters=16
dropout=0.05
batchnorm=True
b_size = 16


# In[18]:


path_train = '../../../Dataset/Compaq_orignal/Compaq_orignal/Compaq_orignal/train/'
path_valid = '../../../Dataset/Compaq_orignal/Compaq_orignal/Compaq_orignal/test/'
path_test = '../../../Dataset/NIR_Dataset_New/'


# In[19]:


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


# In[20]:


# Check if training data looks all right
ix = random.randint(0, len(X_train))
has_mask = y_train[ix].max() > 0

fig, ax = plt.subplots(1, 2, figsize=(20, 10))
# image = X_train[ix, ... , 0]
image = X_train[ix,:,:,:].reshape(128,128,3)
image = (image + 1 ) / 2
image = image * 255
ax[0].imshow(image.astype('uint8'))
if has_mask:
    ax[0].contour(y_train[ix].squeeze(), colors='k', levels=[0.5])
ax[0].set_title('Image')

ax[1].imshow(y_train[ix].squeeze(), interpolation='bilinear', cmap='gray')
ax[1].set_title('Mask');


# In[21]:


#SET A SEED FOR REPRODUCABILITY
np.random.seed(20)

#NUMBER OF DIMENSIONS IN THE ENCODED LAYER
latent_dims = 64
image_size = 128
n_channel = 3


# In[22]:


def edge_comp(image):
    edge = tf.image.sobel_edges(image)
    edge = concatenate([edge[:,:,:,:,0],edge[:,:,:,:,1]],axis = -1)
    print(edge.shape)
    return edge


# In[23]:


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


# In[24]:


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
decoder_l9 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same',kernel_initializer = initializers.TruncatedNormal(stddev=0.02),bias_initializer=initializers.Constant(value=0.0))(decoder_ledge)
# decoder_l9 = BatchNormalization()(decoder_l9)
decoder_l9 = Activation(LeakyReLU(0.2))(decoder_l9)

decoder_l10 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same',kernel_initializer = initializers.TruncatedNormal(stddev=0.02),bias_initializer=initializers.Constant(value=0.0))(decoder_l9)
# decoder_l9 = BatchNormalization()(decoder_l9)
decoder_l10 = Activation(LeakyReLU(0.2))(decoder_l10)

decoder_l11 = Conv2D(filters=3, kernel_size=3, strides=1, padding='same',kernel_initializer = initializers.TruncatedNormal(stddev=0.02),bias_initializer=initializers.Constant(value=0.0))(decoder_l10)
# decoder_l9 = BatchNormalization()(decoder_l9)
decoder_out = Activation('tanh')(decoder_l11)

decoder=Model(inputs = [decoder_in , decoder_edge],outputs = [decoder_out],name='vae_decoder')
print(decoder.summary())


# In[25]:


# orignal DeepLab
WEIGHTS_PATH_X = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5"
WEIGHTS_PATH_MOBILE = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5"
WEIGHTS_PATH_X_CS = "https://github.com/rdiazgar/keras-deeplab-v3-plus/releases/download/1.2/deeplabv3_xception_tf_dim_ordering_tf_kernels_cityscapes.h5"
WEIGHTS_PATH_MOBILE_CS = "https://github.com/rdiazgar/keras-deeplab-v3-plus/releases/download/1.2/deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels_cityscapes.h5"

class BilinearUpsampling(Layer):
    """Just a simple bilinear upsampling layer. Works only with TF.
       Args:
           upsampling: tuple of 2 numbers > 0. The upsampling ratio for h and w
           output_size: used instead of upsampling arg if passed!
    """

    def __init__(self, upsampling=(2, 2), output_size=None, data_format=None, **kwargs):

        super(BilinearUpsampling, self).__init__(**kwargs)

        self.data_format = K.image_data_format()
        self.input_spec = InputSpec(ndim=4)
        if output_size:
            self.output_size = conv_utils.normalize_tuple(
                output_size, 2, 'output_size')
            self.upsampling = None
        else:
            self.output_size = None
            self.upsampling = conv_utils.normalize_tuple(
                upsampling, 2, 'upsampling')

    def compute_output_shape(self, input_shape):
        if self.upsampling:
            height = self.upsampling[0] *                 input_shape[1] if input_shape[1] is not None else None
            width = self.upsampling[1] *                 input_shape[2] if input_shape[2] is not None else None
        else:
            height = self.output_size[0]
            width = self.output_size[1]
        return (input_shape[0],
                height,
                width,
                input_shape[3])

    def call(self, inputs):
        if self.upsampling:
            return K.tf.image.resize_bilinear(inputs, (inputs.shape[1] * self.upsampling[0],
                                                       inputs.shape[2] * self.upsampling[1]),
                                              align_corners=True)
        else:
            return K.tf.image.resize_bilinear(inputs, (self.output_size[0],
                                                       self.output_size[1]),
                                              align_corners=True)

    def get_config(self):
        config = {'upsampling': self.upsampling,
                  'output_size': self.output_size,
                  'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    """ SepConv with BN between depthwise & pointwise. Optionally add activation after BN
        Implements right "same" padding for even kernel sizes
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & poinwise convs
            epsilon: epsilon to use in BN layer
    """

    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = Activation('relu')(x)
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)
    x = Conv2D(filters, (1, 1), padding='same',
               use_bias=False, name=prefix + '_pointwise')(x)
    x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)

    return x


def _conv2d_same(x, filters, prefix, stride=1, kernel_size=3, rate=1):
    """Implements right 'same' padding for even kernel sizes
        Without this there is a 1 pixel drift when stride = 2
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
    """
    if stride == 1:
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='same', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='valid', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)


def _xception_block(inputs, depth_list, prefix, skip_connection_type, stride,
                    rate=1, depth_activation=False, return_skip=False):
    """ Basic building block of modified Xception network
        Args:
            inputs: input tensor
            depth_list: number of filters in each SepConv layer. len(depth_list) == 3
            prefix: prefix before name
            skip_connection_type: one of {'conv','sum','none'}
            stride: stride at last depthwise conv
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & pointwise convs
            return_skip: flag to return additional tensor after 2 SepConvs for decoder
            """
    residual = inputs
    for i in range(3):
        residual = SepConv_BN(residual,
                              depth_list[i],
                              prefix + '_separable_conv{}'.format(i + 1),
                              stride=stride if i == 2 else 1,
                              rate=rate,
                              depth_activation=depth_activation)
        if i == 1:
            skip = residual
    if skip_connection_type == 'conv':
        shortcut = _conv2d_same(inputs, depth_list[-1], prefix + '_shortcut',
                                kernel_size=1,
                                stride=stride)
        shortcut = BatchNormalization(name=prefix + '_shortcut_BN')(shortcut)
        outputs = layers.add([residual, shortcut])
    elif skip_connection_type == 'sum':
        outputs = layers.add([residual, inputs])
    elif skip_connection_type == 'none':
        outputs = residual
    if return_skip:
        return outputs, skip
    else:
        return outputs


def relu6(x):
    return K.relu(x, max_value=6)


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id, skip_connection, rate=1):
    in_channels = inputs._keras_shape[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'expanded_conv_{}_'.format(block_id)
    if block_id:
        # Expand

        x = Conv2D(expansion * in_channels, kernel_size=1, padding='same',
                   use_bias=False, activation=None,
                   name=prefix + 'expand')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name=prefix + 'expand_BN')(x)
        x = Activation(relu6, name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'
    # Depthwise
    x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None,
                        use_bias=False, padding='same', dilation_rate=(rate, rate),
                        name=prefix + 'depthwise')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                           name=prefix + 'depthwise_BN')(x)

    x = Activation(relu6, name=prefix + 'depthwise_relu')(x)

    # Project
    x = Conv2D(pointwise_filters,
               kernel_size=1, padding='same', use_bias=False, activation=None,
               name=prefix + 'project')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                           name=prefix + 'project_BN')(x)

    if skip_connection:
        return Add(name=prefix + 'add')([inputs, x])

    # if in_channels == pointwise_filters and stride == 1:
    #    return Add(name='res_connect_' + str(block_id))([inputs, x])

    return x


# Deeplabs start here
input_shape = (128,128,3)
input_tensor = None
weights = 'None'
classes = 1
backbone = 'xception'
OS = 16

if input_tensor is None:
    img_input = Input(shape=input_shape)
else:
    if not K.is_keras_tensor(input_tensor):
        img_input = Input(tensor=input_tensor, shape=input_shape)
    else:
        img_input = input_tensor

if backbone == 'xception':
    if OS == 8:
        entry_block3_stride = 1
        middle_block_rate = 2  # ! Not mentioned in paper, but required
        exit_block_rates = (2, 4)
        atrous_rates = (12, 24, 36)
    else:
        entry_block3_stride = 2
        middle_block_rate = 1
        exit_block_rates = (1, 2)
        atrous_rates = (6, 12, 18)

    x = Conv2D(32, (3, 3), strides=(2, 2),
               name='entry_flow_conv1_1', use_bias=False, padding='same')(img_input)
    x = BatchNormalization(name='entry_flow_conv1_1_BN')(x)
    x = Activation('relu')(x)

    x = _conv2d_same(x, 64, 'entry_flow_conv1_2', kernel_size=3, stride=1)
    x = BatchNormalization(name='entry_flow_conv1_2_BN')(x)
    x = Activation('relu')(x)

    x = _xception_block(x, [128, 128, 128], 'entry_flow_block1',
                        skip_connection_type='conv', stride=2,
                        depth_activation=False)
    x, skip1 = _xception_block(x, [256, 256, 256], 'entry_flow_block2',
                               skip_connection_type='conv', stride=2,
                               depth_activation=False, return_skip=True)

    x = _xception_block(x, [728, 728, 728], 'entry_flow_block3',
                        skip_connection_type='conv', stride=entry_block3_stride,
                        depth_activation=False)
    for i in range(16):
        x = _xception_block(x, [728, 728, 728], 'middle_flow_unit_{}'.format(i + 1),
                            skip_connection_type='sum', stride=1, rate=middle_block_rate,
                            depth_activation=False)

    x = _xception_block(x, [728, 1024, 1024], 'exit_flow_block1',
                        skip_connection_type='conv', stride=1, rate=exit_block_rates[0],
                        depth_activation=False)
    x = _xception_block(x, [1536, 1536, 2048], 'exit_flow_block2',
                        skip_connection_type='none', stride=1, rate=exit_block_rates[1],
                        depth_activation=True)

else:
    OS = 8
    first_block_filters = _make_divisible(32 * alpha, 8)
    x = Conv2D(first_block_filters,
               kernel_size=3,
               strides=(2, 2), padding='same',
               use_bias=False, name='Conv')(img_input)
    x = BatchNormalization(
        epsilon=1e-3, momentum=0.999, name='Conv_BN')(x)
    x = Activation(relu6, name='Conv_Relu6')(x)

    x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1,
                            expansion=1, block_id=0, skip_connection=False)

    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2,
                            expansion=6, block_id=1, skip_connection=False)
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
                            expansion=6, block_id=2, skip_connection=True)

    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,
                            expansion=6, block_id=3, skip_connection=False)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=4, skip_connection=True)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=5, skip_connection=True)

    # stride in block 6 changed from 2 -> 1, so we need to use rate = 2
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,  # 1!
                            expansion=6, block_id=6, skip_connection=False)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=7, skip_connection=True)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=8, skip_connection=True)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=9, skip_connection=True)

    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=10, skip_connection=False)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=11, skip_connection=True)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=12, skip_connection=True)

    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=2,  # 1!
                            expansion=6, block_id=13, skip_connection=False)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
                            expansion=6, block_id=14, skip_connection=True)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
                            expansion=6, block_id=15, skip_connection=True)

    x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1, rate=4,
                            expansion=6, block_id=16, skip_connection=False)

# end of feature extractor

# branching for Atrous Spatial Pyramid Pooling

# Image Feature branch
#out_shape = int(np.ceil(input_shape[0] / OS))
b4 = AveragePooling2D(pool_size=(int(np.ceil(input_shape[0] / OS)), int(np.ceil(input_shape[1] / OS))))(x)
b4 = Conv2D(256, (1, 1), padding='same',
            use_bias=False, name='image_pooling')(b4)
b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
b4 = Activation('relu')(b4)
b4 = BilinearUpsampling((int(np.ceil(input_shape[0] / OS)), int(np.ceil(input_shape[1] / OS))))(b4)

# simple 1x1
b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
b0 = Activation('relu', name='aspp0_activation')(b0)

# there are only 2 branches in mobilenetV2. not sure why
if backbone == 'xception':
    # rate = 6 (12)
    b1 = SepConv_BN(x, 256, 'aspp1',
                    rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
    # rate = 12 (24)
    b2 = SepConv_BN(x, 256, 'aspp2',
                    rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
    # rate = 18 (36)
    b3 = SepConv_BN(x, 256, 'aspp3',
                    rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)

    # concatenate ASPP branches & project
    x = Concatenate()([b4, b0, b1, b2, b3])
else:
    x = Concatenate()([b4, b0])

x = Conv2D(256, (1, 1), padding='same',
           use_bias=False, name='concat_projection')(x)
x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
x = Activation('relu')(x)
x = Dropout(0.1)(x)

# DeepLab v.3+ decoder

if backbone == 'xception':
    # Feature projection
    # x4 (x2) block
    x = BilinearUpsampling(output_size=(int(np.ceil(input_shape[0] / 4)),
                                        int(np.ceil(input_shape[1] / 4))))(x)
    dec_skip1 = Conv2D(48, (1, 1), padding='same',
                       use_bias=False, name='feature_projection0')(skip1)
    dec_skip1 = BatchNormalization(
        name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
    dec_skip1 = Activation('relu')(dec_skip1)
    x_pred = Concatenate()([x, dec_skip1])
    x = SepConv_BN(x_pred, 256, 'decoder_conv0',
                   depth_activation=True, epsilon=1e-5)
    x = SepConv_BN(x, 256, 'decoder_conv1',
                   depth_activation=True, epsilon=1e-5)

# you can use it with arbitary number of classes
if classes == 21:
    last_layer_name = 'logits_semantic'
else:
    last_layer_name = 'custom_logits_semantic'

x = Conv2D(classes, (1, 1), padding='same', name=last_layer_name,activation="sigmoid")(x)
x = BilinearUpsampling(output_size=(input_shape[0], input_shape[1]))(x)

# Ensure that the model takes into account
# any potential predecessors of `input_tensor`.
if input_tensor is not None:
    inputs = get_source_inputs(input_tensor)
else:
    inputs = img_input

Deeplab = Model(inputs, outputs = [x_pred, x], name='deeplab')
Deeplab.summary()

# load weights

if weights == 'pascal_voc':
    if backbone == 'xception':
        weights_path = get_file('deeplabv3_xception_tf_dim_ordering_tf_kernels.h5',
                                WEIGHTS_PATH_X,
                                cache_subdir='models')
    else:
        weights_path = get_file('deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5',
                                WEIGHTS_PATH_MOBILE,
                                cache_subdir='models')
    Deeplab.load_weights(weights_path, by_name=True)
elif weights == 'cityscapes':
    if backbone == 'xception':
        weights_path = get_file('deeplabv3_xception_tf_dim_ordering_tf_kernels_cityscapes.h5',
                                WEIGHTS_PATH_X_CS,
                                cache_subdir='models')
    else:
        weights_path = get_file('deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels_cityscapes.h5',
                                WEIGHTS_PATH_MOBILE_CS,
                                cache_subdir='models')
    Deeplab.load_weights(weights_path, by_name=True)


def preprocess_input(x):
    """Preprocesses a numpy array encoding a batch of images.
    # Arguments
        x: a 4D numpy array consists of RGB values within [0, 255].
    # Returns
        Input array scaled to [-1.,1.]
    """
    return imagenet_utils.preprocess_input(x, mode='tf')


        # from model import Deeplabv3


# In[26]:


# Prep DeepLab
WEIGHTS_PATH_X = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5"
WEIGHTS_PATH_MOBILE = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5"
WEIGHTS_PATH_X_CS = "https://github.com/rdiazgar/keras-deeplab-v3-plus/releases/download/1.2/deeplabv3_xception_tf_dim_ordering_tf_kernels_cityscapes.h5"
WEIGHTS_PATH_MOBILE_CS = "https://github.com/rdiazgar/keras-deeplab-v3-plus/releases/download/1.2/deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels_cityscapes.h5"

class BilinearUpsampling(Layer):
    """Just a simple bilinear upsampling layer. Works only with TF.
       Args:
           upsampling: tuple of 2 numbers > 0. The upsampling ratio for h and w
           output_size: used instead of upsampling arg if passed!
    """

    def __init__(self, upsampling=(2, 2), output_size=None, data_format=None, **kwargs):

        super(BilinearUpsampling, self).__init__(**kwargs)

        self.data_format = K.image_data_format()
        self.input_spec = InputSpec(ndim=4)
        if output_size:
            self.output_size = conv_utils.normalize_tuple(
                output_size, 2, 'output_size')
            self.upsampling = None
        else:
            self.output_size = None
            self.upsampling = conv_utils.normalize_tuple(
                upsampling, 2, 'upsampling')

    def compute_output_shape(self, input_shape):
        if self.upsampling:
            height = self.upsampling[0] *                 input_shape[1] if input_shape[1] is not None else None
            width = self.upsampling[1] *                 input_shape[2] if input_shape[2] is not None else None
        else:
            height = self.output_size[0]
            width = self.output_size[1]
        return (input_shape[0],
                height,
                width,
                input_shape[3])

    def call(self, inputs):
        if self.upsampling:
            return K.tf.image.resize_bilinear(inputs, (inputs.shape[1] * self.upsampling[0],
                                                       inputs.shape[2] * self.upsampling[1]),
                                              align_corners=True)
        else:
            return K.tf.image.resize_bilinear(inputs, (self.output_size[0],
                                                       self.output_size[1]),
                                              align_corners=True)

    def get_config(self):
        config = {'upsampling': self.upsampling,
                  'output_size': self.output_size,
                  'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    """ SepConv with BN between depthwise & pointwise. Optionally add activation after BN
        Implements right "same" padding for even kernel sizes
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & poinwise convs
            epsilon: epsilon to use in BN layer
    """

    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = Activation('relu')(x)
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)
    x = Conv2D(filters, (1, 1), padding='same',
               use_bias=False, name=prefix + '_pointwise')(x)
    x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)

    return x


def _conv2d_same(x, filters, prefix, stride=1, kernel_size=3, rate=1):
    """Implements right 'same' padding for even kernel sizes
        Without this there is a 1 pixel drift when stride = 2
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
    """
    if stride == 1:
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='same', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='valid', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)


def _xception_block(inputs, depth_list, prefix, skip_connection_type, stride,
                    rate=1, depth_activation=False, return_skip=False):
    """ Basic building block of modified Xception network
        Args:
            inputs: input tensor
            depth_list: number of filters in each SepConv layer. len(depth_list) == 3
            prefix: prefix before name
            skip_connection_type: one of {'conv','sum','none'}
            stride: stride at last depthwise conv
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & pointwise convs
            return_skip: flag to return additional tensor after 2 SepConvs for decoder
            """
    residual = inputs
    for i in range(3):
        residual = SepConv_BN(residual,
                              depth_list[i],
                              prefix + '_separable_conv{}'.format(i + 1),
                              stride=stride if i == 2 else 1,
                              rate=rate,
                              depth_activation=depth_activation)
        if i == 1:
            skip = residual
    if skip_connection_type == 'conv':
        shortcut = _conv2d_same(inputs, depth_list[-1], prefix + '_shortcut',
                                kernel_size=1,
                                stride=stride)
        shortcut = BatchNormalization(name=prefix + '_shortcut_BN')(shortcut)
        outputs = layers.add([residual, shortcut])
    elif skip_connection_type == 'sum':
        outputs = layers.add([residual, inputs])
    elif skip_connection_type == 'none':
        outputs = residual
    if return_skip:
        return outputs, skip
    else:
        return outputs


def relu6(x):
    return K.relu(x, max_value=6)


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id, skip_connection, rate=1):
    in_channels = inputs._keras_shape[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'expanded_conv_{}_'.format(block_id)
    if block_id:
        # Expand

        x = Conv2D(expansion * in_channels, kernel_size=1, padding='same',
                   use_bias=False, activation=None,
                   name=prefix + 'expand')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name=prefix + 'expand_BN')(x)
        x = Activation(relu6, name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'
    # Depthwise
    x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None,
                        use_bias=False, padding='same', dilation_rate=(rate, rate),
                        name=prefix + 'depthwise')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                           name=prefix + 'depthwise_BN')(x)

    x = Activation(relu6, name=prefix + 'depthwise_relu')(x)

    # Project
    x = Conv2D(pointwise_filters,
               kernel_size=1, padding='same', use_bias=False, activation=None,
               name=prefix + 'project')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                           name=prefix + 'project_BN')(x)

    if skip_connection:
        return Add(name=prefix + 'add')([inputs, x])

    # if in_channels == pointwise_filters and stride == 1:
    #    return Add(name='res_connect_' + str(block_id))([inputs, x])

    return x


# Deeplabs start here
input_shape = (128,128,3)
input_tensor = None
weights = 'None'
classes = 1
backbone = 'xception'
OS = 16

if input_tensor is None:
    img_input = Input(shape=input_shape)
else:
    if not K.is_keras_tensor(input_tensor):
        img_input = Input(tensor=input_tensor, shape=input_shape)
    else:
        img_input = input_tensor

if backbone == 'xception':
    if OS == 8:
        entry_block3_stride = 1
        middle_block_rate = 2  # ! Not mentioned in paper, but required
        exit_block_rates = (2, 4)
        atrous_rates = (12, 24, 36)
    else:
        entry_block3_stride = 2
        middle_block_rate = 1
        exit_block_rates = (1, 2)
        atrous_rates = (6, 12, 18)

    x = Conv2D(32, (3, 3), strides=(2, 2),
               name='entry_flow_conv1_1', use_bias=False, padding='same')(img_input)
    x = BatchNormalization(name='entry_flow_conv1_1_BN')(x)
    x = Activation('relu')(x)

    x = _conv2d_same(x, 64, 'entry_flow_conv1_2', kernel_size=3, stride=1)
    x = BatchNormalization(name='entry_flow_conv1_2_BN')(x)
    x = Activation('relu')(x)

    x = _xception_block(x, [128, 128, 128], 'entry_flow_block1',
                        skip_connection_type='conv', stride=2,
                        depth_activation=False)
    x, skip1 = _xception_block(x, [256, 256, 256], 'entry_flow_block2',
                               skip_connection_type='conv', stride=2,
                               depth_activation=False, return_skip=True)

    x = _xception_block(x, [728, 728, 728], 'entry_flow_block3',
                        skip_connection_type='conv', stride=entry_block3_stride,
                        depth_activation=False)
    for i in range(16):
        x = _xception_block(x, [728, 728, 728], 'middle_flow_unit_{}'.format(i + 1),
                            skip_connection_type='sum', stride=1, rate=middle_block_rate,
                            depth_activation=False)

    x = _xception_block(x, [728, 1024, 1024], 'exit_flow_block1',
                        skip_connection_type='conv', stride=1, rate=exit_block_rates[0],
                        depth_activation=False)
    x = _xception_block(x, [1536, 1536, 2048], 'exit_flow_block2',
                        skip_connection_type='none', stride=1, rate=exit_block_rates[1],
                        depth_activation=True)

else:
    OS = 8
    first_block_filters = _make_divisible(32 * alpha, 8)
    x = Conv2D(first_block_filters,
               kernel_size=3,
               strides=(2, 2), padding='same',
               use_bias=False, name='Conv')(img_input)
    x = BatchNormalization(
        epsilon=1e-3, momentum=0.999, name='Conv_BN')(x)
    x = Activation(relu6, name='Conv_Relu6')(x)

    x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1,
                            expansion=1, block_id=0, skip_connection=False)

    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2,
                            expansion=6, block_id=1, skip_connection=False)
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
                            expansion=6, block_id=2, skip_connection=True)

    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,
                            expansion=6, block_id=3, skip_connection=False)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=4, skip_connection=True)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=5, skip_connection=True)

    # stride in block 6 changed from 2 -> 1, so we need to use rate = 2
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,  # 1!
                            expansion=6, block_id=6, skip_connection=False)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=7, skip_connection=True)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=8, skip_connection=True)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=9, skip_connection=True)

    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=10, skip_connection=False)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=11, skip_connection=True)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=12, skip_connection=True)

    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=2,  # 1!
                            expansion=6, block_id=13, skip_connection=False)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
                            expansion=6, block_id=14, skip_connection=True)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
                            expansion=6, block_id=15, skip_connection=True)

    x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1, rate=4,
                            expansion=6, block_id=16, skip_connection=False)

# end of feature extractor

# branching for Atrous Spatial Pyramid Pooling

# Image Feature branch
#out_shape = int(np.ceil(input_shape[0] / OS))
b4 = AveragePooling2D(pool_size=(int(np.ceil(input_shape[0] / OS)), int(np.ceil(input_shape[1] / OS))))(x)
b4 = Conv2D(256, (1, 1), padding='same',
            use_bias=False, name='image_pooling')(b4)
b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
b4 = Activation('relu')(b4)
b4 = BilinearUpsampling((int(np.ceil(input_shape[0] / OS)), int(np.ceil(input_shape[1] / OS))))(b4)

# simple 1x1
b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
b0 = Activation('relu', name='aspp0_activation')(b0)

# there are only 2 branches in mobilenetV2. not sure why
if backbone == 'xception':
    # rate = 6 (12)
    b1 = SepConv_BN(x, 256, 'aspp1',
                    rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
    # rate = 12 (24)
    b2 = SepConv_BN(x, 256, 'aspp2',
                    rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
    # rate = 18 (36)
    b3 = SepConv_BN(x, 256, 'aspp3',
                    rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)

    # concatenate ASPP branches & project
    x = Concatenate()([b4, b0, b1, b2, b3])
else:
    x = Concatenate()([b4, b0])

x = Conv2D(256, (1, 1), padding='same',
           use_bias=False, name='concat_projection')(x)
x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
x = Activation('relu')(x)
x = Dropout(0.1)(x)

# DeepLab v.3+ decoder

if backbone == 'xception':
    # Feature projection
    # x4 (x2) block
    x = BilinearUpsampling(output_size=(int(np.ceil(input_shape[0] / 4)),
                                        int(np.ceil(input_shape[1] / 4))))(x)
    dec_skip1 = Conv2D(48, (1, 1), padding='same',
                       use_bias=False, name='feature_projection0')(skip1)
    dec_skip1 = BatchNormalization(
        name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
    dec_skip1 = Activation('relu')(dec_skip1)
    x_pred = Concatenate()([x, dec_skip1])
    x = SepConv_BN(x_pred, 256, 'decoder_conv0',
                   depth_activation=True, epsilon=1e-5)
    x = SepConv_BN(x, 256, 'decoder_conv1',
                   depth_activation=True, epsilon=1e-5)

# you can use it with arbitary number of classes
if classes == 21:
    last_layer_name = 'logits_semantic'
else:
    last_layer_name = 'custom_logits_semantic'

x = Conv2D(classes, (1, 1), padding='same', name=last_layer_name,activation="sigmoid")(x)
x = BilinearUpsampling(output_size=(input_shape[0], input_shape[1]))(x)

# Ensure that the model takes into account
# any potential predecessors of `input_tensor`.
if input_tensor is not None:
    input_seg_prep = get_source_inputs(input_tensor)
else:
    input_seg_prep = img_input

Deeplab_perp = Model(input_seg_prep, outputs = [x_pred, x], name='deeplab')
Deeplab_perp.summary()

# load weights

if weights == 'pascal_voc':
    if backbone == 'xception':
        weights_path = get_file('deeplabv3_xception_tf_dim_ordering_tf_kernels.h5',
                                WEIGHTS_PATH_X,
                                cache_subdir='models')
    else:
        weights_path = get_file('deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5',
                                WEIGHTS_PATH_MOBILE,
                                cache_subdir='models')
    Deeplab_perp.load_weights(weights_path, by_name=True)
elif weights == 'cityscapes':
    if backbone == 'xception':
        weights_path = get_file('deeplabv3_xception_tf_dim_ordering_tf_kernels_cityscapes.h5',
                                WEIGHTS_PATH_X_CS,
                                cache_subdir='models')
    else:
        weights_path = get_file('deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels_cityscapes.h5',
                                WEIGHTS_PATH_MOBILE_CS,
                                cache_subdir='models')
    Deeplab_perp.load_weights(weights_path, by_name=True)


def preprocess_input(x):
    """Preprocesses a numpy array encoding a batch of images.
    # Arguments
        x: a 4D numpy array consists of RGB values within [0, 255].
    # Returns
        Input array scaled to [-1.,1.]
    """
    return imagenet_utils.preprocess_input(x, mode='tf')


# In[27]:


def denorm(image):
    image  = (image +1)/2
    image = image*255.
    return image


# In[28]:


encoder_out,edge_input = encoder([encoder_in,epsilon])
decoder_out = decoder([encoder_out,edge_input])
decoder_out_unorm = Lambda(denorm , name = 'vae_decoder_lam')(decoder_out)


# In[31]:


#load Model
Deeplab.load_weights('../../Retrain/models/Deeplabs_oPredLSPredfl075.h5')
Deeplab.trainable = False
perp_out_pred , seg_out = Deeplab(decoder_out_unorm)


# In[32]:


# Unet Perp
Deeplab_perp.load_weights('../../Retrain/models/Deeplabs_oPredLSPredfl075.h5')
Deeplab_perp.trainable = False


# In[33]:


skin_net = Model(inputs = [encoder_in,epsilon],outputs = [decoder_out,perp_out_pred])
# skin_net = multi_gpu_model(skin_net, gpus=2)
skin_net.summary()


# In[34]:


# get ground truth for perceptual loss
X_train_perp , _ = Deeplab_perp.predict(X_train , verbose = 1)
X_valid_perp , _ = Deeplab_perp.predict(X_valid , verbose = 1)
X_test_perp , _ = Deeplab_perp.predict(X_test , verbose = 1)


# In[35]:


print(np.shape(X_train_perp) , np.shape(X_valid_perp) , np.shape(X_test_perp))


# In[36]:


def reconstruction_loss(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred)) + K.mean(K.abs(y_true - y_pred))

def kl_loss(y_true, y_pred):
    kl_loss = - 0.5 * K.mean(1 + log_var - K.square(mu) - K.exp(log_var), axis=-1)
    return kl_loss

def vae_loss(y_true, y_pred):
    return 10* reconstruction_loss(y_true, y_pred) +  5 * kl_loss(y_true, y_pred) 


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


# In[41]:


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

def perceptual_loss(y_true , y_pred):
    return K.mean(K.square(y_true - y_pred))

def perceptual_loss_mae(y_true , y_pred):
    return K.mean(K.abs(y_true - y_pred))


# In[42]:


def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

#         return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
#                -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return binary_focal_loss_fixed


# In[43]:


vae_opt = keras.optimizers.RMSprop(lr=0.0001 , decay=1e-3)


# In[44]:


skin_net.compile(loss={'vae_decoder': vae_loss, 
                       'deeplab': perceptual_loss
                      },
              loss_weights={'vae_decoder': 1.0,
                            'deeplab': 2.0
                           },
              optimizer= vae_opt,
              metrics={'vae_decoder': [reconstruction_loss, kl_loss]
                       ,'deeplab':[perceptual_loss]
                      })


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
                                          {'vae_decoder':validation_data,
                                          'deeplab':validation_targets},
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


# In[21]:


NIR_history = AdditionalValidationSets([(X_test, y_test, 'NIR')])


# In[22]:


callbacks = [
    EarlyStopping(patience=25, verbose=1,monitor='val_loss'),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=1e-9, verbose=1,monitor='val_loss'),
    NIR_history,
    ModelCheckpoint('./models/Vae_5Deeplabs_frez_oCompNIRfl0d75.h5', verbose=1, save_best_only=True, save_weights_only=True)
]


# In[23]:


history = skin_net.fit(X_train,
                    {'vae_decoder': X_train
                     , 'deeplab':y_train
                    },
                    batch_size=8,
                    epochs=100,
                    callbacks=callbacks,
                    validation_data=(X_valid, {'vae_decoder': X_valid
                                               , 'deeplab': y_valid
                                              }))


# In[45]:


skin_net.load_weights('../../PL/models/vae10l5kl_2oPredLSPredfl075PL-m4.h5')


# In[28]:


plt.figure(figsize=(16,4))
plt.subplot(1,2,1)
plt.plot(history.history['mean_iou'][1:])
plt.plot(history.history['val_mean_iou'][1:])
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


# In[46]:


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


# In[47]:


# model predict on train and test 
(train_vae_pred , _) = skin_net.predict(X_train , verbose = 1)
(valid_vae_pred , _)  = skin_net.predict(X_valid , verbose = 1)
(test_vae_pred , _)  = skin_net.predict(X_test , verbose = 1)


# In[48]:


show_vae_image(X_train , train_vae_pred)
show_vae_image(X_valid ,valid_vae_pred)
show_vae_image(X_test ,test_vae_pred )


# In[49]:


def Segment_Pred(model , train , valid , test , de_norm = False):
    if not de_norm:
        _ , train_seg = model.predict(denorm(train) , verbose = 1)
        _ , valid_seg = model.predict(denorm(valid) , verbose = 1)
        _ , test_seg = model.predict(denorm(test) , verbose = 1)
    
    else:
        _ , train_seg = model.predict(denorm(train) , verbose = 1)
        _ , valid_seg = model.predict(denorm(valid) , verbose = 1)
        _ , test_seg = model.predict(denorm(test) , verbose = 1)
        
    train_t = (train_seg  > 0.5).astype(np.uint8)
    valid_t = (valid_seg > 0.5).astype(np.uint8)
    test_t = (test_seg > 0.5).astype(np.uint8)
    return train_seg , valid_seg , test_seg , train_t , valid_t , test_t


# In[50]:


def evaluate(true_train , pred_train, true_valid ,pred_valid ,true_test , pred_test):
    iou = mean_iou(true_train , pred_train )
    train_iou = K.print_tensor(iou)
    train_iou = K.eval(train_iou)
    print("Train IOU",train_iou)
    
    iou = mean_iou(true_valid , pred_valid )
    valid_iou = K.print_tensor(iou)
    valid_iou = K.eval(valid_iou)
    print("Valid IOU" , valid_iou)
    
    iou = mean_iou(true_test , pred_test )
    test_iou = K.print_tensor(iou)
    test_iou = K.eval(test_iou)
    print("Test IOU" , test_iou)


# In[51]:


train_seg_true , valid_seg_true , test_seg_true , train_seg_true_t , valid_seg_t , test_seg_t = Segment_Pred(Deeplab ,
                                                                                                            X_train ,
                                                                                                            X_valid,
                                                                                                            X_test)
# evaluate 
evaluate(y_train , train_seg_true , y_valid , valid_seg_true , y_test , test_seg_true)
train_seg_pred , valid_seg_pred , test_seg_pred , train_seg_pred_t , valid_seg_pred_t , test_seg_pred_t = Segment_Pred(Deeplab ,
                                                                                                            train_vae_pred ,
                                                                                                            valid_vae_pred,
                                                                                                            test_vae_pred)

# evaluate 
evaluate(y_train , train_seg_pred , y_valid , valid_seg_pred , y_test , test_seg_pred)


# In[54]:


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


# In[58]:


# Check if training data looks all right
plot_sample(X_train, y_train, train_seg_pred, train_seg_pred_t, ix=14)

# Check if valid data looks all right
plot_sample(X_valid, y_valid, valid_seg_pred, valid_seg_pred_t, ix=19)

# Check if valid data looks all right
plot_sample(X_test, y_test, test_seg_pred, test_seg_pred_t, ix=45)


# In[57]:


def show_vae_image_latent(x_data , predictions ,x_star, image_size = 128):
    n = 5  # how many digits we will display
    plt.figure(figsize=(10, 6))
    for i in range(n):
        # display original
        ax = plt.subplot(3, n, i + 1)
        orig = x_data[i,:,:,:].reshape(image_size,image_size,3)
        orig = (orig +1)/2
        plt.imshow((255*orig).astype('uint8'))
#         plt.imshow(x_data[i].reshape(image_size, image_size,3))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(3, n, i + 1 + n)
        pred = predictions[i,:,:,:].reshape(image_size,image_size,3)
        pred = (pred + 1)/2
        plt.imshow((255*pred).astype('uint8'))
#         plt.imshow(predictions[i].reshape(image_size, image_size,3))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        ax = plt.subplot(3, n, i + 1 + 2*n)
        star_pred = x_star[i,:,:,:].reshape(image_size,image_size,3)
        star_pred = (star_pred + 1)/2
        plt.imshow((255*star_pred).astype('uint8'))
#         plt.imshow(predictions[i].reshape(image_size, image_size,3))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


# In[59]:


# Adam 
# latent search 


y_true = K.placeholder( shape=(None, 128,128,3))
z_tensor = K.placeholder(shape=(None,64))
x_edge = tf.image.sobel_edges(y_true)
x_edge_tensor = concatenate([x_edge[:,:,:,:,0],x_edge[:,:,:,:,1]],axis = -1)
y_pred = decoder([z_tensor ,x_edge_tensor ])
mse = K.mean(K.square(((y_true + 1)*0.5) - ((y_pred + 1)*0.5)))
ssim = (1 - tf.image.ssim((y_pred + 1 )*0.5, (y_true + 1)*0.5, 1.0))
# gradients = K.gradients(ssim, z_tensor)[0]
gradients = K.gradients(mse, z_tensor)[0]
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
# sess.run(tf.global_variables_initializer())
# sess.run(tf.variables_initializer()
steps = 0
batch_Num = 0
gamma = 0.9
gradient_iter = 11
batch_size = 64
# learning_rate = 10
learning_rate = 0.01
beta1 = 0.9
beta2 = 0.99

skin_net.load_weights('../../PL/models/vae10l5kl_2oPredLSPredfl075PL-m4.h5')
skin_net.trainable = False
encoder.trainable = False
decoder.trainable = False
mean_iou_list = []
img_size = 128
prediction_xstar = np.zeros((704, img_size, img_size, 3))
for i in range(gradient_iter):
    next_x_seg = y_test[(batch_Num ) * batch_size : ((batch_Num) + 1) * batch_size]
    next_x_images = X_test[(batch_Num) * batch_size : ((batch_Num) + 1) * batch_size]
#     print(next_x_images.shape , next_x_seg.shape)
    V = np.zeros([batch_size,64])
    M = np.zeros([batch_size , 64])
    #change here
    print("Batch:{}".format(batch_Num))
    z_grad_ascent = np.random.normal(0,1,size = (batch_size,64))
    z_grad_ascent = np.reshape(z_grad_ascent , (batch_size,64))
    test_vae_pred_batch = test_vae_pred[(batch_Num) * batch_size : ((batch_Num) + 1) * batch_size]
    for j in range(1):
        for k in tqdm_notebook(range(50)):
            index  = (k+1)
            grad ,z_edge , z_ssim , z_mse = sess.run([gradients ,x_edge_tensor,ssim ,mse ] ,feed_dict={z_tensor:z_grad_ascent , y_true:next_x_images })
            grad = np.asarray(grad)
            grad = np.reshape(grad, (batch_size,64))
            M = beta1 * M + (1 - beta1)*grad
            Mt = M / (1 - beta1**index)
            V = beta2* V  + (1-beta2)*(grad**2)
            Vt = V / (1-beta2**index)
            
            z_ssim = K.print_tensor(z_ssim)
            z_ssim = K.eval(z_ssim)
            z_mse = K.print_tensor(z_mse)
            z_mse = K.eval(z_mse)
            z_grad_ascent = z_grad_ascent - learning_rate * (Mt/(np.sqrt(Vt) + 0.000000001))
            print("Grad:{} , ssim:{} , mse:{}".format(np.sum(np.abs(grad)), np.sum(np.abs(z_ssim)) , z_mse))
        x_star = sess.run(y_pred , feed_dict = {z_tensor:z_grad_ascent , y_true:next_x_images })
#         print(np.max(x_star),np.min(x_star))
        show_vae_image_latent(next_x_images,test_vae_pred_batch ,x_star )
        x_star_denorm = denorm(x_star)
        _ , xstar_seg = Deeplab.predict(x_star_denorm)
        iou = mean_iou(next_x_seg , xstar_seg)
        z_iou = K.print_tensor(iou)
        z_iou = K.eval(z_iou)
        print("iou:{}".format(z_iou))
        mean_iou_list.append(z_iou)
        preds_xstar_bin = (xstar_seg > 0.5).astype(np.uint8)
        plot_sample(next_x_images, next_x_seg, xstar_seg, preds_xstar_bin, ix=10)
    prediction_xstar[(batch_Num) * batch_size : ((batch_Num) + 1) * batch_size] = x_star
    batch_Num = batch_Num + 1
    
    
print("Mean Iou:{}".format(np.sum(mean_iou_list)/batch_Num))

