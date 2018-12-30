import keras as K
import numpy as np, json

from keras.utils.np_utils import to_categorical

from keras.engine.topology import get_source_inputs
from keras.applications import imagenet_utils
from keras.utils import conv_utils
from keras.utils.data_utils import get_file
from keras.utils import plot_model

from keras.backend import tf as ktf
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, AveragePooling2D
from keras.layers import SeparableConv2D, MaxPooling2D, UpSampling2D, Conv2D, Dropout
from keras.layers import Lambda, Input, BatchNormalization, Concatenate, ZeroPadding2D, Add, Multiply
from keras.models import Model, Sequential, model_from_json
from keras.layers.advanced_activations import PReLU, LeakyReLU
#from keras.applications import ResNet50, VGG19, InceptionV3, MobileNet

from config import *
from helper_custom_layers import *


def down(filters, downKernel, input_):
    down_ = Conv2D(filters, (downKernel, downKernel), padding='same')(input_)
    #down_ = BatchNormalization(epsilon=1e-4,axis=-1)(down_)
    down_ = Activation('relu')(down_)
    down_ = Conv2D(filters, (downKernel, downKernel), padding='same')(down_)
    #down_ = BatchNormalization(epsilon=1e-4,axis=-1)(down_)
    down_res = Activation('relu')(down_)
    down_pool = MaxPooling2D((2, 2), strides=(2, 2))(down_)
    return down_pool


def resDown(filters, downKernel, input_,):
    down_ = Conv2D(filters, downKernel, padding='same', kernel_initializer='he_uniform', kernel_regularizer=K.regularizers.l2(L2PENALTY))(input_)
    down_ = LeakyReLU(alpha=0.03)(down_)

    down_ = Conv2D(filters, downKernel, padding='same', kernel_initializer='he_uniform', kernel_regularizer=K.regularizers.l2(L2PENALTY))(down_)
    down_res = LeakyReLU(alpha=0.03)(down_)
    
    down_pool = MaxPooling2D((2, 2), strides=(2, 2))(down_res)

    return down_pool, down_res


def resUp(filters, upKernel, input_, down_):
    upsample_ = UpSampling2D(size=(2, 2))(input_)

    up_ = Conv2D(filters, upKernel, padding='same', kernel_initializer='he_uniform', kernel_regularizer=K.regularizers.l2(L2PENALTY))(upsample_)
    up_ = LeakyReLU(alpha=0.03)(up_)
    up_ = Concatenate(axis=-1)([down_, up_])

    up_ = Conv2D(filters, upKernel, padding='same', kernel_initializer='he_uniform', kernel_regularizer=K.regularizers.l2(L2PENALTY))(up_)
    up_ = LeakyReLU(alpha=0.03)(up_)

    #up_ = Conv2D(filters, upKernel, padding='same', kernel_initializer='he_uniform')(up_)
    # up_ = BatchNormalization(epsilon=1e-4,axis=-1)(up_)
    # up_ = Activation('relu')(up_)

    return up_

def UNet2048():
    downKernel = 3;
    upKernel = 3;
    
    inputs = Input(shape=(W,H,1))

    #proceed with unet architecture
    down0, down0_res = resDown(FILTER1//2, downKernel, inputs)
    down1, down1_res = resDown(FILTER2//2, downKernel, down0)
    down2, down2_res = resDown(FILTER3//2, downKernel, down1)
    down3, down3_res = resDown(FILTER4//2, downKernel, down2)
    down4, down4_res = resDown(FILTER5//2, downKernel, down3)
    down5, down5_res = resDown(FILTER6//2, downKernel, down4)
    down6, down6_res = resDown(FILTER7//2, downKernel, down5)

    print(down5.shape)
    center = Conv2D(1024, (1, 1), padding='same', kernel_regularizer=K.regularizers.l2(L2PENALTY))(down6)
    center = Activation('relu')(center)
    print(center.shape)

    up6 = resUp(FILTER7//2, upKernel, center, down6_res)
    up5 = resUp(FILTER6//2, upKernel, up6, down5_res)
    up4 = resUp(FILTER5//2, upKernel, up5, down4_res)
    up3 = resUp(FILTER4//2, upKernel, up4, down3_res)
    up2 = resUp(FILTER3//2, upKernel, up3, down2_res)
    up1 = resUp(FILTER2//2, upKernel, up2, down1_res)
    up0 = resUp(FILTER1//2, upKernel, up1, down0_res)


    if NUMCLASSES==1:
        organMask = Conv2D(NUMCLASSES, (1, 1), activation='sigmoid', name='organ_output', kernel_initializer='glorot_uniform')(up0)
    else:
        organMask = Conv2D(NUMCLASSES, (1, 1), activation='softmax', name='organ_output', kernel_initializer='glorot_uniform')(up0)

    model = Model(inputs=inputs, outputs=[organMask],name="model");

    return model


def UNet1024():
    downKernel = 3;
    upKernel = 3;
    
    inputs = Input(shape=(W,H,1))

    #proceed with unet architecture
    down0, down0_res = resDown(FILTER1, downKernel, inputs)
    down1, down1_res = resDown(FILTER2, downKernel, down0)
    down2, down2_res = resDown(FILTER3, downKernel, down1)
    down3, down3_res = resDown(FILTER4, downKernel, down2)
    down4, down4_res = resDown(FILTER5, downKernel, down3)
    down5, down5_res = resDown(FILTER6, downKernel, down4)

    print(down5.shape)
    center = Conv2D(1024, (1, 1), padding='same')(down5)
    #center = BatchNormalization(epsilon=1e-4)(center)
    center = Activation('relu')(center)
    print(center.shape)

    up5 = resUp(FILTER6, upKernel, center, down5_res)
    up4 = resUp(FILTER5, upKernel, up5, down4_res)
    up3 = resUp(FILTER4, upKernel, up4, down3_res)
    up2 = resUp(FILTER3, upKernel, up3, down2_res)
    up1 = resUp(FILTER2, upKernel, up2, down1_res)
    up0 = resUp(FILTER1, upKernel, up1, down0_res)


    if NUMCLASSES==1:
        organMask = Conv2D(NUMCLASSES, (1, 1), activation='sigmoid', name='organ_output', kernel_initializer='glorot_uniform')(up0)
    else:
        organMask = Conv2D(NUMCLASSES, (1, 1), activation='softmax', name='organ_output', kernel_initializer='glorot_uniform')(up0)

    model = Model(inputs=inputs, outputs=[organMask],name="model");

    return model

def UNetTest():
    
    downKernel = 3;
    upKernel = 3;
    
    inputs = Input(shape=(W,H,1))

    #proceed with unet architecture
    down0, down0_res = resDown(FILTER1//2, downKernel, inputs,'relu')
    down1, down1_res = resDown(FILTER2//2, downKernel, down0,'relu')
    down2, down2_res = resDown(FILTER3//2, downKernel, down1,'relu')
    down3, down3_res = resDown(FILTER4//2, downKernel, down2,'relu')    
    center = Conv2D(512, (2, 2), padding='same')(down3)
    center = Activation('relu')(center)
    up3 = resUp(FILTER4//2, upKernel, center, down3_res,'relu')
    up2 = resUp(FILTER3//2, upKernel, up3, down2_res,'relu')
    up1 = resUp(FILTER2//2, upKernel, up2, down1_res,'relu')
    up0 = resUp(FILTER1//2, upKernel, up1, down0_res,'relu')


    if NUMCLASSES==1:
        organMask = Conv2D(NUMCLASSES, (1, 1), activation='sigmoid', name='organ_output', kernel_initializer='glorot_uniform')(up0)
    else:
        organMask = Conv2D(NUMCLASSES, (1, 1), activation='softmax', name='organ_output', kernel_initializer='glorot_uniform')(up0)

    model = Model(inputs=inputs, outputs=[organMask],name="model");

    return model

def UNet512():
    downKernel = 3;
    upKernel = 3;
    
    inputs = Input(shape=(W,H,1))

    #proceed with unet architecture
    down0, down0_res = resDown(FILTER1, downKernel, inputs)
    down1, down1_res = resDown(FILTER2, downKernel, down0)
    down2, down2_res = resDown(FILTER3, downKernel, down1)
    down3, down3_res = resDown(FILTER4, downKernel, down2)
    down4, down4_res = resDown(FILTER5, downKernel, down3)

    center = Conv2D(512, (3, 3), padding='same')(down4)
    #center = BatchNormalization(epsilon=1e-4)(center)
    center = Activation('relu')(center)
    #center = Dropout(0.5)(center)
    center = Conv2D(512, (3, 3), padding='same')(center)
    #center = BatchNormalization(epsilon=1e-4)(center)
    center = Activation('relu')(center)

    up4 = resUp(FILTER5, upKernel, center, down4_res)
    up3 = resUp(FILTER4, upKernel, up4, down3_res)
    up2 = resUp(FILTER3, upKernel, up3, down2_res)
    up1 = resUp(FILTER2, upKernel, up2, down1_res)
    up0 = resUp(FILTER1, upKernel, up1, down0_res)


    if NUMCLASSES==1:
        organMask = Conv2D(NUMCLASSES, (1, 1), activation='sigmoid', name='organ_output', kernel_initializer='glorot_uniform')(up0)
    else:
        organMask = Conv2D(NUMCLASSES, (1, 1), activation='softmax', name='organ_output', kernel_initializer='glorot_uniform')(up0)

    model = Model(inputs=inputs, outputs=[organMask],name="model");

    return model

def STNUnet():
    downKernel = 3;
    upKernel = 2;
    
    inputs = Input(shape=(H,W,1))

    #apply spatial transformation first
    locnet = transform_net(inputs);
    bodyZoom = BilinearInterpolation(output_size=(H,W),invert_matrix=0,name="body_output")([inputs, locnet])

    #proceed with unet architecture
    down0, down0_res = resDown(FILTER1, downKernel, bodyZoom)
    down1, down1_res = resDown(FILTER2, downKernel, down0)
    down2, down2_res = resDown(FILTER3, downKernel, down1)
    down3, down3_res = resDown(FILTER4, downKernel, down2)
    down4, down4_res = resDown(FILTER5, downKernel, down3)

    center = Conv2D(512, (3, 3), padding='same')(down4)
    center = BatchNormalization(epsilon=1e-4)(center)
    center = Activation('relu')(center)
    center = Conv2D(512, (3, 3), padding='same')(center)
    center = BatchNormalization(epsilon=1e-4)(center)
    center = Activation('relu')(center)

    up4 = resUp(FILTER5, upKernel, center, down4_res)
    up3 = resUp(FILTER4, upKernel, up4, down3_res)
    up2 = resUp(FILTER3, upKernel, up3, down2_res)
    up1 = resUp(FILTER2, upKernel, up2, down1_res)
    up0 = resUp(FILTER1, upKernel, up1, down0_res)


    if NUMCLASSES==1:
        organMask = Conv2D(NUMCLASSES, (1, 1), activation='sigmoid', name='organ_output')(up0)
    else:
        organMask = Conv2D(NUMCLASSES, (1, 1), activation='softmax', name='organ_output')(up0)

    #apply inverse tranformation
    up0 = BilinearInterpolation(output_size=(H,W),invert_matrix=1,trainable=False)([organMask,locnet]);
    model2 = Model(inputs=inputs, outputs=[bodyZoom,organMask],name="model");

    return model2



