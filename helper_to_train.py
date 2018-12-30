import numpy as np
import json
from helper_to_models import *
from config import *

import tensorflow as tf
import keras as K
from keras.models import load_model, model_from_json
from keras.utils.np_utils import to_categorical
from keras.losses import categorical_crossentropy


def soft_dice_loss(y_true, y_pred, epsilon=1e-6):

    # skip the batch and class axis for calculating Dice score
    numerator = 2. * K.backend.sum(y_pred * y_true, axis=(1,2))
    denominator = K.backend.sum(K.backend.square(y_pred) + K.backend.square(y_true), axis = (1,2));

    # average over classes and batch
    return 1 - K.backend.mean(numerator / (denominator + epsilon))


def tversky_score(y_true, y_pred, alpha = 0.5, beta = 0.5):

    true_positives = y_true * y_pred;
    false_negatives = y_true * (1 - y_pred);
    false_positives = (1 - y_true) * y_pred;

    num = K.backend.sum(true_positives, axis = (0,1,2)) #compute loss per-batch
    den = num+alpha*K.backend.sum(false_negatives, axis = (0,1,2)) + beta*K.backend.sum(false_positives, axis=(0,1,2))+1
    T = K.backend.mean(num/den)

    return T

def weighted_cross_entropy_bad(y_true,y_pred):
    # avoid division by zero
    w = 1 / (K.backend.sum(y_true,axis=(1,2))+1); #leave batch and classes

    # find classes that don't contribute
    w_mask = tf.equal(w,1.0);
    w = tf.where(w_mask,y=w,x=tf.zeros_like(w));

    # normalize classes to one
    freq = w / tf.reduce_sum(w,axis=(-1),keep_dims=True);

    #calculate cross entropy
    crossEntropy = K.backend.sum(y_true * K.backend.log(y_pred+1e-6),axis=(1,2));

    # calculate weighted loss per class and batch
    weightedEntropy = freq * crossEntropy;
    # sum weightedEntropy over class
    weightedEntropy = - K.backend.sum(weightedEntropy,axis=(-1));
    #now average over batch
    return K.backend.mean(weightedEntropy);


def weighted_cross_entropy(y_true,y_pred):
    
    # get pre-computed class weights
    weights = K.backend.variable(CLASSWEIGHTS);

    #calculate cross entropy
    y_pred /= tf.reduce_sum(y_pred,axis=len(y_pred.get_shape())-1,keep_dims=True)
    _epsilon = tf.convert_to_tensor(K.backend.epsilon(), dtype=y_pred.dtype.base_dtype)
    #clip bad values
    y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon);
    
    # calculate weighted loss per class and batch
    weighted_losses = y_true * tf.log(y_pred) * weights
       
    return -tf.reduce_sum(weighted_losses,len(y_pred.get_shape()) - 1)


def load_json_model(modelName):
    filePath = './checkpoint/'+modelName+".json";
    fileWeight = './checkpoint/'+modelName+"_weights.h5"

    with open(filePath,'r') as fp:
        json_data = fp.read();
    model = model_from_json(json_data,custom_objects={'relu6':relu6,'BilinearUpsampling':BilinearUpsampling,'BilinearInterpolation':BilinearInterpolation})
    model.load_weights(fileWeight)

    return model


def metric_per_label(label,alpha,beta):

    if alpha==beta==1:
        def jaccard(y_true,y_pred):
            y_true = K.backend.argmax(y_true,axis=-1);
            y_pred = K.backend.argmax(y_pred,axis=-1);

            true = K.backend.cast(K.backend.equal(y_true,label),'float32');
            pred = K.backend.cast(K.backend.equal(y_pred,label),'float32');

            return tversky_score(true,pred,alpha,beta)

        return jaccard
    elif alpha==beta==0.5:
        def dice(y_true,y_pred):
            y_true = K.backend.argmax(y_true,axis=-1);
            y_pred = K.backend.argmax(y_pred,axis=-1);

            true = K.backend.cast(K.backend.equal(y_true,label),'float32');
            pred = K.backend.cast(K.backend.equal(y_pred,label),'float32');

            return tversky_score(true,pred,alpha,beta)
        return dice
    else:
        def f1(y_true,y_pred):
            y_true = K.backend.argmax(y_true,axis=-1);
            y_pred = K.backend.argmax(y_pred,axis=-1);

            true = K.backend.cast(K.backend.equal(y_true,label),'float32');
            pred = K.backend.cast(K.backend.equal(y_pred,label),'float32');
            return tversky_score(true,pred,alpha,beta)
        
        return f1


def train_model(trainGen,valGen,stepsPerEpoch,numEpochs,valSteps):

    try:
        model = load_json_model(modelName)
        print("Loading model...");
    except Exception as e:
        print(e);
        print("Creating new model...")
        #model = spatialTransformUnet();
        model = UNet1024();
        #model = UNet512();
        #model = UNet2048();
	


    losses = {
    "organ_output": "categorical_crossentropy"
    #"organ_output": weighted_cross_entropy
    }
    lossWeights = {
    "organ_output": 1.0
    }

    print(model.summary())

    

    #optimizer = K.optimizers.RMSprop(
    #        lr=0.0001, #global learning rate,
    #        rho=0.95, #exponential moving average; r = rho*initial_accumilation+(1-rho)*current_gradient
    #        epsilon=1e-6, #small constant to stabilize division by zero
    #        decay=DECAYRATE
    #        )
    optimizer = K.optimizers.Adam(
            lr = LEARNRATE, decay = DECAYRATE        
            )
    
    chiasm_dice = metric_per_label(1,alpha=0.5,beta=0.5);
    brainstem_dice = metric_per_label(2,alpha=0.5,beta=0.5);
    cord_dice = metric_per_label(3,alpha=0.5,beta=0.5);
    parotid_dice = metric_per_label(4,alpha=0.5,beta=0.5);
    mandible_dice = metric_per_label(5,alpha=0.5,beta=0.5);
    optical_dice = metric_per_label(6,alpha=0.5,beta=0.5);


    #compile model
    model.compile(optimizer=optimizer,
                    loss = losses,#tot_loss,
                    loss_weights=lossWeights,
                    metrics=[chiasm_dice,brainstem_dice,cord_dice,parotid_dice,mandible_dice,optical_dice]
                    );

    #define callbacks
    modelCheckpoint = K.callbacks.ModelCheckpoint("./checkpoint/"+modelName+"_weights.h5",
                                'val_loss',
                                verbose=1,
                                save_best_only=True,
                                save_weights_only=True,
                                mode='min', period=1)

    reduceLearningRate = K.callbacks.ReduceLROnPlateau(monitor='loss',
                                factor=0.5, patience=2,
                                verbose=1, mode='auto',
                                cooldown=1, min_lr=0)

    earlyStopping = K.callbacks.EarlyStopping(monitor='val_loss',
                                patience=3,
                                verbose=1,
                                min_delta = 0.0001,mode='min')
    validationMetric = Metrics(valGen,valSteps,BATCHSIZE);

        
    #save only model
    with open('./checkpoint/'+modelName+'.json','w') as fp:
        fp.write(model.to_json());

    #fit model and store history
    hist = model.fit_generator(trainGen,
              steps_per_epoch = stepsPerEpoch,
              epochs = numEpochs,
              class_weight = 'auto',
              validation_data = valGen,
              validation_steps = valSteps,
              verbose=1,
              callbacks=[validationMetric,modelCheckpoint])

    

    return hist
