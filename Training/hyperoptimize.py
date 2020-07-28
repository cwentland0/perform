# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 22:59:43 2020

@author: ashis
"""

#hyper parameter optimization using HyperOpt

from hyperopt import Trials, STATUS_OK, tpe, rand
from tensorflow.keras import regularizers, models, optimizers
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras
from keras.utils import np_utils
import numpy as np
import scipy.io as sio
import sklearn.metrics
from sklearn.metrics import accuracy_score
from hyperas import optim
from hyperas.distributions import choice, uniform, randint
from sklearn import model_selection
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import Conv1D,Conv2D,Conv2DTranspose,MaxPooling1D,MaxPooling2D,Flatten,Reshape,LeakyReLU,UpSampling1D,UpSampling2D,BatchNormalization



def data():


    file_path = './Scaled Dataset/'
    
    sol_FOM_scaled = np.load(file_path+'solPrim_FOM_scaled.npy')

    U_1 = sol_FOM_scaled[:,0,:].T
    U_2 = sol_FOM_scaled[:,1,:].T
    U_3 = sol_FOM_scaled[:,2,:].T
    U_4 = sol_FOM_scaled[:,3,:].T
    
    training_snapshots = np.zeros((sol_FOM_scaled.shape[2],sol_FOM_scaled.shape[0],1))
    training_snapshots[:,:,0] = U_4[:,:]
    
    #splitting into training, validation and test sets
    X_train, X_test, y_train, y_test = model_selection.train_test_split(training_snapshots, training_snapshots, test_size=0.1, shuffle='False')
    X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train, X_train, test_size=0.1, shuffle='True')

    X_train = np.expand_dims(X_train,1)
    y_train = np.expand_dims(y_train,1)
    X_val = np.expand_dims(X_val,1)
    y_val = np.expand_dims(y_val,1)
    X_test = np.expand_dims(X_test,1)   
    y_test = np.expand_dims(y_test,1)



    return X_train, X_val, X_test, y_train, y_val, y_test


def create_model(X_train, y_train, X_val, y_val,X_test, y_test):
    
    input_shape = X_train.shape
    
    
    #controls
    #n_layers = {{choice([5,10,20,30,40,50,60,70,80,90,100])}}
    
    
    #n_fl = {{choice([1,2,3,4])}}
    
    #k_s = {{choice([1,2,3,4,5,6])}}
        
    n_epochs = 20
    learr = 1e-3
    #encoder model
    
    #defining sensitivity study parameters     
    
    #input = Input() 
    input = Input(shape=(1,input_shape[2],1))
    #x = Reshape(target_shape=(input_shape[1],3))(input)
    #convolution block-1 
    #x = Conv1D(n_layers, 5, strides=1, padding='same', data_format='channels_last', dilation_rate=1, activation=None, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(input)
    x = Conv2D(int({{uniform(1,100)}}), (1,{{choice([1,2,3,4,5,6])}}), strides=1, padding='same', data_format='channels_last', dilation_rate=1, activation=None, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(input)
    #x = MaxPooling1D(pool_size=2, strides=2, padding='same', data_format='channels_last')(x)
    x = MaxPooling2D(pool_size=(1,2), strides=2, padding='same', data_format='channels_last')(x)
    x = tensorflow.keras.layers.ReLU()(x)
    
    #convolution block-2
    #x = Conv1D(n_layers, 5, strides=1, padding='same', data_format='channels_last', dilation_rate=1, activation=None, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(x)
    x = Conv2D(int({{uniform(1,100)}}), (1,{{choice([1,2,3,4,5,6])}}), strides=1, padding='same', data_format='channels_last', dilation_rate=1, activation=None, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(x)
    #x = MaxPooling1D(pool_size=2, strides=2, padding='same', data_format='channels_last')(x)
    x = MaxPooling2D(pool_size=(1,2), strides=2, padding='same', data_format='channels_last')(x)
    x = tensorflow.keras.layers.ReLU()(x)
    
    #convolution block-3
    #x = Conv1D(n_layers, 5, strides=1, padding='same', data_format='channels_last', dilation_rate=1, activation=None, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(x)
    #x = Conv2D(n_layers, (1,5), strides=1, padding='same', data_format='channels_last', dilation_rate=1, activation=None, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(x)
    #x = MaxPooling1D(pool_size=2, strides=2, padding='same', data_format='channels_last')(x)
    #x = MaxPooling2D(pool_size=(1,2), strides=2, padding='same', data_format='channels_last')(x)
    #x = tensorflow.keras.layers.ReLU()(x)
    
    #convolutional layer
    #x = Conv1D(int(3), 5, strides=1, padding='same', data_format='channels_last', dilation_rate=1, activation=None, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(x)
    x = Conv2D({{choice([1,2,3,4])}}, (1,{{choice([1,2,3,4,5,6])}}), strides=1, padding='same', data_format='channels_last', dilation_rate=1, activation=None, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(x)
    x = tensorflow.keras.layers.ReLU()(x)
    
    #flatten
    #x = Flatten()(x)
    
    encoder_PC = models.Model(input,x)
    encoder_PC.compile(optimizer=optimizers.Adam(lr=1e-4), loss='mean_squared_error',metrics=['accuracy'])
    output_shape = encoder_PC.layers[-1].output_shape    
    shape_before_flatten = encoder_PC.layers[-1].input_shape
    shape_after_flatten = encoder_PC.layers[-1].output_shape
    
    #decoder model 
    
    input = Input(batch_shape=(1,1,output_shape[2],output_shape[3]))
    #x = Reshape(target_shape=(shape_before_flatten[1],shape_before_flatten[2]))(input)
    
    #transposed convolutions block-1
    #x = Conv1D(n_layers,5, strides=1, padding='same', data_format='channels_last', dilation_rate=1, activation=None, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(input)
    x = Conv2DTranspose(int({{uniform(1,100)}}), (1,{{choice([1,2,3,4,5,6])}}), strides=(1,2), padding='same', data_format='channels_last', dilation_rate=1, activation=None, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(input)
    x = tensorflow.keras.layers.ReLU()(x)
    #x = UpSampling1D(size=2)(x)
    #x = UpSampling2D(size=(1,2))(x)
    #transposed convolutions block-2
    #x = Conv1D(n_layers, 5, strides=1, padding='same', data_format='channels_last', dilation_rate=1, activation=None, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(x)
    x = Conv2DTranspose(int({{uniform(1,100)}}), (1,{{choice([1,2,3,4,5,6])}}), strides=(1,2), padding='same', data_format='channels_last', dilation_rate=1, activation=None, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(x)
    x = tensorflow.keras.layers.ReLU()(x)
    #x = UpSampling1D(size=2)(x)
    #x = UpSampling2D(size=(1,2))(x)
    
    #transposed convolutions block-3
    #x = Conv1D(n_layers, 5, strides=1, padding='same', data_format='channels_last', dilation_rate=1, activation=None, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(x)
    #x = Conv2DTranspose(n_layers, (1,5), strides=(1,2), padding='same', data_format='channels_last', dilation_rate=1, activation=None, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(x)
    #x = tensorflow.keras.layers.ReLU()(x)
    #x = UpSampling1D(size=2)(x)
    #x = UpSampling2D(size=(1,2))(x)
     
    #transposed convolutions
    #x = Conv1D(3, 5, strides=1, padding='same', data_format='channels_last', dilation_rate=1, activation=None, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(x)
    x = Conv2D(1, (1,{{choice([1,2,3,4,5,6])}}), strides=1, padding='same', data_format='channels_last', dilation_rate=1, activation=None, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(x)
    x = tf.keras.layers.ReLU()(x)
    #x = Reshape(target_shape=(input_shape[1],3))(x)
    
    decoder_PC = models.Model(input,x)
    decoder_PC.compile(optimizer=optimizers.Adam(lr=1e-3), loss='mean_squared_error',metrics=['accuracy'])
    
    input = Input(shape=(1,input_shape[2],1))
    autoencoder_PC = models.Model(input,decoder_PC(encoder_PC(input)))
    autoencoder_PC.compile(optimizer=optimizers.Adam(lr=learr), loss='mean_squared_error',metrics=['accuracy'])
    
    # Fit the model
    #history=autoencoder.fit(training, training, epochs=2000, batch_size=input_shape[0], callbacks=callbacks_list, verbose=1)
    cs = tensorflow.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.5e-7, patience=3000, verbose=1, mode='min', baseline=None, restore_best_weights=True)  
    history = autoencoder_PC.fit(x=X_train, y=y_train, batch_size=32, epochs=n_epochs, verbose=2, callbacks=[cs], validation_data = (X_val,y_val), shuffle=True, validation_freq=1)

    


    score, acc = autoencoder_PC.evaluate(X_test, y_test, verbose=0)
    
    
    return {'loss': -acc, 'status': STATUS_OK, 'model': autoencoder_PC}




#[a,b,c,d] = data()


best_run, best_model = optim.minimize(model=create_model,
                                      data=data,
                                      algo=tpe.suggest,
                                      max_evals=15,
                                      trials=Trials())
X_train, X_val, X_test, y_train, y_val, y_test = data()
print("Evalutation of best performing model:")
print(best_model.evaluate(X_test, y_test))
print("Best performing model chosen hyper-parameters:")
print(best_run)


#saving the models

encoder_PC = best_model.layers[-2]
decoder_PC = best_model.layers[-1]
autoencoder_PC = best_model


# autoencoder_PC.save('./Hyperopt models/Amp Inf/autoencoder_4.h5')
# encoder_PC.save('./Hyperopt models/Amp Inf/encoder_4.h5')
# decoder_PC.save('./Hyperopt models/Amp Inf/decoder_4.h5')
