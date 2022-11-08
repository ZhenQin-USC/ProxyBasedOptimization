import sys
sys.path.append(r'D:\Users\qinzh\Google Drive USC\MATLAB Local\Proxy Opt')
from Utility.processing_func import GenerateSets
import tensorflow as tf
from tensorflow.keras import layers
import math
import time
import numpy as np
from os.path import join
#print(tf.__version__)


# Build CNNRNN model without labeling scheme
def build_modelCNNRNN9(nneck, lr, twindow, twindow2, nfeature, ncontrol):
    inputA = tf.keras.Input(batch_shape=(None, twindow, (nfeature + ncontrol)), name='History')
    inputB = tf.keras.Input(batch_shape=(None, None, ncontrol), name='Control')
    inputstate = tf.keras.Input(batch_shape=(None, nfeature), name='Initial_State')
    is_training = tf.keras.Input(batch_shape=(None, 1), name='Is_Training')
    x1 = layers.Conv1D(8, 3, activation='relu', strides=1, padding='valid', use_bias=False)(inputA)
    # x1b = layers.Conv1D(2,3, activation = 'relu', strides = 1, padding = 'valid', use_bias=False)(x1)
    # x2 = layers.Conv1D(2,3, strides = 2, padding = 'valid', use_bias=False,activation = 'relu')(x1)
    print(x1.shape)
    x2 = layers.Flatten()(x1)
    x2a1 = layers.Dense(nneck, activation='relu', use_bias=False,
                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                        bias_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005))(x2)
    x2b1 = layers.Dense(nfeature, activation='relu', use_bias=False,
                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                        bias_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005))(x2)
    x2a2 = layers.Dense(nneck, activation='relu', use_bias=False,
                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                        bias_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005))(x2)
    x2b2 = layers.Dense(nfeature, activation='relu', use_bias=False,
                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.00001, l2=0.00005),
                        bias_regularizer=tf.keras.regularizers.l1_l2(l1=0.00001, l2=0.00005))(x2)
    z0 = layers.LSTM(nneck, activation='relu', use_bias=True, return_state=False,
                     return_sequences=True, unroll=False,
                     # kernel_initializer='zeros',
                     # recurrent_initializer='identity',
                     kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                     bias_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                     name='GRU1')(inputB, initial_state=[x2a1, x2a2])

    z = layers.LSTM(nfeature, activation='linear', use_bias=True, return_state=False,
                    return_sequences=True, unroll=False,
                    # kernel_initializer='identity',
                    # recurrent_initializer='identity',
                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                    bias_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                    name='GRU2')(z0, initial_state=[x2b1, x2b2])
    # print(z.shape)
    initial_learning_rate = lr
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=300,
        decay_rate=0.95,
        staircase=True)
    model = tf.keras.Model(inputs=[inputA, inputB, inputstate, is_training], outputs=[z, z])
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss=['mse', 'mae'], loss_weights=[1, 0], optimizer=opt, metrics=['accuracy'])
    return model

# Build CNNRNN model without labeling scheme (for CUDA)
def build_model0(nneck, lr, twindow, twindow2, nfeature, ncontrol):
    inputA = tf.keras.Input(batch_shape=(None, twindow, (nfeature + ncontrol)), name='History')
    inputB = tf.keras.Input(batch_shape=(None, None, ncontrol), name='Control')
    inputstate = tf.keras.Input(batch_shape=(None, nfeature), name='Initial_State')
    is_training = tf.keras.Input(batch_shape=(None, 1), name='Is_Training')
    x1 = layers.Conv1D(8, 3, activation='relu', strides=1, padding='valid', use_bias=False)(inputA)
    # x1b = layers.Conv1D(2,3, activation = 'relu', strides = 1, padding = 'valid', use_bias=False)(x1)
    # x2 = layers.Conv1D(2,3, strides = 2, padding = 'valid', use_bias=False,activation = 'relu')(x1)
    print(x1.shape)
    x2 = layers.Flatten()(x1)
    x2a1 = layers.Dense(nneck, activation='relu', use_bias=False,
                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                        bias_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005))(x2)
    x2b1 = layers.Dense(nfeature, activation='relu', use_bias=False,
                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                        bias_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005))(x2)
    x2a2 = layers.Dense(nneck, activation='relu', use_bias=False,
                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                        bias_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005))(x2)
    x2b2 = layers.Dense(nfeature, activation='relu', use_bias=False,
                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.00001, l2=0.00005),
                        bias_regularizer=tf.keras.regularizers.l1_l2(l1=0.00001, l2=0.00005))(x2)

    z00 = layers.LSTM(nneck, activation='tanh', use_bias=True, return_state=False,
                     return_sequences=True, unroll=False,
                     # kernel_initializer='zeros',
                     # recurrent_initializer='identity',
                     kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                     bias_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                     name='GRU1')(inputB, initial_state=[x2a1, x2a2])

    z0 = layers.LSTM(nfeature, activation='tanh', use_bias=True, return_state=False,
                    return_sequences=True, unroll=False,
                    # kernel_initializer='identity',
                    # recurrent_initializer='identity',
                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                    bias_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                    name='GRU2')(z00, initial_state=[x2b1, x2b2])

    z = layers.Dense(nfeature, activation=None, use_bias=True,
                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.00001, l2=0.00005),
                        bias_regularizer=tf.keras.regularizers.l1_l2(l1=0.00001, l2=0.00005))(z0)

    # print(z.shape)
    initial_learning_rate = lr
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=300,
        decay_rate=0.95,
        staircase=True)
    model = tf.keras.Model(inputs=[inputA, inputB, inputstate, is_training], outputs=[z, z])
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss=['mse', 'mae'], loss_weights=[1, 0], optimizer=opt, metrics=['accuracy'])
    return model


# Build CNNRNN model without labeling scheme (for CUDA)
def build_model1(nneck, lr, twindow, twindow2, nfeature, ncontrol):
    inputA = tf.keras.Input(batch_shape=(None, twindow, (nfeature + ncontrol)), name='History')
    inputB = tf.keras.Input(batch_shape=(None, None, ncontrol), name='Control')
    inputstate = tf.keras.Input(batch_shape=(None, nfeature), name='Initial_State')
    is_training = tf.keras.Input(batch_shape=(None, 1), name='Is_Training')
    x1 = layers.Conv1D(8, 3, activation='relu', strides=1, padding='valid', use_bias=False)(inputA)
    # x1b = layers.Conv1D(2,3, activation = 'relu', strides = 1, padding = 'valid', use_bias=False)(x1)
    # x2 = layers.Conv1D(2,3, strides = 2, padding = 'valid', use_bias=False,activation = 'relu')(x1)
    print(x1.shape)
    x2 = layers.Flatten()(x1)
    x2a1 = layers.Dense(nneck, activation='relu', use_bias=False,
                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                        bias_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005))(x2)
    x2b1 = layers.Dense(nfeature, activation='relu', use_bias=False,
                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                        bias_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005))(x2)
    x2a2 = layers.Dense(nneck, activation='relu', use_bias=False,
                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                        bias_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005))(x2)
    x2b2 = layers.Dense(nfeature, activation='relu', use_bias=False,
                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.00001, l2=0.00005),
                        bias_regularizer=tf.keras.regularizers.l1_l2(l1=0.00001, l2=0.00005))(x2)

    z0 = layers.LSTM(nneck, activation='tanh', use_bias=True, return_state=False,
                     return_sequences=True, unroll=False,
                     # kernel_initializer='zeros',
                     # recurrent_initializer='identity',
                     kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                     bias_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                     name='GRU1')(inputB, initial_state=[x2a1, x2a2])

    z = layers.LSTM(nfeature, activation='tanh', use_bias=True, return_state=False,
                    return_sequences=True, unroll=False,
                    # kernel_initializer='identity',
                    # recurrent_initializer='identity',
                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                    bias_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                    name='GRU2')(z0, initial_state=[x2b1, x2b2])
    # print(z.shape)
    initial_learning_rate = lr
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=300,
        decay_rate=0.95,
        staircase=True)
    model = tf.keras.Model(inputs=[inputA, inputB, inputstate, is_training], outputs=[z, z])
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss=['mse', 'mae'], loss_weights=[1, 0], optimizer=opt, metrics=['accuracy'])
    return model


# Build CNNRNN model without labeling scheme
def build_model2(nneck, lr, twindow, twindow2, nfeature, ncontrol):
    inputA = tf.keras.Input(batch_shape=(None, twindow, (nfeature + ncontrol)), name='History')
    inputB = tf.keras.Input(batch_shape=(None, None, ncontrol), name='Control')
    inputstate = tf.keras.Input(batch_shape=(None, nfeature), name='Initial_State')
    is_training = tf.keras.Input(batch_shape=(None, 1), name='Is_Training')
    label = tf.keras.Input(batch_shape=(None, None, nfeature), name='Label')
    x1 = layers.Conv1D(8, 3, activation='relu', strides=1, padding='valid', use_bias=False)(inputA)
    # x1b = layers.Conv1D(2,3, activation = 'relu', strides = 1, padding = 'valid', use_bias=False)(x1)
    # x2 = layers.Conv1D(2,3, strides = 2, padding = 'valid', use_bias=False,activation = 'relu')(x1)
    # print(x1.shape)
    x2 = layers.Flatten()(x1)
    x2a1 = layers.Dense(nneck, activation='relu', use_bias=False,
                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                        bias_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005))(x2)
    x2b1 = layers.Dense(nfeature, activation='relu', use_bias=False,
                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                        bias_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005))(x2)
    x2a2 = layers.Dense(nneck, activation='relu', use_bias=False,
                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                        bias_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005))(x2)
    x2b2 = layers.Dense(nfeature, activation='relu', use_bias=False,
                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                        bias_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005))(x2)
    z0 = layers.LSTM(nneck, activation='relu', use_bias=True, return_state=False,
                     return_sequences=True, unroll=False,
                     # kernel_initializer='zeros',
                     # recurrent_initializer='identity',
                     kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                     bias_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                     name='GRU1')(inputB, initial_state=[x2a1, x2a2])

    z01 = layers.LSTM(nfeature, activation='linear', use_bias=True, return_state=False,
                      return_sequences=True, unroll=False,
                      # kernel_initializer='identity',
                      # recurrent_initializer='identity',
                      kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                      bias_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                      name='GRU2')(z0, initial_state=[x2b1, x2b2])
    # print(label.shape, z01.shape)
    z = layers.Multiply()([label, z01])
    # print(z.shape)
    initial_learning_rate = lr
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=300,
        decay_rate=0.95,
        staircase=True)
    model = tf.keras.Model(inputs=[inputA, inputB, inputstate, is_training, label], outputs=[z, z01])
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss=['mse', 'mae'], loss_weights=[1, 0], optimizer=opt, metrics=['accuracy'])
    return model


# Build CNNRNN model without labeling scheme
def build_model3(nneck, lr, twindow, twindow2, nfeature, ncontrol):
    inputA = tf.keras.Input(batch_shape=(None, twindow, (nfeature + ncontrol)), name='History')
    inputB = tf.keras.Input(batch_shape=(None, None, ncontrol), name='Control')
    inputstate = tf.keras.Input(batch_shape=(None, nfeature), name='Initial_State')
    is_training = tf.keras.Input(batch_shape=(None, 1), name='Is_Training')
    label = tf.keras.Input(batch_shape=(None, None, nfeature), name='Label')
    x1 = layers.Conv1D(4, 3, activation='relu', strides=1, padding='valid', use_bias=False)(inputA)
    # x1b = layers.Conv1D(2,3, activation = 'relu', strides = 1, padding = 'valid', use_bias=False)(x1)
    # x2 = layers.Conv1D(2,3, strides = 2, padding = 'valid', use_bias=False,activation = 'relu')(x1)
    print(x1.shape)
    x2 = layers.Flatten()(x1)
    x2a1 = layers.Dense(nneck, activation='relu', use_bias=False,
                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                        bias_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005))(x2)
    x2b1 = layers.Dense(nfeature, activation='relu', use_bias=False,
                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                        bias_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005))(x2)
    x2a2 = layers.Dense(nneck, activation='relu', use_bias=False,
                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                        bias_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005))(x2)
    x2b2 = layers.Dense(nfeature, activation='relu', use_bias=False,
                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                        bias_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005))(x2)
    z0 = layers.LSTM(nneck, activation='relu', use_bias=True, return_state=False,
                     return_sequences=True, unroll=False,
                     # kernel_initializer='zeros',
                     # recurrent_initializer='identity',
                     kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                     bias_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                     name='GRU1')(inputB, initial_state=[x2a1, x2a2])

    z01 = layers.LSTM(nfeature, activation='linear', use_bias=True, return_state=False,
                      return_sequences=True, unroll=False,
                      # kernel_initializer='identity',
                      # recurrent_initializer='identity',
                      kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                      bias_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                      name='GRU2')(z0, initial_state=[x2b1, x2b2])
    z = layers.Multiply()([label, z01])
    # print(z.shape)
    initial_learning_rate = lr
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=300,
        decay_rate=0.95,
        staircase=True)
    model = tf.keras.Model(inputs=[inputA, inputB, inputstate, is_training, label], outputs=[z, z01])
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss=['mse', 'mae'], loss_weights=[1, 0], optimizer=opt, metrics=['accuracy'])
    return model


# Build CNNRNN model without labeling scheme
def build_model4(nneck, lr, twindow, twindow2, nfeature, ncontrol):
    inputA = tf.keras.Input(batch_shape=(None, twindow, (nfeature + ncontrol)), name='History')
    inputB = tf.keras.Input(batch_shape=(None, None, ncontrol), name='Control')
    inputstate = tf.keras.Input(batch_shape=(None, nfeature), name='Initial_State')
    is_training = tf.keras.Input(batch_shape=(None, 1), name='Is_Training')
    label = tf.keras.Input(batch_shape=(None, None, nfeature), name='Label')
    x1 = layers.Conv1D(8, 3, activation='relu', strides=1, padding='valid', use_bias=False)(inputA)
    # x1b = layers.Conv1D(2,3, activation = 'relu', strides = 1, padding = 'valid', use_bias=False)(x1)
    # x2 = layers.Conv1D(2,3, strides = 2, padding = 'valid', use_bias=False,activation = 'relu')(x1)
    print(x1.shape)
    x2 = layers.Flatten()(x1)
    x2a1 = layers.Dense(nneck, activation='relu', use_bias=False)(x2)
    x2b1 = layers.Dense(nfeature, activation='relu', use_bias=False)(x2)
    x2a2 = layers.Dense(nneck, activation='relu', use_bias=False)(x2)
    x2b2 = layers.Dense(nfeature, activation='relu', use_bias=False)(x2)
    z0 = layers.LSTM(nneck, activation='relu', use_bias=True, return_state=False,
                     return_sequences=True, unroll=False,
                     # kernel_initializer='zeros',
                     # recurrent_initializer='identity',
                     name='GRU1')(inputB, initial_state=[x2a1, x2a2])

    z01 = layers.LSTM(nfeature, activation='linear', use_bias=True, return_state=False,
                      return_sequences=True, unroll=False,
                      # kernel_initializer='identity',
                      # recurrent_initializer='identity',
                      name='GRU2')(z0, initial_state=[x2b1, x2b2])
    z = layers.Multiply()([label, z01])
    # print(z.shape)
    initial_learning_rate = lr
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=300,
        decay_rate=0.95,
        staircase=True)
    model = tf.keras.Model(inputs=[inputA, inputB, inputstate, is_training, label], outputs=[z, z01])
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss=['mse', 'mae'], loss_weights=[1, 0], optimizer=opt, metrics=['accuracy'])
    return model


# Build CNNRNN model without labeling scheme (For CUDA requirements)
def build_model5(nneck, lr, twindow, twindow2, nfeature, ncontrol):
    inputA = tf.keras.Input(batch_shape=(None, twindow, (nfeature + ncontrol)), name='History')
    inputB = tf.keras.Input(batch_shape=(None, None, ncontrol), name='Control')
    inputstate = tf.keras.Input(batch_shape=(None, nfeature), name='Initial_State')
    is_training = tf.keras.Input(batch_shape=(None, 1), name='Is_Training')
    label = tf.keras.Input(batch_shape=(None, None, nfeature), name='Label')
    x1 = layers.Conv1D(8, 3, activation='relu', strides=1, padding='valid', use_bias=False)(inputA)
    # x1b = layers.Conv1D(2,3, activation = 'relu', strides = 1, padding = 'valid', use_bias=False)(x1)
    # x2 = layers.Conv1D(2,3, strides = 2, padding = 'valid', use_bias=False,activation = 'relu')(x1)
    # print(x1.shape)
    x2 = layers.Flatten()(x1)
    x2a1 = layers.Dense(nneck, activation='relu', use_bias=False,
                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                        bias_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005))(x2)
    x2b1 = layers.Dense(nfeature, activation='relu', use_bias=False,
                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                        bias_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005))(x2)
    x2a2 = layers.Dense(nneck, activation='relu', use_bias=False,
                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                        bias_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005))(x2)
    x2b2 = layers.Dense(nfeature, activation='relu', use_bias=False,
                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                        bias_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005))(x2)
    z0 = layers.LSTM(nneck, activation='tanh', use_bias=True, return_state=False,
                     return_sequences=True, unroll=False,
                     kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                     bias_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                     name='GRU1')(inputB, initial_state=[x2a1, x2a2])

    z01 = layers.LSTM(nfeature, activation='tanh', use_bias=True, return_state=False,
                      return_sequences=True, unroll=False,
                      kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                      bias_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                      name='GRU2')(z0, initial_state=[x2b1, x2b2])
    # print(label.shape, z01.shape)
    z = layers.Multiply()([label, z01])
    # print(z.shape)
    initial_learning_rate = lr
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=300,
        decay_rate=0.95,
        staircase=True)
    model = tf.keras.Model(inputs=[inputA, inputB, inputstate, is_training, label], outputs=[z, z01])
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss=['mse', 'mae'], loss_weights=[1, 0], optimizer=opt, metrics=['accuracy'])
    return model


#z1 = layers.TimeDistributed(layers.Dense(nz) ,name="Output")(z11)
def build_model6(nneck, lr, twindow, twindow2, nfeature, ncontrol):
    inputA = tf.keras.Input(batch_shape=(None, twindow, (nfeature + ncontrol)), name='History')
    inputB = tf.keras.Input(batch_shape=(None, None, ncontrol), name='Control')
    inputstate = tf.keras.Input(batch_shape=(None, nfeature), name='Initial_State')
    is_training = tf.keras.Input(batch_shape=(None, 1), name='Is_Training')
    label = tf.keras.Input(batch_shape=(None, None, nfeature), name='Label')
    x1 = layers.Conv1D(8, 3, activation='relu', strides=1, padding='valid', use_bias=False)(inputA)
    # x1b = layers.Conv1D(2,3, activation = 'relu', strides = 1, padding = 'valid', use_bias=False)(x1)
    # x2 = layers.Conv1D(2,3, strides = 2, padding = 'valid', use_bias=False,activation = 'relu')(x1)
    # print(x1.shape)
    x2 = layers.Flatten()(x1)
    x2a1 = layers.Dense(nneck, activation='relu', use_bias=False,
                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                        bias_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005))(x2)
    x2b1 = layers.Dense(nfeature, activation='relu', use_bias=False,
                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                        bias_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005))(x2)
    x2a2 = layers.Dense(nneck, activation='relu', use_bias=False,
                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                        bias_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005))(x2)
    x2b2 = layers.Dense(nfeature, activation='relu', use_bias=False,
                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                        bias_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005))(x2)
    z0 = layers.LSTM(nneck, activation='tanh', use_bias=True, return_state=False,
                     return_sequences=True, unroll=False,
                     kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                     bias_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                     name='GRU1')(inputB, initial_state=[x2a1, x2a2])

    z01 = layers.LSTM(nfeature, activation='tanh', use_bias=True, return_state=False,
                      return_sequences=True, unroll=False,
                      kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                      bias_regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                      name='GRU2')(z0, initial_state=[x2b1, x2b2])
    # print(label.shape, z01.shape)
    z11 = layers.TimeDistributed(layers.Dense(nfeature), name="Output")(z01)
    z = layers.Multiply()([label, z11])
    # print(z.shape)
    initial_learning_rate = lr
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=300,
        decay_rate=0.95,
        staircase=True)
    model = tf.keras.Model(inputs=[inputA, inputB, inputstate, is_training, label], outputs=[z, z01])
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss=['mse', 'mae'], loss_weights=[1, 0], optimizer=opt, metrics=['accuracy'])
    return model


def train_model(model_long, Inputs, Outputs, nbsize, nepochs, validation_split=0, verbose=1, shuffle=True):
    # Inputs=[ historyt, controlt, ...], Outputs=[youtt,youtt]
    # training loop
    a0 = time.time()
    h = model_long.fit(Inputs, Outputs,
                       batch_size=nbsize,
                       epochs=nepochs,
                       validation_split=validation_split, verbose=verbose, shuffle=shuffle)
    loss = np.asarray(h.history['loss'])
    a1 = time.time()
    return model_long, loss, h, (a1 - a0)


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, nepochs, lossbar=[0.04, 0.0065], loss_bar= 0.0035, valloss_bar=0.005,
                 call_batch=True, batch0=1000, batch_lossbar=1):
        self.lossbar = lossbar  # loss thresholds for first two epochs
        self.valloss_bar = valloss_bar  # validation loss threshold for the final epoch
        self.loss_bar = loss_bar  # training loss threshold for the final epoch
        self.nepochs = nepochs
        self.call_batch = call_batch
        self.batch0 = batch0
        self.batch_lossbar = batch_lossbar

    def on_epoch_end(self, epoch, logs=None):
        print("\nEpoch {}: loss: {:.5f}, val_loss: {:.5f}.\n".format(epoch + 1, logs['loss'], logs['val_loss']))
        if math.isnan(logs['loss']):
            print('\nThe training process is terminated at the epoch {} due to nan loss.'.format(epoch))
            raise ValueError('The initialization is not accepted.')
        # if logs['loss']<= self.loss_bar and logs['val_loss'] <= self.valloss_bar:
        #     self.model.stop_training = True
        # if logs['val_loss'] >= 100*logs['loss']:
        #     print('\nThe val_loss {} is much larger than loss {}. It is over-fit.')
        #     raise ValueError('The initialization is not accepted.')
        if epoch is 0:
            if logs['loss'] >= self.lossbar[epoch] or math.isnan(logs['loss']):
                print('\nThe training process is terminated at the first epoch due to high loss.')
                raise ValueError('The initialization is not accepted.')
            else:
                print('\nThe accepted loss for the first epoch is: {}. \nThe loss threshold is {}.'.format(logs['loss'], self.lossbar[epoch]))
            # if math.isnan(logs['val_loss']):
            #     print('\nThe training process is terminated at the first epoch due to nan val_loss.')
            #     raise ValueError('The initialization is not accepted.')
            # else:
            #     print('\nThe accepted val_loss for the first epoch is: {}.'.format(logs['val_loss']))
            self.call_batch = False
        elif epoch is 1:
            if logs['loss'] >= self.lossbar[epoch] or math.isnan(logs['loss']):
                print('\nThe training process is terminated at the second epoch due to high loss.')
                raise ValueError('The initialization is not accepted.')
            else:
                print(
                    '\nThe accepted loss for the second epoch is: {:.4f}. \nThe loss threshold is {}.'.format(logs['loss'], self.lossbar[epoch]))
            # if math.isnan(logs['val_loss']):
            #     print('\nThe training process is terminated at the second epoch due to nan val_loss.')
            #     raise ValueError('The initialization is not accepted.')
            # else:
            #     print('\nThe accepted val_loss for the second epoch is: {:.4f}.'.format(logs['val_loss']))
        elif epoch is self.nepochs - 1:
            if not math.isnan(logs['val_loss']) and logs['val_loss'] < self.valloss_bar and not math.isnan(
                    logs['loss']) and logs['loss'] < self.loss_bar:
                pass
            else:
                print('\nThe val_loss {:.4f} or loss {:.4f} is not accepted at epoch {}.'.format(logs['val_loss'], logs['loss'], epoch + 1))
                raise ValueError('The initialization is not accepted.')

    def on_train_batch_end(self, batch, logs=None):
        if self.call_batch:
            if batch == self.batch0:
                if logs['loss'] >= self.batch_lossbar or math.isnan(logs['loss']):
                    print('\nThe initialization is too far away.')
                    raise ValueError('The initialization is not accepted.')

