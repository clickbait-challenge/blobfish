import os
import tensorflow as tf
import keras
from keras.models import *
from keras.layers import *
from keras import optimizers
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, EarlyStopping
from read_new_dataset import readData
from word2vec import vectorizeDataset, lingFeatures
from plotter import Plotter
import numpy as np
import time
from metrics import *
import matplotlib.pyplot as plt
from sklearn import metrics
import json

start = time.time()

def score(x_VL,y_VL, model):
    """ Calculate sklearn metrics score

    Parameters
    ---------
    x_VL: list
        list of inputs
    y_VL: list
        list of target
    model: Model
        keras model for prediction
        
    """
    tetruth_means = y_VL
    petruth_means = model.predict(x_VL)
    tetruthClass = []
    petruthClass = []


    for i in range(len(tetruth_means)):

        if petruth_means[i] > 0.5:
            petruthClass.append(1)
        else:
            petruthClass.append(0)

        if tetruth_means[i] > 0.5:
            tetruthClass.append(1)
        else:
            tetruthClass.append(0)


    mse = metrics.mean_squared_error(tetruth_means,petruth_means)
    print('Mean Squared Error = '+str(mse))

    accuracy = metrics.accuracy_score(tetruthClass,petruthClass)
    print('accuracy = '+str(accuracy))

    precision = metrics.precision_score(tetruthClass,petruthClass)
    print('precision_score = '+str(precision))

    recall = metrics.recall_score(tetruthClass,petruthClass)
    print('recall_score = '+str(recall))

    f1 = metrics.f1_score(tetruthClass,petruthClass)
    print('f1_score = '+str(f1))


def fullNetworkConcat(x_TR, x_LING, y_TR, x_VL, x_LING_VL, y_VL, g_u, lr, mom, d_e, ep):
    """ Train the FullNetPost model

    Parameters
    ---------
    x_TR: list
        list of train word embedding inputs
    x_LING: list
        list of train features
    y_TR: list
        list of train tagets
    x_VL: list
        list of validation word embedding inputs
    x_LING_VL: list
        list of validation features
    y_VL: list
        list of validation targets
    g_u: int
        gru units
    lr: float
        learning rate
    mom: float
        momentum
    d_e: float
        droput embedding
    ep: int
        epochs
   
    Returns
    ---------
    trainLoss: list
        list of train loss history
    trainAcc: list
        list of train accuracy history
    valLoss: list
        list of validation loss history
    valAcc: list
        list of validation accuracy history
    precision: list
        list of train precision history
    recall: list
        list of train recall history
    f1: list
        list of train f1 history
    precisionVal: list
        list of validation precision history
    recallVal: list
        list of validation recall history
    f1Val: list
        list of validation f1 history
    model: Model
        keras model of the network
        
    """

    wordEmb = Input(shape=(x_LING.shape[1],100,))
    dropoutWE = Dropout(d_e)(wordEmb)
    gru = Bidirectional(GRU(g_u))(dropoutWE)

    lingInput = Input(shape=(x_LING.shape[1],))
    dropoutLF = Dropout(d_e)(lingInput)

    concat = concatenate([gru, dropoutLF])
    concat_out = Dense(1, activation='sigmoid')(concat)
    model = Model(inputs=[wordEmb, lingInput], outputs=[concat_out])

    #sgd_n = optimizers.SGD(lr=lr, momentum=mom, nesterov=False)
    sgd_n = optimizers.Adadelta(lr=lr, rho=0.95, epsilon=None, decay=0.0)

    #model.compile(loss='mse', optimizer=sgd_n, metrics=['accuracy', prec, rec, f_one])
    model.compile(loss='mse', optimizer=sgd_n, metrics=[acc, prec, rec, f_one])

    path = "models/fullNetConc/Early/"
    if not os.path.exists(path):
        os.makedirs(path)
    name = path + "_dropout" + str(d_e) + "_momentum" + str(mom) + "_lrate" + str(lr) + "_gru" + str(g_u)
    checkpointer = ModelCheckpoint(filepath= name + ".hdf5", verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=7, verbose=1)

    training = model.fit([x_TR, x_LING],y_TR, validation_data=([x_VL,x_LING_VL],y_VL), epochs=ep, callbacks=[checkpointer, earlystopper])
    model.summary()

    score([x_TR,x_LING], y_TR, model)

    trainLoss = training.history['loss']
    trainAcc = training.history['acc']
    valLoss = training.history['val_loss']
    valAcc = training.history['val_acc']
    precision = training.history['prec']
    recall = training.history['rec']
    f1 = training.history['f_one']
    precisionVal = training.history['val_prec']
    recallVal = training.history['val_rec']
    f1Val = training.history['val_f_one']

    return trainLoss, trainAcc, valLoss, valAcc, precision, recall, f1, precisionVal, recallVal, f1Val, model



def weNetwork(x_TR, y_TR, x_VL, y_VL, g_u, lr, mom, dropout_embedding, ep):
    """ Train the FullNetPost model

    Parameters
    ---------
    x_TR: list
        list of train word embedding inputs
    y_TR: list
        list of train tagets
    x_VL: list
        list of validation word embedding inputs
    y_VL: list
        list of validation targets
    g_u: int
        gru units
    lr: float
        learning rate
    mom: float
        momentum
    dropout_embedding: float
        droput embedding
    ep: int
        epochs
   
    Returns
    ---------
    trainLoss: list
        list of train loss history
    trainAcc: list
        list of train accuracy history
    valLoss: list
        list of validation loss history
    valAcc: list
        list of validation accuracy history
    precision: list
        list of train precision history
    recall: list
        list of train recall history
    f1: list
        list of train f1 history
    precisionVal: list
        list of validation precision history
    recallVal: list
        list of validation recall history
    f1Val: list
        list of validation f1 history
    model: Model
        keras model of the network
        
    """

    model=Sequential()
    model.add(Dropout(dropout_embedding))
    model.add(Bidirectional(GRU(g_u)))
    model.add(Dense(1, activation='sigmoid'))
    #sgd_n = optimizers.SGD(lr=lr, momentum=mom, nesterov=False)
    sgd_n = optimizers.Adadelta(lr=lr, rho=0.95, epsilon=None, decay=0.0)
    model.compile(loss='mse', optimizer=sgd_n, metrics=[acc, prec, rec, f_one])

    path = "models/weNet/Early/"
    if not os.path.exists(path):
        os.makedirs(path)
    name = path + "_dropout" + str(dropout_embedding) + "_momentum" + str(mom) + "_lrate" + str(lr) + "_gru" + str(g_u)
    checkpointer = ModelCheckpoint(filepath=name + ".hdf5", verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=7, verbose=1)

    #training = model.fit(x_TR, y_TR, epochs=ep)
    training = model.fit(x_TR, y_TR, validation_data= (x_VL,y_VL), epochs=ep, callbacks=[checkpointer, earlystopper])

    score(x_VL, y_VL, model)

    # DATI DAL TRAINING
    trainLoss = training.history['loss']
    valLoss = training.history['val_loss']
    trainAcc = training.history['acc']
    valAcc = training.history['val_acc']
    precision = training.history['prec']
    recall = training.history['rec']
    f1 = training.history['f_one']
    precisionVal = training.history['val_prec']
    recallVal = training.history['val_rec']
    f1Val = training.history['val_f_one']

    return trainLoss, trainAcc, valLoss, valAcc, precision, recall, f1, precisionVal, recallVal, f1Val, model



def lingNetwork(x_LING_TR, y_TR, x_LING_VL, y_VL, g_u, lr, mom, dropout_embedding, ep):
    """ Train the FullNetPost model

    Parameters
    ---------
    x_LING_TR: list
        list of train features
    y_TR: list
        list of train tagets
    x_LING_VL: list
        list of validation features
    y_VL: list
        list of validation targets
    g_u: int
        gru units
    lr: float
        learning rate
    mom: float
        momentum
    dropout_embedding: float
        droput embedding
    ep: int
        epochs
   
    Returns
    ---------
    trainLoss: list
        list of train loss history
    trainAcc: list
        list of train accuracy history
    valLoss: list
        list of validation loss history
    valAcc: list
        list of validation accuracy history
    precision: list
        list of train precision history
    recall: list
        list of train recall history
    f1: list
        list of train f1 history
    precisionVal: list
        list of validation precision history
    recallVal: list
        list of validation recall history
    f1Val: list
        list of validation f1 history
    model: Model
        keras model of the network
        
    """

    model = Sequential()
    model.add(Dropout(dropout_embedding))
    model.add(Bidirectional(GRU(g_u)))
    model.add(Dense(1, activation='sigmoid'))
    sgd_n = optimizers.SGD(lr=lr, momentum=mom, nesterov=False)
    #sgd_n = optimizers.Adadelta(lr=lr, rho=0.95, epsilon=None, decay=0.0)
    model.compile(loss='mse', optimizer=sgd_n, metrics=[acc, prec, rec, f_one])

    path = "models/lingNet/Early/"
    if not os.path.exists(path):
        os.makedirs(path)
    name = path + "_dropout" + str(dropout_embedding) + "_momentum" + str(mom) + "_lrate" + str(lr) + "_gru" + str(g_u)
    checkpointer = ModelCheckpoint(filepath=name + ".hdf5", verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=7, verbose=1)

    training = model.fit(x_LING_TR, y_TR, validation_data=(x_LING_VL, y_VL),epochs=ep, callbacks=[checkpointer, earlystopper])

    score(x_LING_VL, y_VL, model)

    # DATI DAL TRAINING
    trainLoss = training.history['loss']
    valLoss = training.history['val_loss']
    trainAcc = training.history['acc']
    valAcc = training.history['val_acc']
    precision = training.history['prec']
    recall = training.history['rec']
    f1 = training.history['f_one']
    precisionVal = training.history['val_prec']
    recallVal = training.history['val_rec']
    f1Val = training.history['val_f_one']

    return trainLoss, trainAcc, valLoss, valAcc, precision, recall, f1, precisionVal, recallVal, f1Val, model


def fullNetwork(x_CONCAT_TR, y_TR, x_CONCAT_VL, y_VL, g_u, lr, mom, dropout_embedding, ep):
    """ Train the FullNetPost model

    Parameters
    ---------
    x_CONCAT_TR: list
        list of train word embedding inputs and linguistics features
    y_TR: list
        list of train tagets
    x_CONCAT_VL: list
        list of validation word embedding inputs and linguistics features
    y_VL: list
        list of validation targets
    g_u: int
        gru units
    lr: float
        learning rate
    mom: float
        momentum
    dropout_embedding: float
        droput embedding
    ep: int
        epochs
   
    Returns
    ---------
    trainLoss: list
        list of train loss history
    trainAcc: list
        list of train accuracy history
    valLoss: list
        list of validation loss history
    valAcc: list
        list of validation accuracy history
    precision: list
        list of train precision history
    recall: list
        list of train recall history
    f1: list
        list of train f1 history
    precisionVal: list
        list of validation precision history
    recallVal: list
        list of validation recall history
    f1Val: list
        list of validation f1 history
    model: Model
        keras model of the network
        
    """

    model=Sequential()
    model.add(Dropout(dropout_embedding))
    model.add(Bidirectional(GRU(g_u)))
    model.add(Dense(1, activation='sigmoid'))
    #sgd_n = optimizers.SGD(lr=lr, momentum=mom, nesterov=False)
    sgd_n = optimizers.Adadelta(lr=lr, rho=0.95, epsilon=None, decay=0.0)
    model.compile(loss='mse', optimizer=sgd_n, metrics=[acc, prec, rec, f_one])

    path = "models/fullNet/Early/"
    if not os.path.exists(path):
        os.makedirs(path)
    name = path + "_dropout" + str(dropout_embedding) + "_momentum" + str(mom) + "_lrate" + str(lr) + "_gru" + str(g_u)
    checkpointer = ModelCheckpoint(filepath=name + ".hdf5", verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=7, verbose=1)

    training = model.fit(x_CONCAT_TR, y_TR, validation_data = (x_CONCAT_VL,y_VL),epochs=ep, callbacks = [checkpointer, earlystopper])

    score(x_CONCAT_VL, y_VL, model)

    # DATI DAL TRAINING
    trainLoss = training.history['loss']
    valLoss = training.history['val_loss']
    trainAcc = training.history['acc']
    valAcc = training.history['val_acc']
    precision = training.history['prec']
    recall = training.history['rec']
    f1 = training.history['f_one']
    precisionVal = training.history['val_prec']
    recallVal = training.history['val_rec']
    f1Val = training.history['val_f_one']

    return trainLoss, trainAcc, valLoss, valAcc, precision, recall, f1, precisionVal, recallVal, f1Val, model


def networkSettings(vect_data, ling_feat, test=False, longTR = False):
    """ Defines network settings

    Parameters
    ---------
    vect_data: list
        list of word embedding inputs
    ling_feat: list
        list of linguistics features
    test: bool, default False
        true if is test dataset
    longTR: bool, default False
        true for long training
        
    """

    x_LING = ling_feat
    x_LING = (np.array(x_LING)).astype(np.float32)
    x_LING_exp = np.expand_dims(x_LING, axis=2)

    x_TR = vect_data
    x_TR = (pad_sequences(x_TR, maxlen=x_LING.shape[1], dtype='int32', padding='post', truncating='post',
                          value=0.0)).astype(np.float32)

    x_CONCAT = np.concatenate((x_TR, x_LING_exp), axis=2)  
    target = readData(modified=True)

    y_TR = []
    for post in target:
        y_TR.append(float(post['truthMean']))
    y_TR = np.array(y_TR)


    if longTR:
        print("in longTR")

        tr_val = int(np.around((len(x_TR) / 100) * 80))

        # TRAINING SET
        x_TR_TR = x_TR[:tr_val]
        y_TR_TR = y_TR[:tr_val]
        x_LING_TR = x_LING[:tr_val]
        x_LING_exp_TR = x_LING_exp[:tr_val]
        x_CONCAT_TR = x_CONCAT[:tr_val]

        #   VALIDATION SET
        x_VL = x_TR[tr_val:]
        y_VL = y_TR[tr_val:]
        x_LING_VL = x_LING[tr_val:]
        x_LING_exp_VL = x_LING_exp[tr_val:]
        x_CONCAT_VL = x_CONCAT[tr_val:]

        i = 0

        #lr = [0.002,0.003,0.01]
        # lr = [0.1, 0.2, 0.3, 0.4]
        # mom = [0.0]
        # drop = [0.0, 0.2, 0.5]
        # gru_units = [128]
        # ep = 150

        for mod in [1]:
            # for dropout_embedding in drop:
            #     for m in mom:
            #         for lrate in lr:
            #             for g_u in gru_units:

            if mod == 1:
                lrate = 0.30
                m = 0.0
                dropout_embedding = 0.2
                g_u = 128
                ep = 150  # 150

                trainLoss, trainAcc, mse, accuracy, precision, recall, f1, precisionVal, recallVal, f1Val, model = fullNetworkConcat(
                    x_TR_TR, x_LING_TR, y_TR_TR, x_VL, x_LING_VL, y_VL, g_u, lrate, m, dropout_embedding, ep)
                path = "longTraining/models/fullNetConc/"
                name = path + "mse" + str(mse[len(mse) - 1]) + "_accuracy" + str(accuracy[len(mse) - 1]) + "_dropout" + str(
                    dropout_embedding) + "_momentum" + str(m) + "_lrate" + str(lrate) + "_gru" + str(g_u) + ".hdf5"
            if mod == 2:
                lrate = 0.30
                m = 0.0
                dropout_embedding = 0.5
                g_u = 128
                ep = 150  # 150

                trainLoss, trainAcc, mse, accuracy, precision, recall, f1, precisionVal, recallVal, f1Val, model = weNetwork(
                    x_TR_TR, y_TR_TR, x_VL, y_VL, g_u, lrate, m, dropout_embedding, ep)
                path = "longTraining/models/weNet/"
                name = path + "mse" + str(mse[len(mse) - 1]) + "_accuracy" + str(
                    accuracy[len(mse) - 1]) + "_dropout" + str(
                    dropout_embedding) + "_momentum" + str(m) + "_lrate" + str(lrate) + "_gru" + str(g_u) + ".hdf5"
            if mod == 3:
                lrate = 0.4
                m = 0.0
                dropout_embedding = 0.0
                g_u = 128
                ep = 150  # 150

                trainLoss, trainAcc, mse, accuracy, precision, recall, f1, precisionVal, recallVal, f1Val, model = lingNetwork(
                    x_LING_exp_TR, y_TR_TR, x_LING_exp_VL, y_VL, g_u, lrate, m, dropout_embedding, ep)
                path = "longTraining/models/lingNet/"
                name = path + "mse" + str(mse[len(mse) - 1]) + "_accuracy" + str(
                    accuracy[len(mse) - 1]) + "_dropout" + str(
                    dropout_embedding) + "_momentum" + str(m) + "_lrate" + str(lrate) + "_gru" + str(g_u) + ".hdf5"
            if mod == 4:
                lrate = 0.30
                m = 0.0
                dropout_embedding = 0.5
                g_u = 128
                ep = 150  # 150

                trainLoss, trainAcc, mse, accuracy, precision, recall, f1, precisionVal, recallVal, f1Val, model = fullNetwork(
                    x_CONCAT_TR, y_TR_TR, x_CONCAT_VL, y_VL, g_u, lrate, m, dropout_embedding, ep)
                path = "longTraining/models/fullNet/"
                name = path + "mse" + str(mse[len(mse) - 1]) + "_accuracy" + str(
                    accuracy[len(mse) - 1]) + "_dropout" + str(
                    dropout_embedding) + "_momentum" + str(m) + "_lrate" + str(lrate) + "_gru" + str(g_u) + ".hdf5"

            epoch_count = range(1, len(trainLoss) + 1)
            Plotter.plotError(epoch_count, trainLoss, mse, dropout_embedding, m, lrate, g_u, path)
            Plotter.plotAccuracy(epoch_count, trainAcc, accuracy, dropout_embedding, m, lrate, g_u, path)
            Plotter.plotPrecision(epoch_count, precision, precisionVal, dropout_embedding, m, lrate, g_u, path)
            Plotter.plotRecall(epoch_count, recall, recallVal, dropout_embedding, m, lrate, g_u, path)
            Plotter.plotFOne(epoch_count, f1, f1Val, dropout_embedding, m, lrate, g_u, path)

            nomefile_train = path + "model" + str(mod) + "_dropout" + str(
                dropout_embedding) + "_momentum" + str(m) + "_lrate" + str(
                    lrate) + "_gru" + str(g_u) + "_TRAIN.txt"
            with open(nomefile_train, "a") as myfile:
                myfile.write('epoch,mseTrain,mseVal,trainAccuracy,valAccuray,trainPrecision,valPrecision,trainRecall,valRecall,trainF1,valF1 \n')
            for i in epoch_count:
                with open(nomefile_train, "a") as myfile:
                    myfile.write(
                        str(i)+','+ str(trainLoss[i-1]) + ',' + str(mse[i-1]) + ',' + str(trainAcc[i-1]) +
                        ',' + str(accuracy[i-1]) + ',' + str(precision[i-1]) + ',' +
                        str(precisionVal[i-1]) + ',' +
                        str(recall[i-1]) + ',' +
                        str(recallVal[i-1]) + ',' + str(f1[i-1]) + ','+ str(f1Val[i-1])+
                        ' \n')

            print('model: ', mod)
            print('Statesize: \t ' + str(g_u))
            print('dropout embedding: \t ' + str(dropout_embedding))
            print('momentum: \t ' + str(m))
            print('learning rate: \t ' + str(lrate))
            print('precision_score = ' + str(precision[len(mse) - 1]))
            print('recall_score = ' + str(recall[len(mse) - 1]))
            print('f1_score = ' + str(f1[len(mse) - 1]))
            print('Mean Squared Error TR = ' + str(trainLoss[len(trainLoss) - 1]))
            print('Mean Squared Error VL = ' + str(mse[len(mse) - 1]))
            print('accuracy TR= ' + str(trainAcc[len(trainAcc) - 1]))
            print('accuracy VL= ' + str(accuracy[len(mse) - 1]))
            print('\n\n')

            saver = tf.train.Saver()
            sess = keras.backend.get_session()
            name_model="./"+name
            saver.save(sess, name_model)

            model.save(name)

            nomefile = path + "model" + str(mod) + "_dropout" + str(dropout_embedding) + "_momentum" + str(
                m) + "_lrate" + str(lrate) + "_gru" + str(g_u) + ".txt"
            with open(nomefile, "a") as myfile:
                myfile.write(
                    str(g_u) + ',' + str(dropout_embedding) +
                    ',' + str(m) + ',' + str(lrate) + ',' + str(mse[len(mse) - 1]) +
                    ',' + str(accuracy[len(mse) - 1]) + ',' + str(precision[len(mse) - 1]) + ',' +
                    str(recall[len(mse) - 1]) + ',' + str(f1[len(mse) - 1]) + ' \n')


    else:
        i = 0

        lr = [0.001,0.002,0.003,0.01, 0.02, 0.03]
        mom = [0.0]
        drop = [0.0, 0.2]
        gru_units = [128, 256]
        ep = 20

        for mod in [1,2,3,4]:
            for dropout_embedding in drop:
                for m in mom:
                    for lrate in lr:
                        for g_u in gru_units:
                            crossValidation(x_TR, y_TR, x_LING, x_LING_exp, x_CONCAT, mod, g_u, lrate, m, dropout_embedding, ep, i)
                            i = i+1



def crossValidation(x, y, x_ling, x_ling_exp, x_concat, mod, g_u, lrate, m, dropout_embedding, ep, i):
    print("in cross validation")

    # mod = numero modello
    # k = numero di parti in cui dividere il dataset
    k = 5

    input_size = len(x)
    resto = input_size % k
    fold_size = int(input_size / k)
    start_idx = 0
    acc_list = []
    err_list = []

    for index in range(1, k + 1):
        if resto != 0:
            end_idx = start_idx + (fold_size + 1)
            resto = resto - 1
        else:
            end_idx = start_idx + fold_size

        x_VL = x[start_idx:end_idx]
        x_LING_VL = x_ling[start_idx:end_idx]
        x_LING_exp_VL = x_ling_exp[start_idx:end_idx]
        x_CONCAT_VL = x_concat[start_idx:end_idx]
        y_VL = y[start_idx:end_idx]


        x_TR_TR = np.delete(x, np.s_[start_idx:end_idx], axis=0)
        x_LING_TR = np.delete(x_ling, np.s_[start_idx:end_idx], axis=0)
        x_LING_exp_TR = np.delete(x_ling_exp, np.s_[start_idx:end_idx], axis=0)
        x_CONCAT_TR = np.delete(x_concat, np.s_[start_idx:end_idx], axis=0)
        y_TR_TR = np.delete(y, np.s_[start_idx:end_idx], axis=0)

        start_idx = end_idx

        if mod == 1:
            trainLoss, trainAcc, mse, accuracy, precision, recall, f1, precisionVal, recallVal, f1Val, model = fullNetworkConcat(
                x_TR_TR, x_LING_TR, y_TR_TR, x_VL, x_LING_VL, y_VL, g_u, lrate, m, dropout_embedding, ep)
            path = "models/fullNetConc/"
            name = path + "mse" + str(mse[len(mse) - 1]) + "_accuracy" + str(accuracy[len(mse) - 1]) + "_dropout" + str(
                dropout_embedding) + "_momentum" + str(m) + "_lrate" + str(lrate) + "_gru" + str(g_u) + ".hdf5"
        if mod == 2:
            trainLoss, trainAcc, mse, accuracy, precision, recall, f1, precisionVal, recallVal, f1Val, model = weNetwork(
                x_TR_TR, y_TR_TR, x_VL, y_VL, g_u, lrate, m, dropout_embedding, ep)
            path = "models/weNet/"
            name = path + "mse" + str(mse[len(mse) - 1]) + "_accuracy" + str(accuracy[len(mse) - 1]) + "_dropout" + str(
                dropout_embedding) + "_momentum" + str(m) + "_lrate" + str(lrate) + "_gru" + str(g_u) + ".hdf5"
        if mod == 3:
            trainLoss, trainAcc, mse, accuracy, precision, recall, f1, precisionVal, recallVal, f1Val, model = lingNetwork(
                x_LING_exp_TR, y_TR_TR, x_LING_exp_VL, y_VL, g_u, lrate, m, dropout_embedding, ep)
            path = "models/lingNet/"
            name = path + "mse" + str(mse[len(mse) - 1]) + "_accuracy" + str(accuracy[len(mse) - 1]) + "_dropout" + str(
                dropout_embedding) + "_momentum" + str(m) + "_lrate" + str(lrate) + "_gru" + str(g_u) + ".hdf5"
        if mod == 4:
            trainLoss, trainAcc, mse, accuracy, precision, recall, f1, precisionVal, recallVal, f1Val, model = fullNetwork(
                x_CONCAT_TR, y_TR_TR, x_CONCAT_VL, y_VL, g_u, lrate, m, dropout_embedding, ep)
            path = "models/fullNet/"
            name = path + "mse" + str(mse[len(mse) - 1]) + "_accuracy" + str(accuracy[len(mse) - 1]) + "_dropout" + str(
                dropout_embedding) + "_momentum" + str(m) + "_lrate" + str(lrate) + "_gru" + str(g_u) + ".hdf5"

        err_list.append(mse[len(mse)-1])
        acc_list.append(accuracy[len(accuracy)-1])

        pathcv = path +"cross_val"+ str(i) +"/fold" + str(index) + "/"

        epoch_count = range(1, len(trainLoss) + 1)
        Plotter.plotError(epoch_count, trainLoss, mse, dropout_embedding, m, lrate, g_u, pathcv)
        Plotter.plotAccuracy(epoch_count, trainAcc, accuracy, dropout_embedding, m, lrate, g_u, pathcv)
        Plotter.plotPrecision(epoch_count, precision, precisionVal, dropout_embedding, m, lrate, g_u, pathcv)
        Plotter.plotRecall(epoch_count, recall, recallVal, dropout_embedding, m, lrate, g_u, pathcv)
        Plotter.plotFOne(epoch_count, f1, f1Val, dropout_embedding, m, lrate, g_u, pathcv)

        print("FINE PLOT")

        print('fold n: ', index)
        print('model: ', mod)
        print('Statesize: \t ' + str(g_u))
        print('dropout embedding: \t ' + str(dropout_embedding))
        print('momentum: \t ' + str(m))
        print('learning rate: \t ' + str(lrate))
        print('precision_score = ' + str(precision[len(mse) - 1]))
        print('recall_score = ' + str(recall[len(mse) - 1]))
        print('f1_score = ' + str(f1[len(mse) - 1]))
        print('Mean Squared Error TR = ' + str(trainLoss[len(trainLoss) - 1]))
        print('Mean Squared Error VL = ' + str(mse[len(mse) - 1]))
        print('accuracy TR= ' + str(trainAcc[len(trainAcc) - 1]))
        print('accuracy VL= ' + str(accuracy[len(mse) - 1]))
        print('\n\n')

        nomefile = pathcv + "model" + str(mod) + "_dropout" + str(dropout_embedding) + "_momentum" + str(
            m) + "_lrate" + str(lrate) + "_gru" + str(g_u) + ".txt"
        with open(nomefile, "a") as myfile:
            myfile.write(
                str(g_u) + ',' + str(dropout_embedding) +
                ',' + str(m) + ',' + str(lrate) + ',' + str(mse[len(mse) - 1]) +
                ',' + str(accuracy[len(mse) - 1]) + ',' + str(precision[len(mse) - 1]) + ',' +
                str(recall[len(mse) - 1]) + ',' + str(f1[len(mse) - 1]) + ' \n')
        print("fine fold n ", index)

    acc_mean = np.mean(acc_list)
    err_mean = np.mean(err_list)

    # il modello migliore sarà quello che presenta err_mean più basso

    # model.save(name)

    nomefile = path +"cross_val"+ str(i)  + "riassuntoCV"+ str(i)
    with open(nomefile, "a") as myfile:
        myfile.write(str(err_mean) + ', ' + str(acc_mean) + ',' +
            str(g_u) + ',' + str(dropout_embedding) +
            ',' + str(m) + ',' + str(lrate) + ',' + str(mse[len(mse) - 1]) +
            ',' + str(accuracy[len(mse) - 1]) + ',' + str(precision[len(mse) - 1]) + ',' +
            str(recall[len(mse) - 1]) + ',' + str(f1[len(mse) - 1]) + ' \n')


    return trainLoss, trainAcc, mse, accuracy, precision, recall, f1, precisionVal, recallVal, f1Val, model


