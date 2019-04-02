import keras
from keras.models import *
import training_data
import tokenizer
import word2vec
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import argparse
import os
import json
from metrics import *
import tensorflow as tf


def predictOutput(input_dir, output_dir, type_model):
    def prec(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def rec(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def f_one(y_true, y_pred, beta=1):
        if beta < 0:
            raise ValueError(
                'The lowest choosable beta is zero (only precision).')
        # If there are no true positives, fix the F score at 0 like sklearn.
        if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
            return 0
        p = prec(y_true, y_pred)
        r = rec(y_true, y_pred)
        bb = beta**2
        fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
        return fbeta_score

    for fil in os.listdir(input_dir):
        if fil == "instances.jsonl":
            print(fil)
            instances = input_dir + "/" + fil
            print(instances)

    # leggere dataset e trasformarlo in lista
    datas = training_data.getTestData(instances)
    #datas = read_new_dataset.readData(modified=True)

    # modificare dataset
    datas = tokenizer.special_char(datas, "", test=True)

    # tokenizzare
    tokens, post_pos, pos, ids = tokenizer.tokenizer(datas)

    # creare embedding
    # all'interno c'è già il richiamo al modello di w2v allenato sugli unlabeled

    vect_dataset = word2vec.vectorizeDataset(tokens)

    # creare feature linguistiche
    ling_feat = word2vec.lingFeatures(post_pos)

    # caricare il modello ----> METTERE PERCORSO DEL MODELLO MIGLIORE

    dir = "longTraining/models/"

    if type_model == "FullNet":
        model_path = dir + "fullNet/mse0.032553340215849914_accuracy0.842464196462473_dropout0.5_momentum0.0_lrate0.3_gru128.hdf5"
    if type_model == "FullNetPost":
        model_path = dir + "fullNetConc/mse0.032816953588701375_accuracy0.8392816549757715_dropout0.2_momentum0.0_lrate0.3_gru128.hdf5"
    if type_model == "LingNet":
        model_path = dir + "lingNet/mse0.040141911279121346_accuracy0.8167765401498543_dropout0.0_momentum0.0_lrate0.4_gru128.hdf5"
    if type_model == "WordEmbNet":
        model_path = dir + "weNet/mse0.03307730689472056_accuracy0.8388270061784073_dropout0.5_momentum0.0_lrate0.3_gru128.hdf5"

    model = keras.models.load_model(
        model_path, custom_objects={
            'prec': prec,
            'rec': rec,
            'f_one': f_one
        })

    keras_model = "./" + model_path
    saver = tf.train.Saver()
    sess = keras.backend.get_session()
    saver.restore(sess, keras_model)

    # sistemare forma dati passati alla rete
    x_LING = ling_feat
    x_LING = (np.array(x_LING)).astype(np.float32)
    x_LING_exp = np.expand_dims(x_LING, axis=2)

    x_TR = vect_dataset
    x_TR = (pad_sequences(
        x_TR,
        maxlen=x_LING.shape[1],
        dtype='int32',
        padding='post',
        truncating='post',
        value=0.0)).astype(np.float32)

    x_CONCAT = np.concatenate((x_TR, x_LING_exp),
                              axis=2)  # solo per fullNetwork semplice

    # predict
    if type_model == "FullNet":
        outputPred = model.predict(x_CONCAT)
    if type_model == "FullNetPost":
        outputPred = model.predict([x_TR, x_LING])
    if type_model == "LingNet":
        outputPred = model.predict(x_LING_exp)
    if type_model == "WordEmbNet":
        outputPred = model.predict(x_TR)

    # lista con id - otuput predict
    with open(
            os.path.join(output_dir, "results.jsonl"), 'w',
            encoding="utf-8") as output:
        for i in range(len(ids)):
            output.write(
                json.dumps({
                    "id": ids[i],
                    "clickbaitScore": float(outputPred[i])
                }) + '\n')


def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest="input_dir")
    parser.add_argument('-o', dest="output_dir")
    parser.add_argument('-m', dest="type_model")
    args = parser.parse_args()

    predictOutput(args.input_dir, args.output_dir, args.type_model)


__main__()
