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


def predictOutput(input_dir, output_dir):

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
            raise ValueError('The lowest choosable beta is zero (only precision).')
        # If there are no true positives, fix the F score at 0 like sklearn.
        if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
            return 0
        p = prec(y_true, y_pred)
        r = rec(y_true, y_pred)
        bb = beta ** 2
        fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
        return fbeta_score

    
    for fil in os.listdir(input_dir):
        if fil == "instances.jsonl":
            print(fil)
            instances = input_dir+"/"+fil
            print(instances)


    

    # leggere dataset e trasformarlo in lista
    datas = training_data.getTestData(instances)

    # tokenizzare
    tokens, post_pos, pos, ids = tokenizer.tokenizer(datas)

    # creare embedding
            # all'interno c'è già il richiamo al modello di w2v allenato sugli unlabeled
    vect_dataset = word2vec.vectorizeDataset(tokens)

    # creare feature linguistiche
    ling_feat = word2vec.lingFeatures(post_pos)

    # caricare il modello ----> METTERE PERCORSO DEL MODELLO MIGLIORE

        
    # model = keras.models.load_model("longTraining/models/fullNetConc/mse0.03383071418698334_accuracy0.8404182769217581_dropout0.0_momentum0.0_lrate0.03_gru256.h5", custom_objects={'prec':prec, 'rec':rec, 'f_one':f_one})
    model = keras.models.load_model("models/fullNetConc/mse0.06277644574126597_accuracy0.5480489961380518_dropout0.2_momentum0.0_lrate0.1_gru128.h5", custom_objects={'prec':prec, 'rec':rec, 'f_one':f_one})

    # sistemare forma dati passati alla rete

    x_LING = ling_feat
    x_LING = (np.array(x_LING)).astype(np.float32)
    x_LING_exp = np.expand_dims(x_LING, axis=2)

    x_TR = vect_dataset
    x_TR = (pad_sequences(x_TR, maxlen=x_LING.shape[1], dtype='int32', padding='post', truncating='post',
                          value=0.0)).astype(np.float32)

    x_CONCAT = np.concatenate((x_TR, x_LING_exp), axis=2)  # solo per fullNetwork semplice


    # predict
    #fullNetConc
    outputPred = model.predict([x_TR, x_LING])
    #fullNet
    #outputPred = model.predict(x_CONCAT)
    #weNet
    #outputPred = model.predict(x_TR)
    #lingNet
    #outputPred = model.predict(x_LING)

    # lista con id - otuput predict
    with open(os.path.join(output_dir, "results.jsonl"), 'w', encoding="utf-8") as output:
        for i in range(len(ids)):
            output.write(json.dumps({"id": ids[i], "clickbaitScore": float(outputPred[i])}) + '\n')


    '''results = []
    i=0
    for i in ids:
        results.append({
            'id': i,
            'clickbaitScore': outputPred[i]
        })
        i=i+1

    # salvare in formato json
    with open('output', 'w') as outfile:
        json.dump(modified_data, outfile)'''




def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest="input_dir")
    parser.add_argument('-o', dest="output_dir")
    args = parser.parse_args()

    predictOutput(args.input_dir, args.output_dir)



__main__()
