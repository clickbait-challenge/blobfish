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
import read_new_dataset
import tensorflow as tf


def predictOutput(input_dir, output_dir, modello):

    def score(x_VL,y_VL, model):
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

        
    #model = keras.models.load_model("longTraining/models/fullNetConc/mse0.03383071418698334_accuracy0.8404182769217581_dropout0.0_momentum0.0_lrate0.03_gru256.h5", custom_objects={'prec':prec, 'rec':rec, 'f_one':f_one})
    # model = keras.models.load_model("models/fullNetConc/mse0.06277644574126597_accuracy0.5480489961380518_dropout0.2_momentum0.0_lrate0.1_gru128.h5", custom_objects={'prec':prec, 'rec':rec, 'f_one':f_one})
    #model = keras.models.load_model("longTraining/models/lingNet/mse0.04097315482752896_accuracy0.8099568083926368_dropout0.0_momentum0.0_lrate0.1_gru128.h5", custom_objects={'prec':prec, 'rec':rec, 'f_one':f_one})
    
    
    model = keras.models.load_model(modello, custom_objects={'prec':prec, 'rec':rec, 'f_one':f_one})

    saver = tf.train.Saver()
    sess = keras.backend.get_session()
    saver.restore(sess, './keras_model')
    # sistemare forma dati passati alla rete

    x_LING = ling_feat
    x_LING = (np.array(x_LING)).astype(np.float32)
    x_LING_exp = np.expand_dims(x_LING, axis=2)

    x_TR = vect_dataset
    x_TR = (pad_sequences(x_TR, maxlen=x_LING.shape[1], dtype='int32', padding='post', truncating='post',
                          value=0.0)).astype(np.float32)

    x_CONCAT = np.concatenate((x_TR, x_LING_exp), axis=2)  # solo per fullNetwork semplice


    # target = read_new_dataset.readData(modified=True)

    # ids = []    # DA TOGLIERE
    # y_TR = []
    # for post in target:
    #     y_TR.append(float(post['truthMean']))
    #     ids.append(post['id'])      #   DA TOGLIERE
    # y_TR = np.array(y_TR)


    # tr_val = int(np.around((len(x_TR) / 100) * 80))
    # #   VALIDATION SET
    # x_VL = x_TR[tr_val:]
    # y_VL = y_TR[tr_val:]
    # x_LING_VL = x_LING[tr_val:]
    # x_LING_exp_VL = x_LING_exp[tr_val:]
    # x_CONCAT_VL = x_CONCAT[tr_val:]


    #score([x_VL, x_LING_VL], y_VL, model)
    #score([x_TR, x_LING], y_TR, model)

    # predict
    #fullNetConc
    outputPred = model.predict([x_TR, x_LING])
    #fullNet
    #outputPred = model.predict(x_CONCAT)
    #weNet
    #outputPred = model.predict(x_TR)
    #lingNet
    #outputPred = model.predict(x_LING_exp)

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
    parser.add_argument('-m', dest="model")
    args = parser.parse_args()

    predictOutput(args.input_dir, args.output_dir, args.model)



__main__()
