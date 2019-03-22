import gensim
from gensim.models import Word2Vec
import nltk
from nltk import trigrams
from sklearn.decomposition import PCA
from matplotlib import pyplot
import time
from tokenizer import tokenizer
from feature2vec import feature2vec
import numpy as np

#tokens = tokenizer()[0]


def w2v(tokens, min):
    # tokens = concatenazione tokens labeled e tokens unlabeled (fatta fuori, quando si chiama w2v)
    print("+++++++++++++++++++WORD2VEC+++++++++++++++++++")
    '''
        basta allenare word2vec sui soli postText o, solo in questo caso, per dare più dati, si deve prendere anche i targetParagraphs?
    '''
    model = Word2Vec(tokens, min_count=min)
    model.train(tokens, total_examples=len(tokens), epochs=10)
    model.save("word2vec.model")

    return model


def vectorizeDataset(tokens):

    model = Word2Vec.load("word2vec.model")
    '''
       vect_dataset = lista con tutti i post, ognuno dei quali ha, al posto di ogni parola, un vettore numerico (embedding della parola) 
    '''
    vect_dataset = []
    # for post in tokens:
    #     tmp = []
    #     for w in post:
    #         vec_w = model[w]
    #         tmp.append(vec_w)
    #     vect_dataset.append(tmp)
    OOV_alr = False
    unknown = []
    for post in tokens:
        tmp = []
        for w in post:
            if w in model:
                vec_w = model[w]
                #vec_w = concatenateRandomNum(model[w])
            else:
                if OOV_alr:
                    vec_w = unknown
                else:
                    #vec_w = concatenateRandomNum(np.random.uniform(-0.1, 0.1, 100))
                    vec_w = np.random.uniform(-0.1, 0.1, 100)
                    unknown = vec_w
                    OOV_alr = True
            tmp.append(vec_w)
        vect_dataset.append(tmp)
    print("ok vectorized")

    return vect_dataset


def concatenateRandomNum(vec):
    extra_embedding = [np.zeros(100), np.random.uniform(-0.1, 0.1, 100)]
    return np.concatenate((vec, extra_embedding), axis=None)


def lingFeatures(post_pos):
    vec_feat = feature2vec(post_pos)
    vec_feat.execute()
    v = vec_feat.get_vect()
    return v

# print(len(vect_dataset))
# print(list(model.wv.vocab))

# model.save_word2vec_format('/datasets/vect/w2v.txt', binary=False)
'''
# train model
model = Word2Vec(tokens,size=100, min_count=1)
# summarize the loaded model
print("model : ")
print(model)

# summarize vocabulary
words = list(model.wv.vocab)
print("words")
print(words)
# access vector for one word
#print("Model [x]")
#print(model["JJS"])
# save model
# model.save('model.bin')
# load model
# new_model = Word2Vec.load('model.bin')
# print(new_model)



X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()
'''
