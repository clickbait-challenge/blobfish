import training_data
import tokenizer
import read_new_dataset
import word2vec
import RNN

# prendere file jsonl e creare dataset
dataset_lab = training_data.getData()
dataset_unlab = training_data.getDataUnlabeled()

# modificare dataset e salvarli
data_labeled = tokenizer.special_char(dataset_lab,
                                      "new_dataset_specialchar.txt")
data_unlabeled = tokenizer.special_char(
    dataset_unlab, "unlabeled_specialchar.txt", unlab=True)
'''
# se già salvati
data_labeled = read_new_dataset.readData(modified=True)
data_unlabeled = read_new_dataset.readUnlabeled(modified=True)
'''

# tokenizzare
tokens_unlabeled = tokenizer.tokenizer(data_unlabeled)[0]
tokens, post_pos, pos, ids = tokenizer.tokenizer(data_labeled)

# word2vec con unlabeled
tokensconcat = tokens_unlabeled + tokens
word2vec.w2v(tokensconcat, 1)

# embedding dataset per training
vect_dataset = word2vec.vectorizeDataset(tokens)

# feature linguistiche
ling_feat = word2vec.lingFeatures(post_pos)

# training
RNN.networkSettings(vect_dataset, ling_feat, longTR=True)
