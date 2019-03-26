import json

'''
    aggiunto il booleano modified per scegliere quale dataset caricare
'''
def readData(modified=False):
    if modified:
        with open('datasets/new_dataset_specialchar.txt') as jf:
            data = json.load(jf)
    else:
        with open('datasets/new_dataset.txt') as jf:
            data = json.load(jf)
    return data

def readUnlabeled(modified = False):
    if modified:
        with open('datasets/unlabeled_specialchar.txt') as jf:
            data = json.load(jf)
    else:
        with open('datasets/new_dataset_unlabeled.txt') as jf:
            data = json.load(jf)
    return data

d = readData()

