import json
import jsonlines
import io
import pickle
import numpy as np
import tokenize


def create_dataset(data, name):
    path = "datasets/"+name
    with open(path, 'w') as outfile:
        json.dump(data, outfile)


def getTestData(file):
    datas = []
    new_dataset = []

    i=0
    with jsonlines.open(file) as reader:
        for instances in reader:
            datas.append(instances)
            i = i+1

    i=0
    for d in datas:
        new_dataset.append({
            'id': datas[i]['id'],
            'postText': datas[i]['postText']
        })
        i=i+1

    return new_dataset

def getData():
    datas = []
    target = []

    i=0
    file_inst = 'datasets/1/instances.jsonl'
    with jsonlines.open(file_inst) as reader:
        for instances in reader:
            datas.append(instances)

    file_inst2 = 'datasets/2/instances.jsonl'
    with jsonlines.open(file_inst2) as reader:
        for instances in reader:
            datas.append(instances)

    file_truth = 'datasets/1/truth.jsonl'
    with jsonlines.open(file_truth) as reader:
        i = 0
        nd = 0
        for truths in reader:
            target.append(truths)

    file_truth = 'datasets/2/truth.jsonl'
    with jsonlines.open(file_truth) as reader:
        i = 0
        nd = 0
        for truths in reader:
            target.append(truths)

    new_dataset = []
    i = 0
    dataset_idx = 0
    for t in target:
        dato = {}
        dato["truthMean"] = t["truthMean"]
        id_t = t["id"]
        for d in datas:
            id_i = d["id"]
            if (id_i == id_t):
                dato["postText"] = d["postText"]
                dato["id"] = id_i
                new_dataset.append(dato)
                dataset_idx += 1
                break
        i += 1

    return new_dataset


def getDataUnlabeled():

    new_dataset = []

    i = 0
    file_inst = 'datasets/3/instances.jsonl'
    with jsonlines.open(file_inst) as reader:
        for instances in reader:
            new_dataset.append({
                'id': instances['id'],
                'postText': instances['postText'],
                'targetTitle': instances['targetTitle'],
                'targetParagraphs': instances['targetParagraphs']
            })
            i = i + 1
    return new_dataset
