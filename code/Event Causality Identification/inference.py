import torch
import time
import math
import csv
import logging

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

from itertools import islice

# from transformers import  BartForConditionalGeneration
from modeling_bart import  BartForConditionalGeneration
from seq2seq_model import Seq2SeqModel,AutoTokenizer
# from model import BartForConditionalGeneration


class InputExample():
    def __init__(self, input_TXT, event1, event2, labels, type):
        self.input_TXT = input_TXT
        self.event1 = event1
        self.event2 = event2
        self.labels = labels
        self.type = type

def predict_relation(input_TXT, event1, event2):  # 预测一个句子中两个事件的关系
    input_TXT = [input_TXT]*2
    input_ids = tokenizer(input_TXT, return_tensors='pt')['input_ids']
    model.to(device)
    
    relation_dict = {0: 'Cause-Effect', 1: 'NONE'}
    temp_list = []
    
    temp_list.append(event1+"is the cause of "+event2)
    temp_list.append(event1+"has no relation to "+event2)
    output_ids = tokenizer(temp_list, return_tensors='pt',
                           padding=True, truncation=True)['input_ids']
    # 加一个unused字符
    output_ids[:, 0] = 2
    output_length_list = [0]*2

    base_length = ((tokenizer(temp_list[0], return_tensors='pt', padding=True, truncation=True)[
                   'input_ids']).shape)[1]-2

    output_length_list[0:1] = [base_length]*2

    score = [1]*2
    with torch.no_grad():
        
        outputs = model(input_ids=input_ids.to(device), decoder_input_ids=output_ids[:, :output_ids.shape[1] - 2].to(device))
        output = outputs[0]
        # print(tokenizer.decode(output_ids[1, :output_ids.shape[1] - 2]))
        for i in range(output_ids.shape[1] - 3):
        # output = model(input_ids=input_ids.to(device), decoder_input_ids=output_ids.to(device))[0]

            logits = output[:, i, :]
            logits = logits.softmax(dim=1)
            logits = logits.to('cpu').numpy()
            # for j in range(0, 2):
            #     if int(output_ids[j][i + 1]) not in [16, 34, 5, 117, 1303, 9355, 9, 7]:
            #         weight = 1
            #     else:
            #         weight = 1.5
            #     if i < output_length_list[j]:
            #         score[j] = score[j] * (logits[j][int(output_ids[j][i + 1])] ** weight)
            for j in range(0, 2):
                if i < output_length_list[j]:
                    score[j] = score[j] * logits[j][int(output_ids[j][i + 1])]  
    # print(temp_list[0],score[0])
    # print(temp_list[1],score[1])
    # print(relation_dict[(score.index(max(score)))])

    return relation_dict[(score.index(max(score)))]


def cal_time(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# path = "./outputs/best_model"
path = "../Bart-base"
# path = "./exp/eventstoryline_regu"

# path = "./exp/eventstoryline_relpos_regu/checkpoint-534-epoch-3"

tokenizer = AutoTokenizer.from_pretrained(path)

model = BartForConditionalGeneration.from_pretrained(path)


model.eval()
model.config.use_cache = False
# input_ids = tokenizer(input_TXT, return_tensors='pt')['input_ids']
# print(input_ids)
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
examples = []

# f = open('./data/event_train_eventstoryline.csv', 'r',encoding='utf-8')
f = open('./data/test_eventstoryline.csv', 'r',encoding='utf-8')
with f:
    reader = csv.reader(f)
    for row in islice(reader, 1, None):
        input_TXT = row[0]
        event1 = row[2]
        event2 = row[3]
        labels = row[4]
        type = row[5]
        examples.append(InputExample(input_TXT=input_TXT, event1=event1, event2=event2, labels=labels, type=type))


trues_list = []
preds_list = []
trues_list_intra = []
preds_list_intra = []
trues_list_inter = []
preds_list_inter = []
num_01 = len(examples)
num_point = 0
start = time.time()
for example in examples:
    pre_res = predict_relation(example.input_TXT,example.event1, example.event2)
    if example.type == 'intra':
        preds_list_intra.append(pre_res)
        trues_list_intra.append(example.labels)
    else:
        preds_list_inter.append(pre_res)
        trues_list_inter.append(example.labels)

    preds_list.append(pre_res)
    trues_list.append(example.labels)
    # print('%d/%d (%s)' % (num_point+1, num_01, cal_time(start)))
    # print('Pred:', preds_list[num_point])
    # print('Gold:', trues_list[num_point])
    num_point += 1

# intra
print(classification_report(trues_list_intra,preds_list_intra))

intra_results = {
   
    "P": precision_score(trues_list_intra, preds_list_intra,average=None),
    "R": recall_score(trues_list_intra, preds_list_intra,average=None),
    "F": f1_score(trues_list_intra, preds_list_intra,average=None),
}
print(intra_results)

# inter
print(classification_report(trues_list_inter,preds_list_inter))

inter_results = {
    
    "P": precision_score(trues_list_inter, preds_list_inter,average=None),
    "R": recall_score(trues_list_inter, preds_list_inter,average=None),
    "F": f1_score(trues_list_inter, preds_list_inter,average=None)
}
print(inter_results)

# all
print(classification_report(trues_list,preds_list))

results = {
    # "F": f1_score(trues_list, preds_list,average='weighted'),
    # "P": precision_score(trues_list, preds_list,average='weighted'),
    # "R": recall_score(trues_list, preds_list,average='weighted')
    # average has to be one of (None, 'micro', 'macro', 'weighted', 'samples')
    
    "P": precision_score(trues_list, preds_list,average=None),
    "R": recall_score(trues_list, preds_list,average=None),
    "F": f1_score(trues_list, preds_list,average=None)
}
print(results)



