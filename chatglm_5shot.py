import spacy


import tqdm
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
import torch
import random
import torch.utils.data as Data
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import argparse
import pytextrank
from typing import List, Dict
#from data_utils import Scorer, BatchAverage, FScoreMetric, CorpusBLEUMetric
device = 'cuda' if torch.cuda.is_available() else 'cpu'

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textrank")
def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
set_seed(2023)
def pos(s):
  doc = nlp(s)
  poslist = []
  for token in doc:
    poslist.append(token.pos_)
  return ' '.join(poslist)

def truc(s, max_ans_length = 50):
  return ' '.join(s.split()[:max_ans_length])

tokenizer = AutoTokenizer.from_pretrained("../pretrain_model/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("../pretrain_model/chatglm-6b", trust_remote_code=True).half().cuda()
model = model.eval()


train_data = torch.load('canard_train')
dev_data = torch.load('canard_dev')
test_data = torch.load('canard_test')
print(len(train_data), len(dev_data), len(test_data))
prediction = []
template = 'Rewrite an incomplete utterance into an utterance which is semantically equivalent but self-contained to be understood without context. The sentence structure and expression should be consistent. There are 3 examples:\n'


N_cand = 500
N_train = 500

train_sample = random.sample(range(len(train_data)), N_cand + N_train)
cand_index = train_sample[:N_cand]
train_index = train_sample[N_cand:]

candidcate_questions = [train_data[i]['cur'] for i in cand_index]
test_questions = [test_data[i]['cur'] for i in range(len(test_data))]

length = []
for i in range(len(candidcate_questions)):
  length.append(len(candidcate_questions[i].split()))
topk = torch.topk(torch.tensor(length), k = 5).indices
#print(topk)
topk = [cand_index[i] for i in topk]
for i in tqdm.tqdm(range(len(test_data))):
  prompt_list = []
  ind1 = topk[0]
  data1 = train_data[ind1]
  ind2 = topk[1]
  data2 = train_data[ind2]
  ind3 = topk[2]
  data3 = train_data[ind3]
  ind4 = topk[3]
  data4 = train_data[ind4]
  ind5 = topk[4]
  data5 = train_data[ind5]
  prompt = template + 'Input: Context: ' + data1['context'] + \
           ' Current utterance: ' + data1['cur']  + \
            '\nOutput: ' + data1['rewrite'] + \
           '\nInput: Context: ' + data2['context'] + \
           ' Current utterance: ' + data2['cur'] + \
           '\nOutput: ' + data2['rewrite'] + \
           '\nInput: Context: ' + data3['context'] + \
           ' Current utterance: ' + data3['cur'] + \
           '\nOutput: ' + data3['rewrite'] + \
           '\nInput: Context: ' + data4['context'] + \
           ' Current utterance: ' + data4['cur'] + \
           '\nOutput: ' + data4['rewrite'] + \
           '\nInput: Context: ' + data5['context'] + \
           ' Current utterance: ' + data5['cur'] + \
           '\nOutput: ' + data5['rewrite'] + \
           '\nInput: Context: ' + test_data[i]['context'] + \
           ' Current utterance: ' + test_data[i]['cur'] + \
           '\nOutput: ?'

  response, _ =  model.chat(tokenizer, prompt, history = [])
  pred = truc(response.lower().split('\n')[0])
  if pred == '':
    pred = 'hi'
  #print('i = ', i, 'pred = ', pred, 'gold = ', test_data[i]['rewrite'])
  prediction.append({'context': test_data[i]['context'], 'cur': test_data[i]['cur'], 'restate':test_data[i]['rewrite'],
                       'pred': pred})

  torch.save(prediction, './canard_data/length_chatglm_random_five_shot_prediction')


