import os
from transformers import AutoTokenizer, AutoModel
import torch
import random
import numpy as np
tokenizer = AutoTokenizer.from_pretrained("../pretrain_model/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("../pretrain_model/chatglm-6b", trust_remote_code=True).half().cuda()
model = model.eval()

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
set_seed(2023)

def truc(s, max_ans_length=50):
  return ' '.join(s.split()[:max_ans_length])

train_data = torch.load('task_train')
dev_data = torch.load('task_dev')
test_data = torch.load('task_dev')
print(len(train_data), len(dev_data), len(test_data))
prediction = []
template = 'Rewrite an incomplete utterance into an utterance which is semantically equivalent but self-contained to be understood without context. The sentence structure and expression should be consistent. There are 3 examples:\n'


for i in range(len(test_data)):
  prompt_list = []
  ind1 = random.randint(0, len(train_data)-1)
  data1 = train_data[ind1]
  ind2 = random.randint(0, len(train_data) - 1)
  data2 = train_data[ind2]
  ind3 = random.randint(0, len(train_data) - 1)
  data3 = train_data[ind3]
  prompt = template + 'Input: Context: ' + data1['context'] + \
           ' Current utterance: ' + data1['cur']  + \
            '\nOutput: ' + data1['rewrite'] + \
           '\nInput: Context: ' + data2['context'] + \
           ' Current utterance: ' + data2['cur'] + \
           '\nOutput: ' + data2['rewrite'] + \
           '\nInput: Context: ' + data3['context'] + \
           ' Current utterance: ' + data3['cur'] + \
           '\nOutput: ' + data3['rewrite'] + \
           '\nInput: Context: ' + test_data[i]['context'] + \
           ' Current utterance: ' + test_data[i]['cur'] + \
           '\nOutput: ?'

  response, _ =  model.chat(tokenizer, prompt, history = [])
  pred = truc(response.lower().split('\n')[0])
  print('i = ', i, 'pred = ', pred, 'gold = ', test_data[i]['rewrite'])
  prediction.append({'context': test_data[i]['context'], 'cur': test_data[i]['cur'], 'restate':test_data[i]['rewrite'],
                       'pred': pred})

  torch.save(prediction, './chatglm_random_three_shot_prediction_task_seed')


