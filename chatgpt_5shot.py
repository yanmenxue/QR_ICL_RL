import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from transformers import AutoTokenizer, AutoModel
import torch
import random
import numpy as np
import openai
import tqdm
os.environ["http_proxy"]="127.0.0.1:7890"
os.environ["https_proxy"]="127.0.0.1:7890"
openai.api_key = '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'

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
set_seed(2024)
def truc(s, max_ans_length=50):
  return ' '.join(s.split()[:max_ans_length])

# response, history = model.chat(tokenizer, '''Rewrite an incomplete utterance into an utterance which is semantically equivalent but self-contained to be understood without context. The sentence structure and expression should be consistent.
#                            There is an example:
#                            Input: Context: anna politkovskaya, the murder remains unsolved , 2016. Current utterance: did they have any clues ?
#                            Ourput: did investigators have any clues in the unresolved murder of anna politkovskaya ?
#                            Input: Context: anna politkovskaya; the murder remains unsolved , 2016; did they have any clues ?; probably fsb ) are known to have targeted the webmail account of the murdered russian journalist anna politkovskaya . Current utterance:  how did they target her email ?
#                            OUtput: ?''', history=[])
# print(response)


train_data = torch.load('task_train')
dev_data = torch.load('task_dev')
test_data = torch.load('task_dev')
print(len(train_data), len(dev_data), len(test_data))
prediction = []
template = 'Rewrite an incomplete utterance into an utterance which is semantically equivalent but self-contained to be understood without context. The sentence structure and expression should be consistent. There are 5 example:\n'

i = 0
while i < len(test_data):
  ind1 = random.randint(0, len(train_data) - 1)
  data1 = train_data[ind1]
  ind2 = random.randint(0, len(train_data) - 1)
  data2 = train_data[ind2]
  ind3 = random.randint(0, len(train_data) - 1)
  data3 = train_data[ind3]
  ind4 = random.randint(0, len(train_data) - 1)
  data4 = train_data[ind4]
  ind5 = random.randint(0, len(train_data) - 1)
  data5 = train_data[ind5]
  prompt = template + 'Input: Context: ' + data1['context'] + \
           ' Current utterance: ' + data1['cur'] + \
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

  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": prompt}
    ],
    temperature=0.0
  )
  pred = truc(response['choices'][0]['message']['content'].lower())

  #print('res = ', response)

  print('i = ', i , 'pred = ', pred, 'gold = ', test_data[i]['rewrite'])
  prediction.append({'context': test_data[i]['context'], 'cur': test_data[i]['cur'], 'restate':test_data[i]['rewrite'],
                       'pred': pred})
  i += 1

torch.save(prediction, './chatgpt_random_five_shot_task_prediction_task')