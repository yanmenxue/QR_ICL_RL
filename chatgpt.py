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
set_seed(2023)
# response, history = model.chat(tokenizer, '''Rewrite an incomplete utterance into an utterance which is semantically equivalent but self-contained to be understood without context. The sentence structure and expression should be consistent.
#                            There is an example:
#                            Input: Context: anna politkovskaya, the murder remains unsolved , 2016. Current utterance: did they have any clues ?
#                            Ourput: did investigators have any clues in the unresolved murder of anna politkovskaya ?
#                            Input: Context: anna politkovskaya; the murder remains unsolved , 2016; did they have any clues ?; probably fsb ) are known to have targeted the webmail account of the murdered russian journalist anna politkovskaya . Current utterance:  how did they target her email ?
#                            OUtput: ?''', history=[])
# print(response)


train_data = torch.load('canard_train')
dev_data = torch.load('canard_dev')
test_data = torch.load('canard_test')
print(len(train_data), len(dev_data), len(test_data))
prediction = []
template = 'Rewrite an incomplete utterance into an utterance which is semantically equivalent but self-contained to be understood without context. The sentence structure and expression should be consistent. There is an example:\n'

i = 0
while i < len(test_data):
  prompt_list = []

  ind = random.randint(0, len(train_data)-1)
  data = train_data[ind]
  prompt = template + 'Input: Context: ' + data['context'] + \
           ' Current utterance: ' + data['cur']  + \
            '\nOutput: ' + data['rewrite'] + \
            '\nInput: Context: ' + test_data[i]['context'] + \
           ' Current utterance: ' + test_data[i]['cur'] + \
           '\nOutput: ?'

  #print(prompt)
  response = openai.Completion.create(
      model="text-davinci-003",
      prompt=prompt,
      temperature=0.0,
      max_tokens=100,
      top_p=1.0,
      frequency_penalty=0.0,
      presence_penalty=0.0
    )['choices']

  print('i = ', i, 'pred = ', response[0]['text'], 'gold = ', test_data[i]['rewrite'])
  prediction.append({'context': test_data[i]['context'], 'cur': test_data[i]['cur'], 'restate':test_data[i]['rewrite'],
                       'pred': response[0]['text']})
  i += 1

  torch.save(prediction, './canard_data/gpt3.5_random_one_shot_prediction')