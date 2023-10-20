import os
from transformers import AutoTokenizer, AutoModel
import torch
import random
import numpy as np
import openai
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch.utils.data as Data
import tqdm
os.environ["http_proxy"]="127.0.0.1:7890"
os.environ["https_proxy"]="127.0.0.1:7890"
openai.api_key = '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'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
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



class testset(Data.Dataset):
  def __init__(self):
    self.data = torch.load('canard_test')
    print("len = ", len(self.data))
  def __len__(self):
    return len(self.data)
  def __getitem__(self, item):
    return self.data[item]

testdataset = testset()
testloader = Data.DataLoader(testdataset, batch_size=1, shuffle=False)

model = AutoModelForSeq2SeqLM.from_pretrained("../pretrain_model/fast_chat").to(device)
tokenizer = AutoTokenizer.from_pretrained("../pretrain_model/fast_chat")
model = model.eval()
prediction = []
template = 'Rewrite an incomplete utterance into an utterance which is semantically equivalent but self-contained to be understood without context. The sentence structure and expression should be consistent. There is an example:\n'

i = 0
for batch in tqdm.tqdm(testloader):
  prompt_list = []
  for j in range(len(batch['context'])):
    ind = random.randint(0, len(train_data)-1)
    data = train_data[ind]
    prompt = template + 'Input: Context: ' + data['context'] + \
             ' Current utterance: ' + data['cur']  + \
              '\nOutput: ' + data['rewrite'] + \
              '\nInput: Context: ' + batch['context'][j] + \
             ' Current utterance: ' + batch['cur'][j] + \
             '\nOutput: ?'
    prompt_list.append(prompt)

  inputs = tokenizer(prompt_list, return_tensors="pt",padding=True)
  outputs = model.generate(input_ids = inputs['input_ids'].to(device), max_length =  100)

  for j in range(len(batch['context'])):
    pred = ' '.join(tokenizer.batch_decode(outputs, skip_special_tokens=True)[j].split()[1:])
    print('j = ', j , 'pred = ', pred, 'gold = ', batch['rewrite'][j])
    prediction.append({'context': batch['context'][j], 'cur': batch['cur'][j], 'restate':batch['rewrite'][j],
                         'pred': pred})


torch.save(prediction, './canard_data/fast-chat_random_one_shot_prediction')