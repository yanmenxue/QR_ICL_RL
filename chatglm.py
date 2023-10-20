import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from transformers import AutoTokenizer, AutoModel
import torch
import random
tokenizer = AutoTokenizer.from_pretrained("../pretrain_model/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("../pretrain_model/chatglm-6b", trust_remote_code=True).half().cuda()
model = model.eval()
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


for i in range(len(test_data)):
  prompt_list = []
  ind = random.randint(0, len(train_data)-1)
  data = train_data[ind]
  prompt = template + 'Input: Context: ' + data['context'] + \
           ' Current utterance: ' + data['cur']  + \
            '\nOutput: ' + data['rewrite'] + \
            '\nInput: Context: ' + test_data[i]['context'] + \
           ' Current utterance: ' + test_data[i]['cur'] + \
           '\nOutput: ?'

  response, _ =  model.chat(tokenizer, prompt, history = [])
  print('i = ', i, 'pred = ', response, 'gold = ', test_data[i]['rewrite'])
  prediction.append({'context': test_data[i]['context'], 'cur': test_data[i]['cur'], 'restate':test_data[i]['rewrite'],
                       'pred': response})

  torch.save(prediction, './canard_data/chatglm_random_one_shot_prediction')


