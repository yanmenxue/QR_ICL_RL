import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from transformers import AutoTokenizer, AutoModel
import torch
import random
import numpy as np
def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
set_seed(2024)
def post_process(s,max_length = 100):
  return s.lower().split('\n')[0].replace('输入','').replace('输出','')[:max_length]
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


train_data = torch.load('multi_train')
dev_data = torch.load('multi_dev')
test_data = torch.load('multi_test')
print(len(train_data), len(dev_data), len(test_data))
prediction = []
template = '将不完整的话语改写为能在没有上下文的情况下被理解但语义相等且自包含的话语。句子结构和表达应保持一致。以下是5个例子：\n'


for i in range(len(test_data)):
  prompt_list = []
  ind1 = random.randint(0, len(train_data)-1)
  data1 = train_data[ind1]
  ind2 = random.randint(0, len(train_data) - 1)
  data2 = train_data[ind2]
  ind3 = random.randint(0, len(train_data) - 1)
  data3 = train_data[ind3]
  ind4 = random.randint(0, len(train_data) - 1)
  data4 = train_data[ind4]
  ind5 = random.randint(0, len(train_data) - 1)
  data5 = train_data[ind5]
  prompt = template + '输入：上下文：' + data1['context'] + \
           ' 当前话语：' + data1['cur']  + \
            '\n输出：' + data1['rewrite'] + \
           '\n输入：上下文：' + data2['context'] + \
           ' 当前话语：' + data2['cur'] + \
           '\n输出：' + data2['rewrite'] + \
           '\n输入：上下文：' + data3['context'] + \
           ' 当前话语：' + data3['cur'] + \
           '\n输出：' + data3['rewrite'] + \
           '\n输入：上下文：' + data4['context'] + \
           ' 当前话语：' + data4['cur'] + \
           '\n输出：' + data4['rewrite'] + \
           '\n输入：上下文：' + data5['context'] + \
           ' 当前话语：' + data5['cur'] + \
           '\n输出：' + data5['rewrite'] + \
           '\n输入：上下文：' + test_data[i]['context'] + \
           ' 当前话语：' + test_data[i]['cur'] + \
           '\n输出是什么？'

  response, _ =  model.chat(tokenizer, prompt, history = [])
  pred = post_process(response)
  print('i = ', i, 'pred = ', pred, 'gold = ', test_data[i]['rewrite'])
  prediction.append({'context': test_data[i]['context'], 'cur': test_data[i]['cur'], 'restate':test_data[i]['rewrite'],
                       'pred': pred})

  torch.save(prediction, './canard_data/chatglm_random_five_shot_prediction_multi')


