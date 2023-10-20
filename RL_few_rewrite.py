import tqdm
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
import torch
import random
import torch.utils.data as Data
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import List, Dict
#from data_utils import Scorer, BatchAverage, FScoreMetric, CorpusBLEUMetric
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
set_seed(2024)
def post_process(s):
  return s.lower().split('\n')[0].replace('输入','').replace('输出','')
model = BertModel.from_pretrained('../pretrain_model/bert-base-chinese').to(device)
model = torch.nn.DataParallel(model)
tokenizer = BertTokenizer.from_pretrained('../pretrain_model/bert-base-chinese')

def encode(model, sentence):
  encoding = model(tokenizer(sentence, return_tensors='pt')['input_ids'].to(device)).last_hidden_state
  #print(tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors='pt')['input_ids'][0]))
  return torch.mean(encoding[0], dim=0)
def encode_questions(model, input_ids, masks):
  encoddings = model(input_ids).last_hidden_state
  #print(encoddings.shape, masks.shape)
  return torch.sum(encoddings * masks.unsqueeze(2), dim=1) / torch.sum(masks, dim=1).unsqueeze(1)
def cos_similarity(model, sentence1, sentence2):
  s1 = encode(model,sentence1)
  s2 = encode(model,sentence2)
  return torch.sum(s1 * s2) / (torch.norm(s1) * torch.norm(s2))
print(cos_similarity(model,'今 天 天 气 很 好', '今 天 很 不 错'))


train_data = torch.load('rewrite_train')
#dev_data = torch.load('canard_dev')
test_data = torch.load('rewrite_dev')

N_cand = 1000
N_train = 1000
train_sample = random.sample(range(len(train_data)), N_cand + N_train)
cand_index = train_sample[:N_cand]
train_index = train_sample[N_cand:]

candidcate_questions = [train_data[i]['cur'] for i in cand_index]
test_questions = [test_data[i]['cur'] for i in range(len(test_data))]


class trainset(Data.Dataset):
  def __init__(self):
    self.data = [train_data[i] for i in train_index]
    print("train len = ", len(self.data))
  def __len__(self):
    return len(self.data)
  def __getitem__(self, item):
    return self.data[item]

traindataset = trainset()
trainloader = Data.DataLoader(traindataset, batch_size=2, shuffle=True)


# class devset(Data.Dataset):
#   def __init__(self):
#     self.data = torch.load('multi_dev')
#     print("dev len = ", len(self.data))
#   def __len__(self):
#     return len(self.data)
#   def __getitem__(self, item):
#     return self.data[item]
#
# devdataset = devset()
# devloader = Data.DataLoader(devdataset, batch_size=3, shuffle=True)




from evaluate import eval
e = eval()
glm_tokenizer = AutoTokenizer.from_pretrained("../pretrain_model/chatglm-6b", trust_remote_code=True)
glm_model = AutoModel.from_pretrained("../pretrain_model/chatglm-6b", trust_remote_code=True).half().cuda()
glm_model = glm_model.eval()
#glm_model = torch.nn.DataParallel(glm_model)
prediction = []
template = '将不完整的话语改写为能在没有上下文的情况下被理解但语义相等的话语。句子结构和表达应保持一致。以下是5个例子：\n'
#N = 1000
baseline_score = 64.
n_epoch = 10
Loss = nn.KLDivLoss(reduction='batchmean')
lr = 1e-5
optimzer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimzer, mode='max', factor=0.5, patience = 5)

def truc(s, max_ans_length = 50):
  return ' '.join(s.split()[:max_ans_length])
record = []

bert_metric = 0.
bert_metrics = None
for n in range(n_epoch):
  model.train()
  tr_loss = 0.
  for batch in tqdm.tqdm(trainloader):
    tok = tokenizer(batch['cur'], padding=True, return_tensors='pt')
    encodings = encode_questions(model, tok['input_ids'].to(device),
                                            tok['attention_mask'].to(device))
    #print(len(index))
    sample_candidcates = candidcate_questions
    tok_res = tokenizer(sample_candidcates, padding=True, return_tensors='pt')
    candidcate_encodings = encode_questions(model, tok_res['input_ids'].to(device),
                                            tok_res['attention_mask'].to(device))
    candidcate_norms = torch.norm(candidcate_encodings, dim=1)
    norms = torch.norm(encodings, dim=1)
    simi_map = torch.abs_(torch.mm(encodings, candidcate_encodings.T)) / torch.mm(norms.reshape(-1, 1),
                                                                              candidcate_norms.reshape(1, -1))
    top5 = torch.topk(simi_map, k=5, dim=-1).indices
    # print("top3: ", top3.shape)
    for i in range(len(batch['cur'])):
      cur_top5 = [cand_index[j] for j in top5[i]]
      prompt_list = []
      data1 = train_data[cur_top5[0]]
      data2 = train_data[cur_top5[1]]
      data3 = train_data[cur_top5[2]]
      data4 = train_data[cur_top5[3]]
      data5 = train_data[cur_top5[4]]
      prompt = template + '输入：上下文：' + data1['context'] + \
               ' 当前话语：' + data1['cur'] + \
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
               '\n输入：上下文：' + batch['context'][i] + \
               ' 当前话语：' + batch['cur'][i] + \
               '\n输出？'

      # response, _ = glm_model.chat(glm_tokenizer, prompt, history=[], max_length = 2500)
      # pred = truc(response.lower().split('\n')[0])
      response, _ = glm_model.chat(glm_tokenizer, prompt, history=[])
      pred = post_process(response)
      if pred == '':
        pred = '你好'
      #print('i = ', i, 'top5 = ',cur_top5, top5[i],  'pred = ', response, 'gold = ', batch['rewrite'][i])
      e.evaluate_metrics(cur_str=[batch['cur'][i]], restate_str=[batch['rewrite'][i]],
                         predict_str=[pred])
      #print(e.get_metrics(reset=False))
    reward = 100 * e.get_metrics(reset=True)['ROUGE'] - baseline_score
    #print("reward = ", reward)
    presi_label = torch.zeros_like(simi_map).scatter_(1, top5, 1).to(device)
    #print(torch.nonzero(presi_label))
    eps = 1e-8
    loss = reward * Loss(F.log_softmax(simi_map + eps, dim=1), F.softmax(presi_label, dim=1))
    optimzer.zero_grad()
    loss.backward()
    optimzer.step()
    tr_loss += torch.abs_(loss).item()

  print("loss = ", tr_loss / len(trainloader))
  #torch.save(model, 'RL_model')
  model.eval()
  prediction = []
  for i in tqdm.tqdm(range(len(test_questions))):

    q_emb = encode(model, test_questions[i]).reshape(1,-1)
    #print(len(index))
    sample_candidcates = candidcate_questions
    tok_res = tokenizer(sample_candidcates, padding=True, return_tensors='pt')
    candidcate_encodings = encode_questions(model, tok_res['input_ids'].to(device), tok_res['attention_mask'].to(device))
    candidcate_norms = torch.norm(candidcate_encodings, dim=1)
    test_norms = torch.norm(q_emb, dim=1)
    simi_map = torch.abs_(torch.mm(q_emb, candidcate_encodings.T)) / torch.mm(test_norms.reshape(-1,1), candidcate_norms.reshape(1,-1))
    top5 = torch.topk(simi_map, k = 5, dim=-1).indices
    #print("top3: ", top3.shape)

    cur_top5 = [cand_index[i] for i in top5[0]]
    prompt_list = []
    data1 = train_data[cur_top5[0]]
    data2 = train_data[cur_top5[1]]
    data3 = train_data[cur_top5[2]]
    data4 = train_data[cur_top5[3]]
    data5 = train_data[cur_top5[4]]
    prompt = template + '输入：上下文：' + data1['context'] + \
             ' 当前话语：' + data1['cur'] + \
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
             '\n输出？'

    # response, _ = glm_model.chat(glm_tokenizer, prompt, history=[], max_length = 2500)
    # pred = truc(response.lower().split('\n')[0])
    response, _ = glm_model.chat(glm_tokenizer, prompt, history=[])
    pred = post_process(response)
    if pred == '':
      pred = '你好'
    #print('i = ', i, 'pred = ', response.lower().split('\n')[0], 'gold = ', test_data[i]['rewrite'])
    e.evaluate_metrics(cur_str=[test_data[i]['cur']], restate_str=[test_data[i]['rewrite']],
                       predict_str=[pred])
    prediction.append({'context': test_data[i]['context'], 'cur': test_data[i]['cur'], 'restate': test_data[i]['rewrite'],
                       'pred': pred})
  eval_metrics = e.get_metrics(reset=True)
  print("eval metrics = ", eval_metrics)
  scheduler.step(eval_metrics['ROUGE'])
  record.append({'epoch': n, 'loss': tr_loss / len(trainloader), 'metric':eval_metrics })
  torch.save(record, './chatglm_RL_five_shot_record_1000_rewrite_seed2023')
  if eval_metrics['ROUGE'] > bert_metric:
    bert_metric = eval_metrics['ROUGE']
    bert_metrics = eval_metrics
    torch.save(model, 'chatglm_RL_five_shot_model_1000_rewrite_seed2023')
    torch.save(prediction, './canard_data/chatglm_RL_five_shot_prediction_1000_rewrite_seed2023')
print("best metrics = ", bert_metrics)

# e = eval()
# for data in prediction:
#   e.evaluate_metrics(cur_str=[data['cur']], restate_str=[data['restate']], predict_str=[ data['pred'].lower().split('\n')[0]])
# print(e.get_metrics(reset=True))
