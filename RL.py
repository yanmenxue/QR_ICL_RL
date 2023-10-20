import tqdm
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel, T5ForConditionalGeneration
import torch
import random
import torch.utils.data as Data
import torch.nn as nn
import numpy as np
from typing import List, Dict
#from data_utils import Scorer, BatchAverage, FScoreMetric, CorpusBLEUMetric
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
set_seed(2023)

model = BertModel.from_pretrained('../pretrain_model/sentence-bert').to(device)
model = torch.nn.DataParallel(model)
tokenizer = BertTokenizer.from_pretrained('../pretrain_model/sentence-bert')

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


class devset(Data.Dataset):
  def __init__(self):
    self.data = torch.load('canard_dev')
    print("len = ", len(self.data))
  def __len__(self):
    return len(self.data)
  def __getitem__(self, item):
    return self.data[item]

devdataset = devset()
devloader = Data.DataLoader(devdataset, batch_size=2, shuffle=True)


print(cos_similarity(model,'It was a great day', 'Today was awesome'))
train_data = torch.load('canard_train')
#dev_data = torch.load('canard_dev')
test_data = torch.load('canard_test')
candidcate_questions = [train_data[i]['cur'] for i in range(len(train_data))]
test_questions = [test_data[i]['cur'] for i in range(len(test_data))]
from evaluate import eval
e = eval()

glm_tokenizer = AutoTokenizer.from_pretrained("../pretrain_model/chatglm-6b", trust_remote_code=True)
glm_model = AutoModel.from_pretrained("../pretrain_model/chatglm-6b", trust_remote_code=True).half().cuda()
glm_model = glm_model.eval()
#glm_model = torch.nn.DataParallel(glm_model)
prediction = []
template = 'Rewrite an incomplete utterance into an utterance which is semantically equivalent but self-contained to be understood without context. The sentence structure and expression should be consistent. There are 5 examples:\n'
c = range(len(candidcate_questions))
N = 1000
baseline_score = 52.
n_epoch = 10
Loss = nn.CrossEntropyLoss()
lr = 3e-5
optimzer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimzer, mode='max', factor=0.5, patience = 5)
for n in range(n_epoch):
  model.train()
  tr_loss = 0.
  for batch in tqdm.tqdm(devloader):
    index = random.sample(c, N)
    dev_tok = tokenizer(batch['cur'], padding=True, return_tensors='pt')
    dev_encodings = encode_questions(model, dev_tok['input_ids'].to(device),
                                            dev_tok['attention_mask'].to(device))
    #print(len(index))
    sample_candidcates = [candidcate_questions[i] for i in index]
    tok_res = tokenizer(sample_candidcates, padding=True, return_tensors='pt')
    candidcate_encodings = encode_questions(model, tok_res['input_ids'].to(device),
                                            tok_res['attention_mask'].to(device))
    candidcate_norms = torch.norm(candidcate_encodings, dim=1)
    dev_norms = torch.norm(dev_encodings, dim=1)
    simi_map = torch.abs_(torch.mm(dev_encodings, candidcate_encodings.T)) / torch.mm(dev_norms.reshape(-1, 1),
                                                                              candidcate_norms.reshape(1, -1))
    top5 = torch.topk(simi_map, k=5, dim=-1).indices
    # print("top3: ", top3.shape)
    for i in range(len(batch['cur'])):
      cur_top5 = [index[j] for j in top5[i]]
      prompt_list = []
      data1 = train_data[cur_top5[0]]
      data2 = train_data[cur_top5[1]]
      data3 = train_data[cur_top5[2]]
      data4 = train_data[cur_top5[3]]
      data5 = train_data[cur_top5[4]]
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
               '\nInput: Context: ' + batch['context'][i] + \
               ' Current utterance: ' + batch['cur'][i] + \
               '\nOutput: ?'

      response, _ = glm_model.chat(glm_tokenizer, prompt, history=[])
      #print('i = ', i, 'top5 = ',cur_top5, top5[i],  'pred = ', response, 'gold = ', batch['rewrite'][i])
      e.evaluate_metrics(cur_str=[batch['cur'][i]], restate_str=[batch['rewrite'][i]],
                         predict_str=[response.lower().split('\n')[0]])
      #print(e.get_metrics(reset=False))
    reward = 100 * e.get_metrics(reset=True)['ROUGE'] - baseline_score
    #print("reward = ", reward)
    presi_label = torch.zeros_like(simi_map).scatter_(1, top5, 1).to(device)
    #print(torch.nonzero(presi_label))
    loss = reward * Loss(simi_map, presi_label)
    optimzer.zero_grad()
    loss.backward()
    optimzer.step()
    tr_loss += loss.item()

  print("loss = ", tr_loss / len(devloader))
  torch.save(model, 'RL_model')
  model.eval()
  prediction = []
  for i in tqdm.tqdm(range(len(test_questions))):

    index = random.sample(c, N)
    q_emb = encode(model, test_questions[i]).reshape(1,-1)
    #print(len(index))
    sample_candidcates = [candidcate_questions[i] for i in index]
    tok_res = tokenizer(sample_candidcates, padding=True, return_tensors='pt')
    candidcate_encodings = encode_questions(model, tok_res['input_ids'].to(device), tok_res['attention_mask'].to(device))
    candidcate_norms = torch.norm(candidcate_encodings, dim=1)
    test_norms = torch.norm(q_emb, dim=1)
    simi_map = torch.abs_(torch.mm(q_emb, candidcate_encodings.T)) / torch.mm(test_norms.reshape(-1,1), candidcate_norms.reshape(1,-1))
    top5 = torch.topk(simi_map, k = 5, dim=-1).indices
    #print("top3: ", top3.shape)

    cur_top5 = [index[i] for i in top5[0]]
    prompt_list = []
    data1 = train_data[cur_top5[0]]
    data2 = train_data[cur_top5[1]]
    data3 = train_data[cur_top5[2]]
    data4 = train_data[cur_top5[3]]
    data5 = train_data[cur_top5[4]]
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

    response, _ = glm_model.chat(glm_tokenizer, prompt, history=[])
    #print('i = ', i, 'pred = ', response, 'gold = ', test_data[i]['rewrite'])
    e.evaluate_metrics(cur_str=[test_data[i]['cur']], restate_str=[test_data[i]['rewrite']],
                       predict_str=[response.lower().split('\n')[0]])
    prediction.append({'context': test_data[i]['context'], 'cur': test_data[i]['cur'], 'restate': test_data[i]['rewrite'],
                       'pred': response})
  eval_metrics = e.get_metrics(reset=True)
  print("eval metrics = ", eval_metrics)
  scheduler.step(eval_metrics['ROUGE'])
  torch.save(prediction, './canard_data/chatglm_RL_five_shot_prediction')
# e = eval()
# for data in prediction:
#   e.evaluate_metrics(cur_str=[data['cur']], restate_str=[data['restate']], predict_str=[ data['pred'].lower().split('\n')[0]])
# print(e.get_metrics(reset=True))