import tqdm
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
import torch
import random
import torch.utils.data as Data
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import argparse
from typing import List, Dict
#from data_utils import Scorer, BatchAverage, FScoreMetric, CorpusBLEUMetric
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

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
# print(cos_similarity(model,'It was a great day', 'Today was awesome'))


if __name__ == '__main__':
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("--dataset",  type=str, choices=['canard','task','rewrite'],
                          help="Please specify a model file to evaluate")
  arg_parser.add_argument("--seed", type=int,
                          help="Please specify a model file to evaluate")
  arg_parser.add_argument("--num_cand", type=int,
                          help="Please specify a model file to evaluate")
  arg_parser.add_argument("--num_train", type=int,
                          help="Please specify a model file to evaluate")
  arg_parser.add_argument("--lr", type=float, default=3e-5,
                          help="Please specify a model file to evaluate")
  arg_parser.add_argument("--num_epoch", type=int, default=10,
                          help="Please specify a model file to evaluate")
  arg_parser.add_argument("--baseline_score", type=float,
                          help="Please specify a model file to evaluate")
  arg_parser.add_argument("--max_ans_length", type=int, default=50,
                          help="Please specify a model file to evaluate")
  arg_parser.add_argument("--topk", type=int, default=5,
                          help="Please specify a model file to evaluate")
  arg_parser.add_argument("--weight_decay", type=float, default=0.01,
                          help="Please specify a model file to evaluate")
  arg_parser.add_argument("--factor", type=float, default=0.5,
                          help="Please specify a model file to evaluate")
  arg_parser.add_argument("--patience", type=int, default=5,
                          help="Please specify a model file to evaluate")
  arg_parser.add_argument("--btsize", type=int, default=2,
                          help="Please specify a model file to evaluate")

  parsed_args = arg_parser.parse_args()

  set_seed(parsed_args.seed)

  if parsed_args.dataset =='canard':
    train_data = torch.load('canard_train')
    test_data = torch.load('canard_test')
  elif parsed_args.dataset =='task':
    train_data = torch.load('task_train')
    test_data = torch.load('task_dev')
  elif parsed_args.dataset =='rewrite':
    train_data = torch.load('rewrite_train')
    test_data = torch.load('rewrite_dev')

  if parsed_args.dataset == 'canard' or parsed_args.dataset == 'task':
    model = BertModel.from_pretrained('../pretrain_model/sentence-bert').to(device)
    model = torch.nn.DataParallel(model)
    tokenizer = BertTokenizer.from_pretrained('../pretrain_model/sentence-bert')
  elif parsed_args.dataset =='rewrite':
    model = BertModel.from_pretrained('../pretrain_model/bert-base-chinese').to(device)
    model = torch.nn.DataParallel(model)
    tokenizer = BertTokenizer.from_pretrained('../pretrain_model/bert-base-chinese')

  N_cand = parsed_args.num_cand
  N_train = parsed_args.num_train

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
  trainloader = Data.DataLoader(traindataset, batch_size=parsed_args.btsize, shuffle=True)


  # class devset(Data.Dataset):
  #   def __init__(self):
  #     self.data = dev_data
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
  template_en = 'Rewrite an incomplete utterance into an utterance which is semantically equivalent but self-contained to be understood without context. The sentence structure and expression should be consistent. There are ' + str(parsed_args.topk) + ' examples:'
  template_zh = '将不完整的话语改写为能在没有上下文的情况下被理解但语义相等的话语。句子结构和表达应保持一致。以下是' + str(parsed_args.topk) + '个例子：'
  #N = 1000
  baseline_score = parsed_args.baseline_score
  n_epoch = parsed_args.num_epoch
  Loss = nn.KLDivLoss(reduction='batchmean')
  lr = parsed_args.lr
  optimzer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=parsed_args.weight_decay)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimzer, mode='max', factor=parsed_args.factor, patience = parsed_args.patience)

  def truc(s, max_ans_length = parsed_args.max_ans_length):
    return ' '.join(s.split()[:max_ans_length])
  record = []

  bert_metric = 0.
  bert_metrics = None
  for n in range(n_epoch):
    #test pretrained model
    for i in tqdm.tqdm(range(len(test_questions))):

      q_emb = encode(model, test_questions[i]).reshape(1,-1)
      #print(len(index))
      sample_candidcates = candidcate_questions
      tok_res = tokenizer(sample_candidcates, padding=True, return_tensors='pt')
      candidcate_encodings = encode_questions(model, tok_res['input_ids'].to(device), tok_res['attention_mask'].to(device))
      candidcate_norms = torch.norm(candidcate_encodings, dim=1)
      test_norms = torch.norm(q_emb, dim=1)
      simi_map = torch.abs_(torch.mm(q_emb, candidcate_encodings.T)) / torch.mm(test_norms.reshape(-1,1), candidcate_norms.reshape(1,-1))
      top = torch.topk(simi_map, k = parsed_args.topk, dim=-1).indices
      #print("top3: ", top3.shape)

      cur_top = [cand_index[i] for i in top[0]]
      prompt_list = []
      if parsed_args.dataset == 'canard' or parsed_args.dataset == 'task':
        prompt = template_en
        for k in range(parsed_args.topk-1, -1, -1):
          data = train_data[cur_top[k]]
          prompt += '\nInput: Context: ' + data['context'] + \
                    ' Current utterance: ' + data['cur'] + \
                    '\nOutput: ' + data['rewrite']
        prompt += '\nInput: Context: ' + test_data[i]['context'] + \
                 ' Current utterance: ' + test_data[i]['cur'] + \
                 '\nOutput: ?'
      elif parsed_args.dataset == 'rewrite':
        prompt = template_zh
        for k in range(parsed_args.topk-1, -1, -1):
          data = train_data[cur_top[k]]
          prompt += '\n输入：上下文：' + data['context'] + \
             ' 当前话语：' + data['cur'] + \
             '\n输出：' + data['rewrite']
        prompt += '\n输入：上下文：' + test_data[i]['context'] + \
           ' 当前话语：' + test_data[i]['cur'] + \
           '\n输出：？'

      # response, _ = glm_model.chat(glm_tokenizer, prompt, history=[], max_length = 2500)
      # pred = truc(response.lower().split('\n')[0])
      response, _ = glm_model.chat(glm_tokenizer, prompt, history=[])
      pred = truc(response.lower().split('\n')[0])
      if pred == '':
        pred = 'hi'
      #print('i = ', i, 'pred = ', response.lower().split('\n')[0], 'gold = ', test_data[i]['rewrite'])
      e.evaluate_metrics(cur_str=[test_data[i]['cur']], restate_str=[test_data[i]['rewrite']],
                         predict_str=[pred])
      prediction.append({'context': test_data[i]['context'], 'cur': test_data[i]['cur'], 'restate': test_data[i]['rewrite'],
                         'pred': pred})
    eval_metrics = e.get_metrics(reset=True)
    print("eval metrics = ", eval_metrics)

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
      top = torch.topk(simi_map, k=parsed_args.topk, dim=-1).indices
      # print("top3: ", top3.shape)
      for i in range(len(batch['cur'])):
        cur_top = [cand_index[j] for j in top[i]]
        prompt_list = []
        if parsed_args.dataset == 'canard' or parsed_args.dataset == 'task':
          prompt = template_en
          for k in range(parsed_args.topk-1, -1, -1):
            data = train_data[cur_top[k]]
            prompt += '\nInput: Context: ' + data['context'] + \
                   ' Current utterance: ' + data['cur']+ \
                   '\nOutput: ' + data['rewrite']
          prompt += '\nInput: Context: ' + batch['context'][i] + \
                   ' Current utterance: ' + batch['cur'][i] + \
                   '\nOutput: ?'
        elif parsed_args.dataset == 'rewrite':
          prompt = template_zh
          for k in range(parsed_args.topk-1, -1, -1):
            data = train_data[cur_top[k]]
            prompt +=  '\n输入：上下文：' + data['context'] + \
               ' 当前话语：' + data['cur'] + \
               '\n输出：' + data['rewrite']
          prompt += '\n输入：上下文：' + batch['context'][i] + \
               ' 当前话语：' + batch['cur'][i] + \
               '\n输出：？'

        # response, _ = glm_model.chat(glm_tokenizer, prompt, history=[], max_length = 2500)
        # pred = truc(response.lower().split('\n')[0])
        response, _ = glm_model.chat(glm_tokenizer, prompt, history=[])
        pred = truc(response.lower().split('\n')[0])
        if pred == '':
          pred = 'hi'
        #print('i = ', i, 'top5 = ',cur_top5, top5[i],  'pred = ', response, 'gold = ', batch['rewrite'][i])
        e.evaluate_metrics(cur_str=[batch['cur'][i]], restate_str=[batch['rewrite'][i]],
                           predict_str=[pred])
        #print(e.get_metrics(reset=False))
      reward = 100 * e.get_metrics(reset=True)['ROUGE'] - baseline_score
      #print("reward = ", reward)
      presi_label = torch.zeros_like(simi_map).scatter_(1, top, 1).to(device)
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
      top = torch.topk(simi_map, k = parsed_args.topk, dim=-1).indices
      #print("top3: ", top3.shape)

      cur_top = [cand_index[i] for i in top[0]]
      prompt_list = []
      if parsed_args.dataset == 'canard' or parsed_args.dataset == 'task':
        prompt = template_en
        for k in range(parsed_args.topk-1, -1, -1):
          data = train_data[cur_top[k]]
          prompt += '\nInput: Context: ' + data['context'] + \
                    ' Current utterance: ' + data['cur'] + \
                    '\nOutput: ' + data['rewrite']
        prompt += '\nInput: Context: ' + test_data[i]['context'] + \
                 ' Current utterance: ' + test_data[i]['cur'] + \
                 '\nOutput: ?'
      elif parsed_args.dataset == 'rewrite':
        prompt = template_zh
        for k in range(parsed_args.topk-1, -1, -1):
          data = train_data[cur_top[k]]
          prompt += '\n输入：上下文：' + data['context'] + \
             ' 当前话语：' + data['cur'] + \
             '\n输出：' + data['rewrite']
        prompt += '\n输入：上下文：' + test_data[i]['context'] + \
           ' 当前话语：' + test_data[i]['cur'] + \
           '\n输出：？'

      # response, _ = glm_model.chat(glm_tokenizer, prompt, history=[], max_length = 2500)
      # pred = truc(response.lower().split('\n')[0])
      response, _ = glm_model.chat(glm_tokenizer, prompt, history=[])
      pred = truc(response.lower().split('\n')[0])
      if pred == '':
        pred = 'hi'
      #print('i = ', i, 'pred = ', response.lower().split('\n')[0], 'gold = ', test_data[i]['rewrite'])
      e.evaluate_metrics(cur_str=[test_data[i]['cur']], restate_str=[test_data[i]['rewrite']],
                         predict_str=[pred])
      prediction.append({'context': test_data[i]['context'], 'cur': test_data[i]['cur'], 'restate': test_data[i]['rewrite'],
                         'pred': pred})
    eval_metrics = e.get_metrics(reset=True)
    print("eval metrics = ", eval_metrics)
    scheduler.step(eval_metrics['ROUGE'])
    record.append({'epoch': n, 'loss': tr_loss / len(trainloader), 'metric':eval_metrics })
    torch.save(record, './order_chatglm_RL_'+str(parsed_args.topk) + '_shot_record_' + str(parsed_args.dataset) + '_cand' + str(parsed_args.num_cand) + '_train' + str(parsed_args.num_train) + '_seed' + str(parsed_args.seed))
    if eval_metrics['ROUGE'] > bert_metric:
      bert_metric = eval_metrics['ROUGE']
      bert_metrics = eval_metrics
      torch.save(model, './order_chatglm_RL_' + str(parsed_args.topk)+'_shot_model_' + str(parsed_args.dataset) + '_cand' + str(parsed_args.num_cand) + '_train' + str(parsed_args.num_train) + '_seed' + str(parsed_args.seed))
      torch.save(prediction, './order_chatglm_RL_'+str(parsed_args.topk) + '_shot_prediction_' + str(parsed_args.dataset) + '_cand' + str(parsed_args.num_cand) + '_train' + str(parsed_args.num_train) + '_seed' + str(parsed_args.seed))
  print("best metrics = ", bert_metrics)

# e = eval()
# for data in prediction:
#   e.evaluate_metrics(cur_str=[data['cur']], restate_str=[data['restate']], predict_str=[ data['pred'].lower().split('\n')[0]])
# print(e.get_metrics(reset=True))
