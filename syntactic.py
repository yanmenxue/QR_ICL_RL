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

def pos(s):
  doc = nlp(s)
  poslist = []
  for token in doc:
    poslist.append(token.pos_)
  return ' '.join(poslist)

def find_lcsubstr(s1, s2):
  # 生成0矩阵，为方便后续计算，比字符串长度多了一列
  m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]
  mmax = 0  # 最长匹配的长度
  p = 0  # 最长匹配对应在s1中的最后一位
  for i in range(len(s1)):
    for j in range(len(s2)):
      if s1[i] == s2[j]:
        m[i + 1][j + 1] = m[i][j] + 1
        if m[i + 1][j + 1] > mmax:
          mmax = m[i + 1][j + 1]
          p = i + 1
  return s1[0:p-mmax] + s1[p:]


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


  class testset(Data.Dataset):
    def __init__(self):
      self.data = test_data
    def __len__(self):
      return len(self.data)
    def __getitem__(self, item):
      return self.data[item]

  testdataset = testset()
  testloader = Data.DataLoader(testdataset, batch_size=50, shuffle=False)

  from evaluate import eval
  e = eval()
  prediction = []
  template_en = 'Rewrite an incomplete utterance into an utterance which is semantically equivalent but self-contained to be understood without context. The sentence structure and expression should be consistent. There are ' + str(parsed_args.topk) + ' examples:'
  template_zh = '将不完整的话语改写为能在没有上下文的情况下被理解但语义相等的话语。句子结构和表达应保持一致。以下是' + str(parsed_args.topk) + '个例子：'
  #N = 1000
  def truc(s, max_ans_length = parsed_args.max_ans_length):
    return ' '.join(s.split()[:max_ans_length])


  model_rl = torch.load('./chatglm_RL_' + str(parsed_args.topk)+'_shot_model_' + str(parsed_args.dataset) + '_cand' + str(parsed_args.num_cand) + '_train' + str(parsed_args.num_train) + '_seed' + str(parsed_args.seed))
  model_rank = torch.load('./chatglm_rank_' + str(parsed_args.topk)+'_shot_model_' + str(parsed_args.dataset) + '_cand' + str(parsed_args.num_cand) + '_train' + str(parsed_args.num_train) + '_seed' + str(parsed_args.seed))


  rl_sum = 0
  rank_sum = 0
  rl_sum_rewite = 0
  rank_sum_rewrite = 0

  rl_syn = 0
  rank_syn = 0
  rl_syn_rewite = 0
  rank_syn_rewrite = 0

  rl_chunk = 0
  rank_chunk = 0
  rl_chunk_rewrite = 0
  rank_chunk_rewrite = 0

  rl_num = 0
  rank_num = 0
  for batch in tqdm.tqdm(testloader):
    tok = tokenizer(batch['cur'], padding=True, return_tensors='pt')
    encodings = encode_questions(model_rl, tok['input_ids'].to(device),
                                 tok['attention_mask'].to(device))
    #q_emb = encode(model_rl, test_questions[i]).reshape(1, -1)
    # print(len(index))
    sample_candidcates = candidcate_questions
    tok_res = tokenizer(sample_candidcates, padding=True, return_tensors='pt')
    candidcate_encodings = encode_questions(model_rl, tok_res['input_ids'].to(device),
                                            tok_res['attention_mask'].to(device))
    candidcate_norms = torch.norm(candidcate_encodings, dim=1)
    test_norms = torch.norm(encodings, dim=1)
    simi_map = torch.abs_(torch.mm(encodings, candidcate_encodings.T)) / torch.mm(test_norms.reshape(-1, 1),
                                                                              candidcate_norms.reshape(1, -1))
    top = torch.topk(simi_map, k=parsed_args.topk, dim=-1).indices
    # print("top3: ", top3.shape)
    for j in range(len(batch['cur'])):
      if batch['rewrite'][j].split() ==  batch['cur'][j].split():
        continue
      cur_top = [cand_index[i] for i in top[j]]
      for k in range(1):
        data = train_data[cur_top[k]]
        rl_sum += len(data['cur'].split())
        rl_sum_rewite += len(data['rewrite'].split())

        rl_syn += len(set(pos(data['cur']).split()))
        rl_syn_rewite += len(set(pos(data['rewrite']).split()))

        rl_chunk += len(nlp(data['cur'])._.phrases)
        rl_chunk_rewrite += len(nlp(data['rewrite'])._.phrases)

        rl_num += 1
        #print(batch['rewrite'][j], batch['cur'][j],pos(' '.join(find_lcsubstr(batch['rewrite'][j].split(), batch['cur'][j].split()))))
        # e.evaluate_metrics(cur_str=[''], restate_str=[
        #   pos(' '.join(find_lcsubstr(batch['rewrite'][j].split(), batch['cur'][j].split())))],
        #                    predict_str=[pos(' '.join(find_lcsubstr(data['rewrite'].split(), data['cur'].split())))])
    #print(e.get_metrics(reset=False))
  # eval_metrics = e.get_metrics(reset=True)
  # print("rl metrics = ", eval_metrics)

  for batch in tqdm.tqdm(testloader):
    tok = tokenizer(batch['cur'], padding=True, return_tensors='pt')
    encodings = encode_questions(model_rank, tok['input_ids'].to(device),
                                 tok['attention_mask'].to(device))
    #q_emb = encode(model_rl, test_questions[i]).reshape(1, -1)
    # print(len(index))
    sample_candidcates = candidcate_questions
    tok_res = tokenizer(sample_candidcates, padding=True, return_tensors='pt')
    candidcate_encodings = encode_questions(model_rank, tok_res['input_ids'].to(device),
                                            tok_res['attention_mask'].to(device))
    candidcate_norms = torch.norm(candidcate_encodings, dim=1)
    test_norms = torch.norm(encodings, dim=1)
    simi_map = torch.abs_(torch.mm(encodings, candidcate_encodings.T)) / torch.mm(test_norms.reshape(-1, 1),
                                                                              candidcate_norms.reshape(1, -1))
    top = torch.topk(simi_map, k=parsed_args.topk, dim=-1).indices
    # print("top3: ", top3.shape)
    for j in range(len(batch['cur'])):
      if batch['rewrite'][j].split() == batch['cur'][j].split():
        continue
      cur_top = [cand_index[i] for i in top[j]]
      for k in range(1):
        data = train_data[cur_top[k]]
        rank_sum += len(data['cur'].split())
        rank_sum_rewrite += len(data['rewrite'].split())

        rank_syn += len(set(pos(data['cur']).split()))
        rank_syn_rewrite += len(set(pos(data['rewrite']).split()))

        rank_chunk += len(nlp(data['cur'])._.phrases)
        rank_chunk_rewrite += len(nlp(data['rewrite'])._.phrases)

        rank_num += 1

        #print(test_data[i]['cur'], data['cur'], pos(test_data[i]['cur']), pos(data['cur']))
  #       e.evaluate_metrics(cur_str=[''], restate_str=[pos(' '.join(find_lcsubstr(batch['rewrite'][j].split(),batch['cur'][j].split())))],
  #                        predict_str=[pos(' '.join(find_lcsubstr(data['rewrite'].split(),data['cur'].split())))])
  #   #print(e.get_metrics(reset=False))
  # eval_metrics = e.get_metrics(reset=True)
  # print("rank metrics = ", eval_metrics)
print(rl_sum/ rl_num, rank_sum/ rank_num)
print(rl_syn/ rl_num, rank_syn/ rank_num)
print(rl_chunk/ rl_num, rank_chunk/ rank_num)

print(rl_sum_rewite/ rl_num, rank_sum_rewrite/ rank_num)
print(rl_syn_rewite/ rl_num, rank_syn_rewrite/ rank_num)
print(rl_chunk_rewrite/ rl_num, rank_chunk_rewrite/ rank_num)


# print((rl_sum_rewite - rl_sum)/ rl_num, (rank_sum_rewrite - rank_sum)/ rank_num)
# print((rl_syn_rewite - rl_syn)/ rl_num, (rank_syn_rewrite - rank_syn) / rank_num)
#
