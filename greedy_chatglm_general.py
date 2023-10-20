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

  arg_parser.add_argument("--max_ans_length", type=int, default=50,
                          help="Please specify a model file to evaluate")
  arg_parser.add_argument("--topk", type=int, default=5,
                          help="Please specify a model file to evaluate")
  arg_parser.add_argument("--greedy", type=int, default=10,
                          help="Please specify a model file to evaluate")
  arg_parser.add_argument("--print_seq", type=int, default=20,
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


  N_cand = parsed_args.num_cand
  N_train = parsed_args.num_train

  train_sample = random.sample(range(len(train_data)), N_cand + N_train)
  cand_index = train_sample[:N_cand]
  train_index = train_sample[N_cand:]

  candidcate_questions = [train_data[i]['cur'] for i in cand_index]
  test_questions = [test_data[i]['cur'] for i in range(len(test_data))]


  from evaluate import eval
  e = eval()
  e_greedy = eval()
  glm_tokenizer = AutoTokenizer.from_pretrained("../pretrain_model/chatglm-6b", trust_remote_code=True)
  glm_model = AutoModel.from_pretrained("../pretrain_model/chatglm-6b", trust_remote_code=True).half().cuda()
  glm_model = glm_model.eval()
  #glm_model = torch.nn.DataParallel(glm_model)
  prediction = []
  template_en = 'Rewrite an incomplete utterance into an utterance which is semantically equivalent but self-contained to be understood without context. The sentence structure and expression should be consistent. There are ' + str(parsed_args.topk) + ' examples:'
  template_zh = '将不完整的话语改写为能在没有上下文的情况下被理解但语义相等的话语。句子结构和表达应保持一致。以下是' + str(parsed_args.topk) + '个例子：'

  def truc(s, max_ans_length = parsed_args.max_ans_length):
    return ' '.join(s.split()[:max_ans_length])

  prediction = []
  for i in tqdm.tqdm(range(len(test_questions))):
    max_rouge = -1.
    max_pred = None
    for j in range(parsed_args.greedy):
      top = random.sample(range(N_cand), parsed_args.topk)
      cur_top = [cand_index[i] for i in top]
      prompt_list = []
      if parsed_args.dataset == 'canard' or parsed_args.dataset == 'task':
        prompt = template_en
        for k in range(parsed_args.topk):
          data = train_data[cur_top[k]]
          prompt += '\nInput: Context: ' + data['context'] + \
                    ' Current utterance: ' + data['cur'] + \
                    '\nOutput: ' + data['rewrite']
        prompt += '\nInput: Context: ' + test_data[i]['context'] + \
                 ' Current utterance: ' + test_data[i]['cur'] + \
                 '\nOutput: ?'
      elif parsed_args.dataset == 'rewrite':
        prompt = template_zh
        for k in range(parsed_args.topk):
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
      e_greedy.evaluate_metrics(cur_str=[test_data[i]['cur']], restate_str=[test_data[i]['rewrite']],
                         predict_str=[pred])
      greedy_metrics = e_greedy.get_metrics(reset=True)
      if greedy_metrics["ROUGE"] > max_rouge:
        max_rouge = greedy_metrics["ROUGE"]
        max_pred = pred
    prediction.append({'context': test_data[i]['context'], 'cur': test_data[i]['cur'], 'restate': test_data[i]['rewrite'],
                       'pred': max_pred})
    e.evaluate_metrics(cur_str=[test_data[i]['cur']], restate_str=[test_data[i]['rewrite']],
                              predict_str=[max_pred])
    if i % parsed_args.print_seq == 0:
      print(e.get_metrics(reset=False))
  eval_metrics = e.get_metrics(reset=True)
  print("eval metrics = ", eval_metrics)


# e = eval()
# for data in prediction:
#   e.evaluate_metrics(cur_str=[data['cur']], restate_str=[data['restate']], predict_str=[ data['pred'].lower().split('\n')[0]])
# print(e.get_metrics(reset=True))
