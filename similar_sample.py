from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
import torch
import random
from typing import List, Dict
#from data_utils import Scorer, BatchAverage, FScoreMetric, CorpusBLEUMetric
device = 'cuda' if torch.cuda.is_available() else 'cpu'
random.seed(2023)
# class eval():
#   def __init__(self):
#     self.metrics = {'ROUGE': BatchAverage(),
#                     '_ROUGE1': BatchAverage(),
#                     '_ROUGE2': BatchAverage(),
#                     # TODO: You can speed up the code by disable BLEU since
#                     #  the corpus-based BLEU metric is much time-consuming.
#                     'BLEU': CorpusBLEUMetric(),
#                     'EM': BatchAverage(),
#                     'F1': FScoreMetric(prefix="1"),
#                     'F2': FScoreMetric(prefix="2"),
#                     'F3': FScoreMetric(prefix="3")}
#
#   def evaluate_metrics(self, restate_str: List[str], predict_str: List[str], cur_str: List[str]):
#     """
#     BLEU Score
#     """
#     self.metrics['BLEU'](restate_str, predict_str)
#     """
#     Exact Match Score
#     """
#     em_score = Scorer.em_score(restate_str, predict_str)
#     self.metrics['EM'](em_score)
#
#     """
#     ROUGE Score
#     """
#     rouge1, rouge2, rouge = Scorer.rouge_score(restate_str, predict_str)
#     self.metrics['ROUGE'](rouge)
#     self.metrics['_ROUGE1'](rouge1)
#     self.metrics['_ROUGE2'](rouge2)
#
#     """
#     F-Score (note this one is the rewriting F-score)
#     See definition in paper: https://ai.tencent.com/ailab/nlp/dialogue/papers/EMNLP_zhufengpan.pdf
#     """
#     i1c, p1c, r1c, i2c, p2c, r2c, i3c, p3c, r3c = Scorer.restored_count(
#       restate_str, predict_str, cur_str)
#     self.metrics['F1'](i1c, p1c, r1c)
#     self.metrics['F2'](i2c, p2c, r2c)
#     self.metrics['F3'](i3c, p3c, r3c)
#
#   def get_metrics(self, reset: bool = False) -> Dict[str, float]:
#     other_metrics = {k: v.get_metric(reset) for k, v in self.metrics.items() if k not in ['F1', 'F2', 'F3', 'BLEU']}
#     f_metrics_dict = {k: v.get_metric(reset) for k, v in self.metrics.items() if k in ['F1', 'F2', 'F3']}
#     f_metrics_dict = {**f_metrics_dict['F1'], **f_metrics_dict['F2'], **f_metrics_dict['F3']}
#     bleu_metrics = self.metrics['BLEU'].get_metric(reset)
#     return {**other_metrics, **f_metrics_dict, **bleu_metrics}

model = BertModel.from_pretrained('../pretrain_model/sentence-bert').to(device)
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

print(cos_similarity(model,'It was a great day', 'Today was awesome'))
train_data = torch.load('canard_train')
dev_data = torch.load('canard_dev')
test_data = torch.load('canard_test')
candidcate_questions = [train_data[i]['cur'] for i in range(len(train_data))]
#candidcate_encoddings = []
# i = 0
# while i < 100:#len(candidcate_questions):
#   sentences = candidcate_questions[i: i+10]
#   tok_res = tokenizer(sentences, padding=True, return_tensors='pt')
#   candidcate_encoddings.append(encode_questions(model, tok_res['input_ids'].to(device), tok_res['attention_mask'].to(device)))
#   i += 10
# candidcate_encoddings = torch.cat(candidcate_encoddings, dim=0)
#
test_questions = [test_data[i]['cur'] for i in range(len(test_data))]
# test_encodings = []
# i = 0
# while i < len(test_questions):
#   sentences = test_questions[i: i+10]
#   tok_res = tokenizer(sentences, padding=True, return_tensors='pt')
#   test_encodings.append(encode_questions(model, tok_res['input_ids'].to(device), tok_res['attention_mask'].to(device)))
#   i += 10
# test_encodings = torch.cat(test_encodings, dim=0)
#
# candidcate_norms = torch.norm(candidcate_encoddings, dim=1)
# test_norms = torch.norm(test_encodings, dim=1)
# simi_map = torch.mm(test_encodings, candidcate_encoddings.T) / torch.mm(test_norms.reshape(-1,1), candidcate_norms.reshape(1,-1))
# top3 = torch.topk(simi_map, k = 3, dim=-1).indices

glm_tokenizer = AutoTokenizer.from_pretrained("../pretrain_model/chatglm-6b", trust_remote_code=True)
glm_model = AutoModel.from_pretrained("../pretrain_model/chatglm-6b", trust_remote_code=True).half().cuda()
glm_model = glm_model.eval()
prediction = []
template = 'Rewrite an incomplete utterance into an utterance which is semantically equivalent but self-contained to be understood without context. The sentence structure and expression should be consistent. There are 5 examples:\n'
c = range(len(candidcate_questions))
N = 1500
for i in range(len(test_questions)):

  index = random.sample(c, N)
  q_emb = encode(model, test_questions[i]).reshape(1,-1)
  print(len(index))
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
  print('i = ', i, 'pred = ', response, 'gold = ', test_data[i]['rewrite'])
  prediction.append({'context': test_data[i]['context'], 'cur': test_data[i]['cur'], 'restate': test_data[i]['rewrite'],
                     'pred': response})
torch.save(prediction, './canard_data/chatglm_similar_three_shot_prediction')
# e = eval()
# for data in prediction:
#   e.evaluate_metrics(cur_str=[data['cur']], restate_str=[data['restate']], predict_str=[ data['pred'].lower().split('\n')[0]])
# print(e.get_metrics(reset=True))
