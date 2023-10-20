from typing import List, Dict
from data_utils import Scorer, BatchAverage, FScoreMetric, CorpusBLEUMetric
import torch
import re

class eval():
  def __init__(self):
    self.metrics = {'ROUGE': BatchAverage(),
                    '_ROUGE1': BatchAverage(),
                    '_ROUGE2': BatchAverage(),
                    # TODO: You can speed up the code by disable BLEU since
                    #  the corpus-based BLEU metric is much time-consuming.
                    'BLEU': CorpusBLEUMetric(),
                    'EM': BatchAverage(),
                    'F1': FScoreMetric(prefix="1"),
                    'F2': FScoreMetric(prefix="2"),
                    'F3': FScoreMetric(prefix="3")}

  def evaluate_metrics(self, restate_str: List[str], predict_str: List[str], cur_str: List[str]):
    """
    BLEU Score
    """
    self.metrics['BLEU'](restate_str, predict_str)
    """
    Exact Match Score
    """
    em_score = Scorer.em_score(restate_str, predict_str)
    self.metrics['EM'](em_score)

    """
    ROUGE Score
    """
    rouge1, rouge2, rouge = Scorer.rouge_score(restate_str, predict_str)
    self.metrics['ROUGE'](rouge)
    self.metrics['_ROUGE1'](rouge1)
    self.metrics['_ROUGE2'](rouge2)

    """
    F-Score (note this one is the rewriting F-score)
    See definition in paper: https://ai.tencent.com/ailab/nlp/dialogue/papers/EMNLP_zhufengpan.pdf
    """
    i1c, p1c, r1c, i2c, p2c, r2c, i3c, p3c, r3c = Scorer.restored_count(
      restate_str, predict_str, cur_str)
    self.metrics['F1'](i1c, p1c, r1c)
    self.metrics['F2'](i2c, p2c, r2c)
    self.metrics['F3'](i3c, p3c, r3c)

  def get_metrics(self, reset: bool = False) -> Dict[str, float]:
    other_metrics = {k: v.get_metric(reset) for k, v in self.metrics.items() if k not in ['F1', 'F2', 'F3', 'BLEU']}
    f_metrics_dict = {k: v.get_metric(reset) for k, v in self.metrics.items() if k in ['F1', 'F2', 'F3']}
    f_metrics_dict = {**f_metrics_dict['F1'], **f_metrics_dict['F2'], **f_metrics_dict['F3']}
    bleu_metrics = self.metrics['BLEU'].get_metric(reset)
    return {**other_metrics, **f_metrics_dict, **bleu_metrics}

e = eval()
# e.evaluate_metrics(cur_str=['how did they target her email ?'], restate_str=["how did fsb target the murdered journalist anna politkovskaya 's email ?"], predict_str=["In the unsolved murder case of Anna Politkovskaya in 2016, were there any clues indicating how the individuals, possibly associated with the FSB, targeted her webmail account?"])
# print(e.get_metrics(reset=True))
# e.evaluate_metrics(cur_str=['how did they target her email ?'], restate_str=["how did fsb target the murdered journalist anna politkovskaya 's email ?"], predict_str=["In the unresolved murder of Anna Politkovskaya in 2016, how did the individuals, likely associated with the FSB, target Anna Politkovskaya's email?"])
# print(e.get_metrics(reset=True))
# e.evaluate_metrics(cur_str=['how did they target her email ?'], restate_str=["how did fsb target the murdered journalist anna politkovskaya 's email ?"], predict_str=["How did the individuals, likely associated with the FSB, target Anna Politkovskaya's webmail account in the unsolved murder case from 2016?"])
# print(e.get_metrics(reset=True))
# e.evaluate_metrics(cur_str=['how did they target her email ?'], restate_str=["how did fsb target the murdered journalist anna politkovskaya 's email ?"], predict_str=["How were the webmail account of the murdered Russian journalist Anna Politkovskaya targeted?"])
# print(e.get_metrics(reset=True))
#
# e.evaluate_metrics(cur_str=['how did they target her email ?'], restate_str=["how did fsb target the murdered journalist anna politkovskaya 's email ?"], predict_str=["How were the webmail account of the murdered Russian journalist Anna Politkovskaya targeted?"])
# print(e.get_metrics(reset=True))
#
# e.evaluate_metrics(cur_str=['how did they target her email ?'], restate_str=["how did fsb target the murdered journalist anna politkovskaya 's email ?"], predict_str=["How did the FSB target the webmail account of the murdered Russian journalist Anna Politkovskaya ?".lower()])
# print(e.get_metrics(reset=True))

evaluations = torch.load('./chatgpt_random_five_shot_task_prediction_task')
for data in evaluations:
  #print('data = ', re.sub('\n', '', data['pred'].lower()))
  e.evaluate_metrics(cur_str=[data['cur']], restate_str=[data['restate']], predict_str=[ data['pred'].lower().split('\n')[0]])
print(e.get_metrics(reset=True))