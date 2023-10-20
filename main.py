from sentence_transformers import SentenceTransformer, util
import torch
from transformers import BertTokenizer, BertModel, T5Tokenizer, T5ForConditionalGeneration
import torch.nn.functional as F
# model = SentenceTransformer('../pretrain_model/sentence-bert')
# sentence = 'Peking is a beautiful city'
# sentence1 = 'It was a great day'
# sentence2 = 'Today was awesome'
# sentence1_representation = model.encode(sentence1)
# sentence2_representation = model.encode(sentence2)
# cosin_sim = util.pytorch_cos_sim(sentence1_representation,sentence2_representation)
# print(cosin_sim)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = BertModel.from_pretrained('../pretrain_model/sentence-bert').to(device)
tokenizer = BertTokenizer.from_pretrained('../pretrain_model/sentence-bert')
def encode(model, sentence):
  encoding = model(tokenizer(sentence, return_tensors='pt')['input_ids'].to(device)).last_hidden_state
  #print(tokenizer.convert_ids_to_tokens(tokenizer(sentence, return_tensors='pt')['input_ids'][0]))
  return torch.mean(encoding[0], dim=0)
def cos_similarity(model, sentence1, sentence2):
  s1 = encode(model,sentence1)
  s2 = encode(model,sentence2)
  return torch.mm(s1.reshape(1,-1), s2.reshape(-1,1)) / (torch.norm(s1) * torch.norm(s2))

# print(cos_similarity(model,'It was a great day', 'Today was awesome'))
#
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
#
# model = AutoModelForSeq2SeqLM.from_pretrained("../pretrain_model/fast_chat")
# tokenizer = AutoTokenizer.from_pretrained("../pretrain_model/fast_chat")
#
# inputs = tokenizer('''Rewrite an incomplete utterance into an utterance which is semantically equivalent but self-contained to be understood without context. The sentence structure and expression should be consistent.
#                            There is an example:
#                            Input: Context: anna politkovskaya, the murder remains unsolved , 2016. Current utterance: did they have any clues ?
#                            Ourput: did investigators have any clues in the unresolved murder of anna politkovskaya ?
#                            Input: Context: anna politkovskaya; the murder remains unsolved , 2016; did they have any clues ?; probably fsb ) are known to have targeted the webmail account of the murdered russian journalist anna politkovskaya . Current utterance:  how did they target her email ?
#                            OUtput: ?''', return_tensors="pt", padding = 'max_length', max_length = 2500)
# outputs = model.generate(**inputs, max_length=100)
# print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

# from rank_bm25 import BM25Okapi
#
# corpus = [
#     "how about serving mediterranean food ?",
#     "can you recommend a restaurant to me ? i do n't want to spend a lot of money .",
#     "can you help me find a russian restaurant ?",
#     "i do n't care .",
#   "yes , i would like world food ."
# ]
# s1 = "how about serving mediterranean food ?"
# s2 ="can you recommend a restaurant to me ? i do n't want to spend a lot of money ."
# tokenized_corpus = [doc.split(" ") for doc in corpus]
#
# bm25 = BM25Okapi(tokenized_corpus)
#
# query = "how about mediterranean food ?"
# tokenized_query = query.split(" ")
#
# doc_scores = bm25.get_scores(tokenized_query)
# print(F.softmax(torch.tensor(doc_scores)), type(doc_scores))
# print(cos_similarity(model,query,s1), cos_similarity(model,query,s2))
# array([0.        , 0.93729472, 0.        ])

# import spacy
# import nltk
# en_nlp = spacy.load('en_core_web_sm')
# zh_nlp = spacy.load('zh_core_web_sm')
# def stem(s, language = 'en'):
#   if language == 'en':
#     doc_spacy = en_nlp(s)
#     return ' '.join([token.lemma_ for token in doc_spacy])
#   else:
#     return s
#
# print(stem("今天天骑很好！",language='zh'))


import spacy
# 必须导入pytextrank，虽然表面上没用上，
import pytextrank

# example text
text = "Sirhan Sirhan , in his first television interview , called Sen. Robert F. Kennedy his hero , but said he killed the presidential candidate more than 20 years ago because he felt betrayed by Kennedy 's support for Israel ."


# 加载模型和依赖
nlp = spacy.load("en_core_web_sm")

# 此处调用“PyTextRank”包
nlp.add_pipe("textrank")
doc = nlp(text)
print(len(doc._.phrases))
# 读出短语、词频和权重
for phrase in doc._.phrases:
    # 短语
    print(phrase.text)
    # 权重、词频
    print(phrase.rank, phrase.count)
    '''# 短语的列表
    print(phrase.chunks)'''
