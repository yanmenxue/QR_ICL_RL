from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
import torch
import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AutoModelForSeq2SeqLM.from_pretrained("../pretrain_model/fast_chat").to(device)
tokenizer = AutoTokenizer.from_pretrained("../pretrain_model/fast_chat")
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
template = 'Rewrite an incomplete utterance into an utterance which is semantically equivalent but self-contained to be understood without context. The sentence structure and expression should be consistent. There are 3 examples:\n'
model = AutoModelForSeq2SeqLM.from_pretrained("../pretrain_model/fast_chat").to(device)
tokenizer = AutoTokenizer.from_pretrained("../pretrain_model/fast_chat")
model = model.eval()

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
  prompt = template + 'Input: Context: ' + data1['context'] + \
           ' Current utterance: ' + data1['cur']  + \
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

  inputs = tokenizer(prompt, return_tensors="pt", padding=True)
  outputs = model.generate(input_ids=inputs['input_ids'].to(device), max_length=100)
  pred = ' '.join(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].split()[1:])
  print('i = ', i, 'pred = ', pred, 'gold = ', test_data[i]['rewrite'])
  prediction.append({'context': test_data[i]['context'], 'cur': test_data[i]['cur'], 'restate':test_data[i]['rewrite'],
                       'pred': pred})

  torch.save(prediction, './canard_data/fast-chat_random_five_shot_prediction')


