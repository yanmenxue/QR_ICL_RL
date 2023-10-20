import torch
import random
import openai
import os
os.environ["http_proxy"]="127.0.0.1:7890"
os.environ["https_proxy"]="127.0.0.1:7890"
random.seed(2023)
openai.api_key = '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'


# def num_tokens_from_string(string: str, encoding_name: str) -> int:
#   """Returns the number of tokens in a text string."""
#   import tiktoken
#   encoding = tiktoken.get_encoding(encoding_name)
#   num_tokens = len(encoding.encode(string))
#   return num_tokens

train_data = torch.load('canard_train')
dev_data = torch.load('canard_dev')
test_data = torch.load('canard_test')
print(len(train_data), len(dev_data), len(test_data))
prediction = []
template = 'Rewrite an incomplete utterance into an utterance which is semantically equivalent but self-contained to be understood without context. The sentence structure and expression should be consistent. There is an example:\n'
num = 0
sep = 1
while num < 100:
  for i in range(sep):
    prompt_list = []
    ind = random.randint(0, len(train_data)-1)
    data = train_data[ind]
    prompt = template + 'Input: Context: ' + data['context'] + \
             ' Current utterance: ' + data['cur']  + \
              '\nOutput: ' + data['rewrite'] + \
              '\nInput: Context: ' + test_data[num+i]['context'] + \
             ' Current utterance: ' + test_data[num+i]['cur'] + \
             '\nOutput: ?'
    prompt_list.append(prompt)
    print('prompt = ', prompt, 'gold = ', test_data[num+i]['rewrite'])
  #num += num_tokens_from_string(prompt, "cl100k_base")
  # response = openai.Completion.create(
  #   model="text-davinci-003",
  #   prompt=prompt_list,
  #   temperature=0.0,
  #   max_tokens=100,
  #   top_p=1.0,
  #   frequency_penalty=0.0,
  #   presence_penalty=0.0
  # )
  # #print("i = ", i, 'prompt = ', prompt, 'golden = ', test_data[i]['rewrite'], 'pred = ', response['choices'][0]['text'])
  # for choice in response.choices:
  #   prediction.append({'context': test_data[num + choice.index]['context'], 'cur': test_data[num + choice.index]['cur'], 'restate':test_data[num + choice.index]['rewrite'],
  #                      'pred': choice['text']})
  num += sep
  print("num = ", num)
  #torch.save(prediction, './canard_data/random_one_shot_prediction')

#print("num = ", num)

