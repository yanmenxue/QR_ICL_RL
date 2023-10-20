import json

import openai
import os
os.environ["http_proxy"]="127.0.0.1:7890"
os.environ["https_proxy"]="127.0.0.1:7890"
openai.api_key = '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'
response = openai.ChatCompletion.create(
          model="gpt-3.5-turbo",
            messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content":
                          '''translate the next sentence into Chinese: The overall procedure typically consists of the
modules of retrieval source construction, question representation,
graph based reasoning and answer ranking. These modules
will encounter different challenges for complex KBQA.
Firstly, the retrieval source construction module extracts a
question-specific graph from KBs, which covers a wide range
of relevant facts for each question. Due to unneglectable incompleteness
of source KBs [Min et al., 2013], the correct
reasoning paths may be absent from the extracted graph. This
issue is more likely to occur in the case of complex questions.
Secondly, question representation module understands
the question and generates instructions to guide the reasoning
process. This step becomes challenging when the question
is complicated. After that, reasoning on graph is conducted
through semantic matching. When dealing with complex
questions, such methods rank answers through semantic similarity
without traceable reasoning in the graph, which hinders
reasoning analysis and failure diagnosis. Eventually, this system
encounters the same training challenge under weak supervision
signals (i.e., question-answer pairs). The following
parts illustrate how prior work deal with these challenges.'''}
                                                    ],
  temperature=0.0
            )
print(response)
print(response['choices'][0]['message']['content'].replace('\\', '\\\\'))
# print( eval(repr(response['choices'][0]['message']['content']).replace('\\', '\\\\')) )
#

# response = openai.Completion.create(
#   model="text-curie-001",
#   prompt='''Rewrite an incomplete utterance into an utterance which is semantically equivalent but self-contained to be understood without context. The sentence structure and expression should be consistent. There are 5 example:
# Input: Context: hello Current utterance: i need a restaurant that serves european food please
# Output: i need a restaurant that serves european food please
# Input: Context: hello Current utterance: hello , i am looking for a restaurant on the south side of town that serves unusual food .
# Output: hello , i am looking for a restaurant on the south side of town that serves unusual food .
# Input: Context: hello Current utterance: i am looking for a japanese restaurant in the centre of town .
# Output: i am looking for a japanese restaurant in the centre of town .
# Input: Context: hello Current utterance: i am looking for an expensive french restaurant .
# Output: i am looking for an expensive french restaurant .
# Input: Context: hello Current utterance: i would like to find a restaurant in the east part of town that serves gastropub food .
# Output: i would like to find a restaurant in the east part of town that serves gastropub food .
# Input: Context: how about italian restaurants in the west part of town ?; there are several italian restaurants in the west part of town . la margherita is cheap and prezzo is moderate . would you like the address of one of those ? Current utterance: what is the phone number ?
# Output: ?''',
#   temperature=0.0,
#   max_tokens=1000,
#   top_p=1.0,
#   frequency_penalty=0.0,
#   presence_penalty=0.0
# )

# response = openai.Completion.create(
#   model="text-davinci-003",
#   prompt="translate the sentence into Chinese: Rewrite an incomplete utterance into an utterance which is semantically equivalent but self-contained to be understood without context. The sentence structure and expression should be consistent. There are 5 example:",
#   temperature=0,
#   max_tokens=60,
#   top_p=1.0,
#   frequency_penalty=0.0,
#   presence_penalty=0.0
# )
print("response = ", response)