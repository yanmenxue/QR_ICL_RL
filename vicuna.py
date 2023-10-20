import json
import requests
import time

def json_send(data, url):
    headers = {"Content-type": "application/json",
               "Accept": "text/plain", "charset": "UTF-8"}
    response = requests.post(url=url, headers=headers, data=json.dumps(data))
    return json.loads(response.text)


if __name__ == "__main__":
    url = 'http://172.31.222.180:8505/vicuna_13b'
    # url = 'http://172.31.222.180:8506/vicuna_13b'

    print("Start inference")

    while True:
        # input_text could be str or list of str
        input_text = '''Rewrite an incomplete utterance into an utterance which is semantically equivalent but self-contained to be understood without context. The sentence structure and expression should be consistent.
                           There is an example:
                           Input: Context: anna politkovskaya, the murder remains unsolved , 2016. Current utterance: did they have any clues ?
                           Ourput: did investigators have any clues in the unresolved murder of anna politkovskaya ?
                           Input: Context: anna politkovskaya; the murder remains unsolved , 2016; did they have any clues ?; probably fsb ) are known to have targeted the webmail account of the murdered russian journalist anna politkovskaya . Current utterance:  how did they target her email ?
                           OUtput: ?'''
        data = {"input": input_text} # If any of the parameters are specified, other parameters should be specified as well.
        data = {"input": input_text, "max_new_tokens": 100, "num_return_sequences": 1, "num_beams": 1, "do_sample": False}
        result = json_send(data, url)
        # print(eval(result["output"]))
        print(type(result["output"]))
        print("Output", result["output"])
        break