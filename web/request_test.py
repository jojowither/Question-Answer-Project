import requests
import json
import yaml
from pprint import pprint
import time

with open('config.yaml', 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)

def time_count_wrapper(func):
    def time_count(*args, **kwargs):
        ts = time.time()
        response = func(*args, **kwargs)
        te = time.time()
        print(f'\n{"="*40}')
        print (f"Time consume: {te-ts:.3f} s")
        print(f'{"="*40}\n')
        return response
    return time_count


@time_count_wrapper
def resquest_post(url, data):
    response = requests.post(url=f"{url}/predictions/qa_server", data=data)
    return response


def main():
    context = input('Type the text: ')
    question_list = []
    while True:
        question = input('Type the question (type "exit" to close): ')
        if question=='exit':
            break
        question_list.append(question)

    t = time.localtime()
    timestamp = time.strftime('%Y%m%d%H%M%S', t)

    profile = {}
    profile['context'] = context
    if question_list==[]:
        profile["questions"] = []
    else:
        profile["questions"] = []
        for question in question_list:
            profile["questions"].append({"content":question, 
                                         "ID":f"{str(9999)}-{question}"})

    payload = json.dumps(profile)
    response = resquest_post(url=config['BASE_URL'], data=payload)
    data = response.json()

    print()
    print('The response:')
    pprint(data)
    print()
    # print(json.dumps(data, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()