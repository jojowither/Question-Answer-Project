import json
import logging
import os
from pathlib import Path
import collections
import ast
import yaml
from fire import Fire
import time

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from health_qa.src.utils import chunks_ques_wrapper

# change the directory to the project root
logger = logging.getLogger(__name__)
project_path = str(Path(__file__).parent.parent.parent)
os.chdir(project_path)

config = 'server/config/handler_config.yaml'
with open(config, 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)

initialized = False
initialized_result = None


def initialize(context):
    global initialized
    global initialized_result

    if initialized:
        return
    initialized = True

    # get cuda deivce
    properties = context.system_properties
    device = torch.device("cuda:" + str(properties.get("gpu_id"))
                          if torch.cuda.is_available() else "cpu")

    # Load a trained model and vocabulary that you have fine-tuned
    model = AutoModelForQuestionAnswering.from_pretrained(config['MODEL_DIR'])  # , force_download=True)
    tokenizer = AutoTokenizer.from_pretrained(config['MODEL_DIR'], do_lower_case=True)
    model.to(device)
    model.eval()

    initialized_result = model, tokenizer, device


def process_request(requests, context):
    _contexts = []
    _questions = []
    for request in requests:
        _context = []
        text_json = request.get("data")
        if text_json is None:
            # convert bytearray to utf-8
            text_json = request.get("body").decode('utf-8')
            
            basic_info = ast.literal_eval(text_json)
            text = basic_info.get('context')
            questions = basic_info.get('questions')

        if not questions:
            questions_list = []
        else:
            questions_list = {q['content']:q['ID'] for q in questions}
        _context.append(text)
        _contexts.append(_context)
        _questions.append(questions_list)

    return _contexts, _questions


# @chunks_ques_wrapper(all_ques, size=12)
def inference(model, tokenizer, device, text, questions=None):
    inputs = tokenizer(questions, text*len(questions), add_special_tokens=True, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].tolist()
    inputs.to(device)
    with torch.no_grad():
        answer_scores = model(**inputs)

    # To get the score by softmax
    answer_start_scores, answer_end_scores = F.softmax(torch.stack(answer_scores), dim=-1)
    # Get the most likely beginning of answer with the argmax of the score
    answer_s_max_scores, answer_starts = torch.max(answer_start_scores, -1)  
    # Get the most likely end of answer with the argmax of the score
    answer_e_max_scores, answer_ends = torch.max(answer_end_scores, -1)
    answer_ends = answer_ends + 1
    confidences = (answer_s_max_scores + answer_e_max_scores) / 2 
    
    # release the memory to avoid the OOM
    torch.cuda.empty_cache()

    return input_ids, answer_starts, answer_ends, confidences


def postprocessing(tokenizer, input_ids, answer_starts, answer_ends, confidences, all_ques):
    answers = []
    for input_id, answer_start, answer_end in zip(input_ids, answer_starts, answer_ends):
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_id[answer_start:answer_end]))
        answer = answer.replace(' ', '')
        
        # the question does not have the answer
        if '[CLS]' in answer or '[SEP]' in answer or answer=='':
            answer = '無答案'
        answers.append(answer)
  
    response = collections.defaultdict(dict)
    for question, answer, confidence in zip(all_ques, answers, confidences):
        qid = all_ques[question]
        response[qid].update({'Question': question})
        response[qid].update({'Answer': answer})
        response[qid].update({'Confidence': f'{confidence:.2f}'})
    response = dict(sorted(response.items(), key=lambda pair: pair[0]))

    basic_info = {}
    t = time.localtime()
    timestamp = time.strftime('%Y%m%d%H%M%S', t)
    basic_info['timestamp'] = str(timestamp)
    basic_info['Questions'] = response

    return basic_info


def handler(data, context):
    if data is None:
        initialize(context)
        return None

    assert initialized_result is not None
    model, tokenizer, device = initialized_result

    responses = []
    contexts, _questions = process_request(data, context)
    for context, questions_dict in zip(contexts, _questions):
        questions = list(questions_dict.keys())
        input_ids, answer_starts, answer_ends, confidences = chunks_ques_wrapper(all_ques=questions, size=13)(inference)(model, tokenizer, device, text=context)
        response = postprocessing(tokenizer, input_ids, answer_starts, answer_ends, confidences, questions_dict)
        responses.append(response)
    return responses


def main_test_handler(loop_count=1, batch_size=3):
    from easydict import EasyDict 
    from tqdm import tqdm
    import json

    test_json_path = 'server/test_json/sample_text1.txt'
    json_context = json.load(open(test_json_path))

    data = [{
            'body': json.dumps(json_context).encode('utf8')
            }] * batch_size

    context = {
        'system_properties': {
            'gpu_id': '0'
        }
    }

    context = EasyDict(context)
    responses = handler(None, context)
    responses = handler(data, context)

    timer = tqdm(total=batch_size*loop_count,
                 smoothing=0.0, dynamic_ncols=True)
    for _ in range(loop_count):
        handler(data, context)
        timer.update(batch_size)

if __name__ == '__main__':
    Fire(main_test_handler)