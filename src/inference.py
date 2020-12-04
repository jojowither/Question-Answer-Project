from transformers import AutoTokenizer, AutoModelForQuestionAnswering, BertConfig
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from pathlib import Path
import json
import collections
from utils import time_count_wrapper, chunks_ques_wrapper
import yaml
from torchsummary import summary
from fire import Fire

with open('config/inference_config.yaml', 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)

NUMBER = 1
TEXT = config['TEXT'][NUMBER]
QUESTIONS = config['QUESTIONS'][NUMBER]

def initialize():
    # Load a trained model and vocabulary that you have fine-tuned
    model = AutoModelForQuestionAnswering.from_pretrained(config['MODEL_DIR'])  # , force_download=True)
    tokenizer = AutoTokenizer.from_pretrained(config['MODEL_DIR'], do_lower_case=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f'\nFinish loading the model...')
    print(f'The model version is {Path(config["MODEL_DIR"]).name}\n')
    return model, tokenizer, device


@chunks_ques_wrapper(all_ques=QUESTIONS, size=12)
def inference(model, tokenizer, device, text=TEXT,questions=QUESTIONS):
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
    torch.cuda.empty_cache()

    return input_ids, answer_starts, answer_ends, confidences


def postprocess(tokenizer, input_ids, answer_starts, answer_ends, confidences):
    answers = []
    for input_id, answer_start, answer_end in zip(input_ids, answer_starts, answer_ends):
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_id[answer_start:answer_end]))
        answer = answer.replace(' ', '')
        if '[CLS]' in answer or '[SEP]' in answer or answer=='':
            answer = '無答案'
        answers.append(answer)

    qa_pair = collections.defaultdict(dict)
    for question, answer, confidence in zip(QUESTIONS, answers, confidences):
        qa_pair[question].update({'Answer': answer})
        qa_pair[question].update({'Confidence': f'{confidence:.2f}'})
    qa_pair = dict(sorted(qa_pair.items(), key=lambda pair: QUESTIONS.index(pair[0])))

    for question, answer_set in qa_pair.items():
        print(f"Question: {question}")
        print(f"Answer: {answer_set['Answer']}")
        print(f"Confidence: {answer_set['Confidence']}")
        print()

    save_result(qa_pair)


def save_result(qa_pair):
    complete_pair = {}
    complete_pair['text'] = TEXT[0]
    complete_pair['qa_pair'] = qa_pair

    try:
        data = json.load(open(config['OUTPUT_JSON_PATH'], encoding='utf8'))
    except:
        data = collections.defaultdict(list)
    finally:
        data['data'].append(complete_pair)
        json.dump(data, open(config['OUTPUT_JSON_PATH'], 'w', encoding='utf8'), ensure_ascii=False, indent='\t')
        print(f'Finish saving the inference result by json format.')


@time_count_wrapper
def main():
    model, tokenizer, device = initialize()
    input_ids, answer_starts, answer_ends, confidences = inference(model, tokenizer, device)
    postprocess(tokenizer, input_ids, answer_starts, answer_ends, confidences)

    # if summary_model:
    #     summary(model, [(12, 285)]*3)
    #     # show how to modify the summary


if __name__ == "__main__":
    main()
