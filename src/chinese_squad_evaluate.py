import collections
import json
import math
import re
import string
import pandas as pd
import yaml

from transformers.tokenization_bert import BasicTokenizer
from transformers.utils import logging


logger = logging.get_logger(__name__)


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)

    try:
        gold_toks = list(gold_toks[0])
    except:
        gold_toks = ['']
    try: 
        pred_toks = list(pred_toks[0])
    except:
        pred_toks = ['']

    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if num_same == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return 0, 0, 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def get_raw_scores(examples, preds, detail=False):
    """
    Computes the exact and f1 scores from the examples and the model predictions
    """
    exact_scores = {}
    f1_scores = {}
    precision_scores = {}
    recall_scores = {}
    questions_json = collections.defaultdict(dict)
    logger.info(f"  Start to compute metrics")

    for example in examples:
        qas_id = example.qas_id
        gold_answers = [answer["text"] for answer in example.answers if normalize_answer(answer["text"])]

        if not gold_answers:
            # For unanswerable questions, only correct answer is empty string
            gold_answers = [""]

        if qas_id not in preds:
            print("Missing prediction for %s" % qas_id)
            continue

        prediction = preds[qas_id]
        exact_scores[qas_id] = max(compute_exact(a, prediction) for a in gold_answers)
        f1_scores[qas_id] = max(compute_f1(a, prediction)[0] for a in gold_answers)
        precision_scores[qas_id] = max(compute_f1(a, prediction)[1] for a in gold_answers)
        recall_scores[qas_id] = max(compute_f1(a, prediction)[2] for a in gold_answers)

        if detail:
            questions_json[qas_id].update({'context':example.context_text})
            questions_json[qas_id].update({'question':example.question_text})
            questions_json[qas_id].update({'answer':gold_answers[0]})

    logger.info(f"  Finish computing metrics")
    if detail:
        return exact_scores, f1_scores, precision_scores, recall_scores, questions_json
    else:
        return exact_scores, f1_scores, precision_scores, recall_scores


def apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh):
    new_scores = {}
    for qid, s in scores.items():
        pred_na = na_probs[qid] > na_prob_thresh
        if pred_na:
            new_scores[qid] = float(not qid_to_has_ans[qid])
        else:
            new_scores[qid] = s
    return new_scores


def make_eval_dict(thresholds, qid_list=None):
    exact_scores, f1_scores, precision_scores, recall_scores = thresholds
    if not qid_list:
        total = len(exact_scores)
        return collections.OrderedDict(
            [
                ("exact", 100.0 * sum(exact_scores.values()) / total),
                ("f1", 100.0 * sum(f1_scores.values()) / total),
                ("precision", 100.0 * sum(precision_scores.values()) / total),
                ("recall", 100.0 * sum(recall_scores.values()) / total),
                ("total", total),
            ]
        )
    else:
        total = len(qid_list)
        return collections.OrderedDict(
            [
                ("exact", 100.0 * sum(exact_scores[k] for k in qid_list) / total),
                ("f1", 100.0 * sum(f1_scores[k] for k in qid_list) / total),
                ("precision", 100.0 * sum(precision_scores[k] for k in qid_list) / total),
                ("recall", 100.0 * sum(recall_scores[k] for k in qid_list) / total),
                ("total", total),
            ]
        )


def merge_eval(main_eval, new_eval, prefix):
    for k in new_eval:
        main_eval["%s_%s" % (prefix, k)] = new_eval[k]


def find_best_thresh_v2(preds, scores, na_probs, qid_to_has_ans):
    num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
    cur_score = num_no_ans
    best_score = cur_score
    best_thresh = 0.0
    qid_list = sorted(na_probs, key=lambda k: na_probs[k])
    for i, qid in enumerate(qid_list):
        if qid not in scores:
            continue
        if qid_to_has_ans[qid]:
            diff = scores[qid]
        else:
            if preds[qid]:
                diff = -1
            else:
                diff = 0
        cur_score += diff
        if cur_score > best_score:
            best_score = cur_score
            best_thresh = na_probs[qid]

    has_ans_score, has_ans_cnt = 0, 0
    for qid in qid_list:
        if not qid_to_has_ans[qid]:
            continue
        has_ans_cnt += 1

        if qid not in scores:
            continue
        has_ans_score += scores[qid]

    return 100.0 * best_score / len(scores), best_thresh, 1.0 * has_ans_score / has_ans_cnt


def find_all_best_thresh_v2(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans):
    best_exact, exact_thresh, has_ans_exact = find_best_thresh_v2(preds, exact_raw, na_probs, qid_to_has_ans)
    best_f1, f1_thresh, has_ans_f1 = find_best_thresh_v2(preds, f1_raw, na_probs, qid_to_has_ans)
    main_eval["best_exact"] = best_exact
    main_eval["best_exact_thresh"] = exact_thresh
    main_eval["best_f1"] = best_f1
    main_eval["best_f1_thresh"] = f1_thresh
    main_eval["has_ans_exact"] = has_ans_exact
    main_eval["has_ans_f1"] = has_ans_f1


def find_best_thresh(preds, scores, na_probs, qid_to_has_ans):
    num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
    cur_score = num_no_ans
    best_score = cur_score
    best_thresh = 0.0
    qid_list = sorted(na_probs, key=lambda k: na_probs[k])
    for _, qid in enumerate(qid_list):
        if qid not in scores:
            continue
        if qid_to_has_ans[qid]:
            diff = scores[qid]
        else:
            if preds[qid]:
                diff = -1
            else:
                diff = 0
        cur_score += diff
        if cur_score > best_score:
            best_score = cur_score
            best_thresh = na_probs[qid]
    return 100.0 * best_score / len(scores), best_thresh


def find_all_best_thresh(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans):
    best_exact, exact_thresh = find_best_thresh(preds, exact_raw, na_probs, qid_to_has_ans)
    best_f1, f1_thresh = find_best_thresh(preds, f1_raw, na_probs, qid_to_has_ans)

    main_eval["best_exact"] = best_exact
    main_eval["best_exact_thresh"] = exact_thresh
    main_eval["best_f1"] = best_f1
    main_eval["best_f1_thresh"] = f1_thresh

def apply_all_thresholds(no_answer_probs, qas_id_to_has_answer, no_answer_probability_threshold, *scores):
    thresholds = []
    for score in scores:
        threshold = apply_no_ans_threshold(
            score, no_answer_probs, qas_id_to_has_answer, no_answer_probability_threshold
        )
        thresholds.append(threshold)
    return thresholds

def squad_evaluate(examples, preds, no_answer_probs=None, no_answer_probability_threshold=1.0):
    qas_id_to_has_answer = {example.qas_id: bool(example.answers) for example in examples}
    has_answer_qids = [qas_id for qas_id, has_answer in qas_id_to_has_answer.items() if has_answer]
    no_answer_qids = [qas_id for qas_id, has_answer in qas_id_to_has_answer.items() if not has_answer]

    if no_answer_probs is None:
        no_answer_probs = {k: 0.0 for k in preds}

    exact, f1, precision, recall = get_raw_scores(examples, preds)

    #Order: [exact, f1, precision, recall]
    thresholds = apply_all_thresholds(no_answer_probs, qas_id_to_has_answer, 
                no_answer_probability_threshold, *(exact, f1, precision, recall))

    evaluation = make_eval_dict(thresholds)

    if has_answer_qids:
        has_ans_eval = make_eval_dict(thresholds, qid_list=has_answer_qids)
        merge_eval(evaluation, has_ans_eval, "HasAns")

    if no_answer_qids:
        no_ans_eval = make_eval_dict(thresholds, qid_list=no_answer_qids)
        merge_eval(evaluation, no_ans_eval, "NoAns")

    if no_answer_probs:
        find_all_best_thresh(evaluation, preds, exact, f1, no_answer_probs, qas_id_to_has_answer)

    return evaluation


def detail2excel(questions, preds, f1s, precisions, recalls, path='../output/detail_eval.xlsx'):
    context_col = []
    question_col = []
    ground_truth_col = []
    pred_col = []
    f1_col = []
    precision_col = []
    recall_col = []
    logger.info(f"  Prepare to construct excel")

    for qid, f1 in f1s.items():
        context_col.append(questions[qid]['context'])
        question_col.append(questions[qid]['question'])
        ground_truth_col.append(questions[qid]['answer'])
        pred_col.append(preds[qid])
        f1_col.append(f1)
        precision_col.append(precisions[qid])
        recall_col.append(recalls[qid])

    details = {
        'Context': context_col,
        'Question': question_col,
        'Ground Truth': ground_truth_col,
        'Prediction': pred_col,
        'F1': f1_col,
        'Precision': precision_col,
        'Recall': recall_col
    }
    df = pd.DataFrame(details)
    analysis_df = analyze_df(df)

    with pd.ExcelWriter(path) as writer:
        df.to_excel(writer, sheet_name='All Questions')
        analysis_df.to_excel(writer, sheet_name='Analysis Questions')
    logger.info(f"  Finish saving dataframe to {path}")


def analyze_df(df, config_path='config/inference_config.yaml'):
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)

    analysis_df = pd.DataFrame(columns=(('Question', 'numbers', 'F1', 'Precision', 'Recall')))
    
    question_list = config['QUESTIONS']
    for question in question_list:
        ques_df = df.loc[df['Question']==question]
        ques_dict = {'Question': question,
                     'numbers': len(ques_df),
                     'F1': round(ques_df['F1'].mean(),2),
                     'Precision': round(ques_df['Precision'].mean(),2),
                     'Recall': round(ques_df['Recall'].mean(),2)
                    }
        analysis_df = analysis_df.append(ques_dict, True)
    return analysis_df


def detail_evaluate(examples, preds, no_answer_probs=None, no_answer_probability_threshold=1.0):
    qas_id_to_has_answer = {example.qas_id: bool(example.answers) for example in examples}
    has_answer_qids = [qas_id for qas_id, has_answer in qas_id_to_has_answer.items() if has_answer]
    no_answer_qids = [qas_id for qas_id, has_answer in qas_id_to_has_answer.items() if not has_answer]

    if no_answer_probs is None:
        no_answer_probs = {k: 0.0 for k in preds}

    exact, f1, precision, recall, questions = get_raw_scores(examples, preds, detail=True)
    detail2excel(questions, preds, f1, precision, recall)

    return None
    