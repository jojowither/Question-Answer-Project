import random
import numpy as np
import torch
import time

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def time_count_wrapper(func):
    def time_count():
        ts = time.time()
        func()
        te = time.time()
        print(f"Time consume: {te-ts:.3f} s")

    return time_count


def chunks(all_ques, size=12):
    for i in range(0, len(all_ques), size):
        yield all_ques[i:i+size]


def chunks_ques_wrapper(all_ques, size=12):
    def _chunks_ques_wrapper(func):
        def chunks_ques(*args, **kwargs):
            all_input_ids, all_answer_starts, all_answer_ends, all_scores = [], [], [], []
            ques_set = list(chunks(all_ques=all_ques, size=size))

            for questions in ques_set:
                input_ids, answer_starts, answer_ends, scores = func(*args, **kwargs, questions=questions)
                all_input_ids.extend(input_ids)
                all_answer_starts.extend(answer_starts)
                all_answer_ends.extend(answer_ends)
                all_scores.extend(scores)

            all_answer_starts = torch.as_tensor(all_answer_starts)
            all_answer_ends = torch.as_tensor(all_answer_ends)

            return all_input_ids, all_answer_starts, all_answer_ends, all_scores
        return chunks_ques
    return _chunks_ques_wrapper

