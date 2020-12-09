from pathlib import Path
from pprint import pprint

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from imgcat import imgcat
from tensorboard.backend.event_processing.event_accumulator import \
    EventAccumulator

matplotlib.use("module://imgcat")


def get_newest_log(path='runs/', verbose=True):
    tensorboard_dir = Path(path)
    file_list = [x for x in tensorboard_dir.iterdir() if x.is_dir()]
    newest_dir = file_list[-1]
    newest_log = str(list(newest_dir.glob('event*'))[0])
    if verbose:
        print('The log path:')
        print(f'{newest_log}\n')
    return newest_log


def get_event_acc(newest_log, verbose=True):
    event_acc = EventAccumulator(newest_log)
    event_acc.Reload()
    if verbose:
        # Show all tags in the log file
        pprint(event_acc.Tags())
    return event_acc


def draw(event_acc, col_list=['eval_f1', 'eval_HasAns_f1', 'eval_NoAns_f1'], 
        color_list=['#FF5151', '#6A6AFF', 'orange'], save_path=''):
    #================F1================
    fig, ax = plt.subplots()
    for col, color in zip(col_list, color_list):
        steps = []
        values = []
        if col not in event_acc.Tags()['scalars']:
            continue
        for s in event_acc.Scalars(col):
            steps.append(s.step)
            values.append(s.value)

        if col == 'eval_f1':
            ax.plot(steps, values, color=color, label=col, linewidth=3)
        else:
            ax.plot(steps, values, color=color, label=col)
    
    yticks = np.arange(50, 110, 10)
    ax.set_yticks(yticks)
    ax.set_ylabel('F1 Score')
    ax.set_xlabel('Steps')
    ax.set_title('Eval on Dev Set')
    ax.grid()
    ax.legend(loc=4)
    fig.show() 
    fig.savefig(f'{save_path/"eval_f1.png"}')
    print()
    #================loss================
    fig, ax = plt.subplots()
    steps = []
    values = []
    for s in event_acc.Scalars('loss'):
        steps.append(s.step)
        values.append(s.value)
    ax.plot(steps, values, color='orange', label='training loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Steps')
    ax.grid()
    ax.legend()
    fig.show() 
    fig.savefig(f'{save_path/"loss.png"}')


def main():
    newest_log = get_newest_log()
    event_acc = get_event_acc(newest_log)
    draw(event_acc, save_path=Path(newest_log).parent)


if __name__=='__main__':
    main()
