import json
import requests
import streamlit as st

from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import yaml
import time

from health_qa.src.plot_tensorboard import get_newest_log, get_event_acc

with open('config.yaml', 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)


def type_context():
    context = st.text_area("Context", height=50)
    return context


###### 改標題，換branch
def type_questions():
    question_num = st.text_input("How many questions you want to ask?")
    if question_num=='':
        st.stop()

    question_list = []
    for i in range(int(question_num)):
        custom_question = st.text_input("Question(type Enter to next question)", key=str(i))
        if custom_question=='':
            st.stop()
        question_list.append(custom_question)
    return question_list


def summarize():
    context = type_context()
    question_list = type_questions()

    if st.button("Summarize"):
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


        st.subheader('Show the payload of the request')
        st.write(profile)
        payload = json.dumps(profile)

        st.subheader("Posting...")
        ts = time.time()
        response = requests.post(f"{config['BASE_URL']}/predictions/qa_server", data=payload)
        te = time.time()
        data = response.json()
        
        st.subheader('Show the answer')
        st.write(data)
        st.subheader(f"\nTime consume: {te-ts:.3f} s")
        return data


def draw_f1(event_acc, 
            col_list=['eval_f1', 'eval_HasAns_f1', 'eval_NoAns_f1'], 
            color_list=['#FF5151', '#6A6AFF', 'orange']):
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
    st.pyplot(fig)


def draw_loss(event_acc):
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
    st.pyplot(fig)


def show_the_plot():
    newest_log = get_newest_log(config['LOG_PATH'], verbose=False)
    event_acc = get_event_acc(newest_log, verbose=False)
    draw_f1(event_acc)
    draw_loss(event_acc)


def main():
    st.title("Welcome to Question Answer System!")
    st.write("Type any content and questions you want to ask")  

    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Run the app", "Show evaluation plot"])

    if app_mode == "Run the app":
        summarize()
    elif app_mode == "Show evaluation plot":
        show_the_plot()


if __name__=='__main__':
    main()