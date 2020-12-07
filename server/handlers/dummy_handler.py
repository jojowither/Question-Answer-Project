from question_answer_project.server.handlers.qa_handler import handler

# because the CWD of buliding the torchserve is /tmp/xxxx,
# we must use dummy handler to change work directory to get the model path
def handle(data, context):
    return handler(data, context)