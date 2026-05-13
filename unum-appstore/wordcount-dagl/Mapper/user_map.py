from mapreduce import emitPerReducerSingle

def user_map(event):
    text = event
    words = text.split()
    for word in words:
        emitPerReducerSingle(word)
