# process NYT corpus
## Code Modified from GitHub: kedz/summarization-datasets
## and nlpyang/Presumm

import json
import pathlib

global min_sents
global max_sents
global max_tokens_per_sent
global min_tokens_per_sent

def get_paths(root_dir):
    paths = [x for x in root_dir.glob("*")]
    paths.sort()
    return paths

def rindex(lst, value):
    lst.reverse()
    i = lst.index(value)
    lst.reverse()
    return len(lst) - i - 1

# Counts the min and max num of sentences
# Counts the min and max num of tokens per sentence
def count_sentences(story):
    global min_sents
    global max_sents
    global max_tokens_per_sent
    global min_tokens_per_sent
    with open(story, 'r', encoding='utf-8') as f:
        data = json.load(f)
        cur_sents_length = len(data['src'])
        salient_sents = data['tgt'].count(1)
        token_sizes = [len(tokens) for tokens in data['src']]
        curr_max_tps = max(token_sizes)
        curr_min_tps = min(token_sizes)
        if (salient_sents < 3):
            print("{} only has {} salient sentences!".format(story, salient_sents))
        if (cur_sents_length > max_sents):
            max_sents = cur_sents_length
        if (cur_sents_length < min_sents):
            min_sents = cur_sents_length
        if (curr_max_tps > max_tokens_per_sent):
            max_tokens_per_sent = curr_max_tps
        if (curr_min_tps < min_tokens_per_sent):
            min_tokens_per_sent = curr_min_tps

def main():
    global min_sents
    global max_sents
    global max_tokens_per_sent
    global min_tokens_per_sent
    min_sents = 100
    max_sents = 0
    max_tokens_per_sent = 0
    min_tokens_per_sent = 100
    test_paths  = get_paths(pathlib.Path("processed_data/presumm/nyt/test/"))
    for story in test_paths:
        count_sentences(story)
    print("NYT: max number of sentences: ", max_sents)
    print("NYT: min number of sentences: ", min_sents)
    print("NYT: max number of tokens per sentence: ", max_tokens_per_sent)
    print("NYT: min number of tokens per sentence: ", min_tokens_per_sent)
    min_sents = 100
    max_sents = 0
    max_tokens_per_sent = 0
    min_tokens_per_sent = 100
    test_paths  = get_paths(pathlib.Path("processed_data/presumm/turning_point/test/"))
    for story in test_paths:
        count_sentences(story)
    print("Turning Point: max number of sentences: ", max_sents)
    print("Turning Point: min number of sentences: ", min_sents)
    print("Turning Point: max number of tokens per sentence: ", max_tokens_per_sent)
    print("Turning Point: min number of tokens per sentence: ", min_tokens_per_sent)
    min_sents = 100
    max_sents = 0
    max_tokens_per_sent = 0
    min_tokens_per_sent = 100
    test_paths  = get_paths(pathlib.Path("processed_data/presumm/propp_learner/test/"))
    for story in test_paths:
        count_sentences(story)
    print("Propp Learner: max number of sentences: ", max_sents)
    print("Propp Learner: min number of sentences: ", min_sents)
    print("Propp Learner: max number of tokens per sentence: ", max_tokens_per_sent)
    print("Propp Learner: min number of tokens per sentence: ", min_tokens_per_sent)
    

if __name__ == "__main__":
    main()