# process booksum paragraph results into chapter level examples.
# Jordan Kettles, 2022

## Code Modified from GitHub: kedz/summarization-datasets
## and nlpyang/Presumm
import argparse
import json
import multiprocessing
import os
import pathlib
import re
import rouge_papier
import pandas as pd
import stanza
import math

def get_paths(root_dir):
    paths = [x for x in root_dir.glob("chapter_summary_aligned_*_split.jsonl")]
    return paths

def extract_doc(content):
    return content['summary_path'],  content['book_id'], content['source']

def get_summary(summary_path):
    summary_path = pathlib.Path("../booksum/scripts/" + summary_path)
    with open(summary_path, encoding='utf-8') as f:
        summary_json = json.load(f)
        return summary_json["summary"]

def prepare_example(text, file_name):
    global nlp
    inputs = []
    doc = nlp(text)
    for sent in doc.sentences:
        tokens_all = [w for w in sent.words
                        if w.text.strip() != '']
        if len(tokens_all) == 0:
            continue
        tokens = [w.text.strip() for w in tokens_all]
        pretty_text = sent.text.strip()
        pretty_text = re.sub(r"\r|\n|\t", r" ", pretty_text)
        pretty_text = re.sub(r"\s+", r" ", pretty_text)
        inputs.append({"tokens": tokens, "text": pretty_text, "word_count": len(pretty_text.split())})
    for i, inp in enumerate(inputs, 1):
        inp["sentence_id"] = i

    input_texts = [inp["text"] if inp["word_count"] > 2 else "@@@@@"
                    for inp in inputs[:50]] # inputs[:50]

    # for text in input_texts:
    #     print(text)
    # print(summary)
    example = {"id": file_name, "inputs": inputs}

    return example

def save_output(outputs_dir, example):
    # No labels for this one, we want to compare using ROUGE-1 F1 Score.
    outputs_path = outputs_dir / "booksum_summaries.{}.test.json".format(example["id"])
    with open(outputs_path, "w", encoding='utf-8') as outfile:
        doc = {
                "src": [],
                }
        for i, sent in enumerate(example["inputs"][::]):
            # save each sentence as a list.
            text = sent["tokens"]
            pretty_text = " ".join(text)
            pretty_text = re.sub(r"\r|\n|\t", r" ", pretty_text)
            pretty_text = re.sub(r"\s+", r" ", pretty_text)
            words = pretty_text.split()
            doc["src"].append(words)
        json.dump(doc, outfile)

        return True

def init_worker():
    global nlp
    nlp = stanza.Pipeline('en', processors='tokenize')

def worker(args):

    content, outputs_dir = args

    # Process BookSum JSON file containing chapter information to get document and summary text. 
    doc_data = extract_doc(content)
    if doc_data is None:
        return False
    summary_path, chapter_id, source = doc_data
    # book id = book (+ act # + ) + scene # / chapter #

    
    # bookid replace " " with "_" and lowercase
    chapter_id = chapter_id.replace(" ", "_").lower() + "." + source


    outputs_path = outputs_dir / "booksum_summaries.{}.test.json".format(chapter_id)

    if os.path.exists(outputs_path):
        return False

    # BookSum script failed to download pinkmonkey summaries.
    if (source == "pinkmonkey"):
        return False
    
    # get summary
    try:
        summary = get_summary(summary_path.replace(":", ""))
    except FileNotFoundError:
        return False

    try:
        assert len(summary) > 0
    except:
        return False
    # Create labels
    example = prepare_example(summary, chapter_id)
    
    save_output(outputs_dir, example)
    
    
# Processes each example individually.
def preprocess_part(chapters_dir, outputs_dir, procs=16):

    # Open JSON file
    outputs_dir.mkdir(exist_ok=True, parents=True)

    f = open(chapters_dir, encoding='utf-8')

    def data_iter():
        for line in f:
            content = json.loads(line)
            # yield the content of the JSON
            yield content, outputs_dir

    pool = multiprocessing.Pool(procs, initializer=init_worker)
    count = 0
    for i, is_good in enumerate(pool.imap(worker, data_iter()), 1):
        if is_good:
            count += 1
            print("{}".format(count), end="\r", flush=True)
    print()

def main():

    procs = 8

    paths = get_paths(pathlib.Path("../booksum/alignments/chapter-level-summary-alignments"))

    print("Processing part 1 / 3 {}".format(paths[0]))

    preprocess_part(
        paths[0],
        pathlib.Path("processed_data") / "presumm" / "booksum_summaries",
        procs=procs)

    print("Processing part 2 / 3 {}".format(paths[1]))

    preprocess_part(
        paths[1],
        pathlib.Path("processed_data") / "presumm" / "booksum_summaries",
        procs=procs)

    print("Processing part 3 / 3 {}".format(paths[2]))

    preprocess_part(
        paths[2],
        pathlib.Path("processed_data") / "presumm" / "booksum_summaries",
        procs=procs)


if __name__ == "__main__":
    main()