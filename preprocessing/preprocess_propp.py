# process ProppLearner corpus

## Code Modified from GitHub: kedz/summarization-datasets
## and nlpyang/Presumm

import argparse
import csv
import json
from lib2to3.pgen2 import token
from lib2to3.pgen2.tokenize import tokenize
import multiprocessing
import pathlib
import re
import stanza

def get_paths(root_dir):
    paths = [x for x in root_dir.glob("*")]
    paths.sort()
    return paths

def extract_doc(content):
    with open(content, mode='r', encoding='utf-8') as f:
        read_csv = csv.reader(f, delimiter='\t')

        text = []
        labels = []
        for row in read_csv:
            labels.append(int(row[0]))
            text.append(row[1])
        # We don't want 0 <|endoftext|> at the end.
        return "\n\n".join(text[:-1]), labels[:-1]

def prepare_example(summary_text, ranks, story_num, nlp):
    inputs = []
    doc = nlp(summary_text)
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

    labels = [1 if r > 0 else 0 for r in ranks]
    if len(labels) < len(inputs):
        labels.extend([0] * (len(inputs) - len(labels)))
    example = {"id": story_num, "inputs": inputs}
    return example, labels

def save_output(outputs_dir, example, labels, model):
    # if the amount of tokens up to the third index sentence is greater than 512, return false.
    label_locs = [i for i, value in enumerate(labels) if value == 1]
    token_count = 0
    for x in range(label_locs[2]+1):
        token_count += len(example["inputs"][x]["tokens"])
    token_count += (label_locs[2]+1)*2

    print(label_locs)
    print(example['id'], token_count)

    if token_count > 512:
        return False

    if model == "barthes":
        outputs_path = outputs_dir / "{}.tsv".format(example["id"])
        with open(outputs_path, "w", encoding='utf-8') as outfile:
                for i, sent in enumerate(example["inputs"][::]):
                    text = sent["tokens"]
                    pretty_text = " ".join(text)
                    pretty_text = re.sub(r"\r|\n|\t", r" ", pretty_text)
                    pretty_text = re.sub(r"\s+", r" ", pretty_text)
                    if labels[i] == 1:
                        print("1\t", end="", file=outfile)
                        print(pretty_text, file=outfile)
                    else:
                        print("0\t", end="", file=outfile)
                        print(pretty_text, file=outfile)
                print("0\t", end="", file=outfile)
                print("<|endoftext|>", end="", file=outfile)
        return True
    
    elif model == "presumm":
        outputs_path = outputs_dir / "propp.{}.test.json".format(example["id"])
        with open(outputs_path, "w", encoding='utf-8') as outfile:
                doc = {
                        "src": [],
                        "tgt": labels
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

def process(content, outputs_dir, model, nlp, story_num):

    # Process tsv to get document and summary text. 
    doc_data = extract_doc(content)
    if doc_data is None:
        return False
    summary, labels = doc_data
    assert len(summary.split("\n\n")) == len(labels)
    example, labels = prepare_example(
        summary, labels, story_num, nlp)

    save_output(outputs_dir, example, labels, model)  

def preprocess_part(input_dir, outputs_dir, model, procs=16):
    outputs_dir.mkdir(exist_ok=True, parents=True)
    nlp = stanza.Pipeline('en', processors='tokenize', tokenize_pretokenized=True)
    for idx, story in enumerate(input_dir):
        process(story, outputs_dir, model, nlp, idx)
    

def main(args):

    procs = min(multiprocessing.cpu_count(), 16)

    test_path  = get_paths(pathlib.Path("../coling2020-modeling-event-salience/data-processed"))
    print(test_path)

    preprocess_part(
        test_path, 
        args.data_dir / args.model / "propp_learner" / "test",
        args.model,
        procs=procs)

if __name__ == "__main__":
    stanza.download('en')
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    main(args)