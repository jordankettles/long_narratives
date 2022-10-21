# process Booksum Corpus

## Code Modified from GitHub: kedz/summarization-datasets
## and nlpyang/Presumm
import argparse
import json
import multiprocessing
import os
import pathlib
import re
import stanza
import rouge_papier

def get_paths(root_dir):
    print(root_dir)
    paths = [x for x in root_dir.glob("chapter_summary_aligned_*_split.jsonl.gathered.stable")]
    paths.sort()
    return paths

def extract_doc(content):
    return content['text'],  content['summary'], content['title']

def prepare_example(text, summary, file_name):
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
                    for inp in inputs[:50]]

    ranks, pairwise_ranks = rouge_papier.compute_extract(
        input_texts, summary, mode="sequential", ngram=1,
        remove_stopwords=True, length=100)

    labels = [1 if r > 0 else 0 for r in ranks]
    if len(labels) < len(inputs):
        labels.extend([0] * (len(inputs) - len(labels)))
    example = {"id": file_name, "inputs": inputs}
    return example, labels

def save_output(outputs_dir, example, labels, model):

    # if the amount of tokens up to the third index sentence is greater than 512, return false.
    # label_locs = [i for i, value in enumerate(labels) if value == 1]
    # token_count = 0
    # for x in range(label_locs[2]+1):
    #     token_count += len(example["inputs"][x]["tokens"])
    # token_count += (label_locs[2]+1)*2

    # if token_count > 512:
    #     return False

    # print(label_locs)
    # print(example['id'], token_count)

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
        outputs_path = outputs_dir / "booksum.{}.train.json".format(example["id"])
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

def init_worker():
    global nlp
    nlp = stanza.Pipeline('en', processors='tokenize')

def worker(args):

    content, outputs_dir, model = args

    # Process JSON object to get document and summary text. 
    doc_data = extract_doc(content)
    if doc_data is None:
        return False
    text, summary, file_name = doc_data

    outputs_path = outputs_dir / "{}.tsv".format(file_name)

    if os.path.exists(outputs_path):
        return False
    example, labels = prepare_example(text, summary, file_name)

    assert len(labels) == len(example['inputs'])

    save_output(outputs_dir, example, labels, model)
        

def preprocess_part(input_dir, outputs_dir, model, procs=16):

    # Open JSON file
    outputs_dir.mkdir(exist_ok=True, parents=True)

    f = open(input_dir, encoding='utf-8')
    
    def data_iter():
        for line in f:
            content = json.loads(line)
            # yield the content of the JSON
            yield content, outputs_dir, model

    pool = multiprocessing.Pool(procs, initializer=init_worker)
    count = 0
    for i, is_good in enumerate(pool.imap(worker, data_iter()), 1):
        if is_good:
            count += 1
            print("{}".format(count), end="\r", flush=True)
    print()
    

def main(args):

    procs = 8

    paths = get_paths(pathlib.Path("../booksum/alignments/paragraph-level-summary-alignments"))
    
    test_path = paths[0]
    train_path = paths[1]
    valid_path = paths[2]



    preprocess_part(
        valid_path, 
        args.output / args.model / "booksum" / "valid",
        args.model,
        procs=procs)

    # preprocess_part(
    #     test_path, 
    #     args.output / args.model / "booksum" / "test",
    #     args.model,
    #     procs=procs)

    # preprocess_part(
    #     train_path, 
    #     args.output / args.model / "booksum" / "train",
    #     args.model,
    #     procs=procs)

if __name__ == "__main__":
    stanza.download('en')
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    main(args)