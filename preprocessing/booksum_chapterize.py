# process booksum paragraph results into chapter level examples.
# Jordan Kettles, 2022

## Code Modified from GitHub: kedz/summarization-datasets
## and nlpyang/Presumm
import argparse
import json
import multiprocessing
import pathlib
import re
import rouge_papier
import pandas as pd

presumm_threshold = 0.25

def get_paths(root_dir):
    paths = [x for x in root_dir.glob("chapter_summary_aligned_*_split.jsonl")]
    paths.sort()
    return paths

def extract_doc(content):
    return content['summary_path'],  content['book_id'], content['source']

def get_summary(summary_path):
    summary_path = pathlib.Path("../booksum/scripts/" + summary_path)
    with open(summary_path, encoding='utf-8') as f:
        summary_json = json.load(f)
        return summary_json["summary"]

def get_text(chapter_id, input_dir):
    paths = [x for x in input_dir.glob("ext_bert_booksum." + chapter_id + "*")]
    paths.sort()
    print(paths)
    text_lines = []
    for example in paths:
        df = pd.read_csv(example, sep='\t', engine='python', encoding='utf-8')
        gold_label = df["label"].astype(int) # Paragraph level labels.
        predictions = df["score"].astype(float) # Paragraph level predictions.
        text = df["text"].astype(str) # this is pre-tokenized.
        label_locs = [idx for idx, label in enumerate(gold_label) if label == 1]
        pred_locs = [idx for idx, guess in enumerate(predictions) if guess >= presumm_threshold]
        for prediction in pred_locs:
            text_lines.append(text[prediction])
    return text_lines


def prepare_example(text, summary, file_name):
    inputs = []
    for sent in text:
        tokens_all = [w for w in sent.split(" ")
                        if w.strip() != '']
        if len(tokens_all) == 0:
            continue
        tokens = [w.strip() for w in tokens_all]
        pretty_text = sent.strip()
        pretty_text = re.sub(r"\r|\n|\t", r" ", pretty_text)
        pretty_text = re.sub(r"\s+", r" ", pretty_text)
        inputs.append({"tokens": tokens, "text": pretty_text, "word_count": len(pretty_text.split())})
    for i, inp in enumerate(inputs, 1):
        inp["sentence_id"] = i

    input_texts = [inp["text"] if inp["word_count"] > 2 else "@@@@@"
                    for inp in inputs[:50]] 

    ranks, pairwise_ranks = rouge_papier.compute_extract(
        input_texts, [summary], mode="sequential", ngram=1,
        remove_stopwords=True, length=100)

    labels = [1 if r > 0 else 0 for r in ranks]
    # why is this if statement here?
    if len(labels) < len(inputs):
        labels.extend([0] * (len(inputs) - len(labels)))
    example = {"id": file_name, "inputs": inputs}
    return example, labels

def save_output(outputs_dir, example, labels):

    # print(label_locs)
    # print(example['id'], token_count)
    if 'test' in str(outputs_dir):
        outputs_path = outputs_dir / "booksum.{}.test.json".format(example["id"])
    elif 'valid' in str(outputs_dir):
        outputs_path = outputs_dir / "booksum.{}.valid.json".format(example["id"])
    else:
        print("Invalid output dir")
        exit(-1)
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

def worker(args):

    content, outputs_dir, input_dir = args

    # Process JSON object to get document and summary text. 
    doc_data = extract_doc(content)
    if doc_data is None:
        return False
    summary_path, chapter_id, source = doc_data
    # book id = book (+ act # + ) + scene # / chapter #

    # BookSum script failed to download pinkmonkey summaries.
    if (source == "pinkmonkey"):
        return False

    # bookid replace " " with "_" and lowercase
    chapter_id = chapter_id.replace(" ", "_").lower() + "." + source

    # get text
    text = get_text(chapter_id + "-stable-", input_dir)

    if (len(text) == 0):
        print("Couldn't find chapter " + chapter_id)
        return False

    # # get summary
    # summary = get_summary(summary_path)

    # assert len(summary) > 0

    # # Create labels
    # example, labels = prepare_example(text, summary, chapter_id)

    # assert len(labels) == len(example['inputs'])
    
    # save_output(outputs_dir, example, labels)
        

def preprocess_part(input_dir, chapters_dir, outputs_dir, procs=8):

    # Open JSON file
    outputs_dir.mkdir(exist_ok=True, parents=True)

    f = open(chapters_dir, encoding='utf-8')
    
    def data_iter():
        for line in f:
            content = json.loads(line)
            # yield the content of the JSON
            yield content, outputs_dir, input_dir

    pool = multiprocessing.Pool(procs)
    count = 0
    for i, is_good in enumerate(pool.imap(worker, data_iter()), 1):
        if is_good:
            count += 1
            print("{}".format(count), end="\r", flush=True)
    print()
    

def main(args):

    procs = 1

    paths = get_paths(pathlib.Path("../booksum/alignments/chapter-level-summary-alignments"))
    
    test_path = paths[0]
    train_path = paths[1]
    val_path = paths[2]

    preprocess_part(
        args.input,
        test_path, 
        pathlib.Path("processed_data")  / "presumm" / "booksum_chapter" / "test",
        procs=procs)

    # preprocess_part(
    # args.input,
    # val_path, 
    # pathlib.Path("processed_data")  / "presumm" / "booksum_chapter" / "valid",
    # procs=procs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=pathlib.Path, required=True)
    args = parser.parse_args()
    main(args)
