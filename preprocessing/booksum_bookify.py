# process booksum chapter level results into book level examples.
# Apologies for making Bookify a verb.
# Jordan Kettles, 2022

## Code Modified from GitHub: kedz/summarization-datasets
## and nlpyang/Presumm
import os
import argparse
import json
import multiprocessing
import pathlib
import re
import rouge_papier


presumm_threshold = 0.25

def extract_doc(content):
    return content['summary_path'],  content['title'], content['source']

def get_summary(summary_path):
    summary_path = pathlib.Path("../booksum/scripts/" + summary_path)
    with open(summary_path, encoding='utf-8') as f:
        summary_json = json.load(f)
        return summary_json["summary"]

def get_text(book_title, source, input_dir):

    text_lines = []

    paths = [x for x in input_dir.glob("ext_bert_booksum." + book_title + "*" + source + "*")]
    # paths.sort()
    print(paths)
    # for example in paths:
    #     df = pd.read_csv(example, sep='\t', engine='python', encoding='utf-8')
    #     predictions = df["score"].astype(float) # Paragraph level predictions.
    #     text = df["text"].astype(str) # this is pre-tokenized.
    #     pred_locs = [idx for idx, guess in enumerate(predictions) if guess >= presumm_threshold]
    #     for prediction in pred_locs:
    #         text_lines.append(text[prediction])
    # return text_lines


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
                    for inp in inputs[:50]] # inputs[:50]

    # for text in input_texts:
    #     print(text)
    # print(summary)

    ranks, pairwise_ranks = rouge_papier.compute_extract(
        input_texts, [summary], mode="sequential", ngram=1,
        remove_stopwords=True, length=100)

    labels = [1 if r > 0 else 0 for r in ranks]
    # why is this if statement here?
    if len(labels) < len(inputs):
        labels.extend([0] * (len(inputs) - len(labels)))
    example = {"id": file_name, "inputs": inputs}

    return example, labels

def save_output(outputs_dir, example, labels, model):

    # print(label_locs)
    # print(example['id'], token_count)

    if model == "presumm":

        outputs_path = outputs_dir / "booksum.{}.test.json".format(example["id"])
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
    return False

def worker(args):

    content, outputs_dir, input_dir, model = args

    # Process BookSum JSON file containing chapter information to get document and summary text. 
    doc_data = extract_doc(content)
    if doc_data is None:
        return False
    summary_path, book_title, source = doc_data

    # BookSum script failed to download pinkmonkey summaries.
    if (source == "pinkmonkey"):
        return False
    
    # bookid replace " " with "_" and lowercase
    book_title = book_title.replace(" ", "_").lower()

    # get text

    outputs_path = outputs_dir / "booksum.{}.test.json".format(book_title)
    if os.path.exists(outputs_path):
        return False

    
    text = get_text(book_title, source, input_dir)

    # if len(text) == 0:
    #     print("Couldn't find chapter " + chapter_id)
    #     return False


    # # get summary
    # summary = get_summary(summary_path)

    # assert len(summary) > 0

    # # Create labels
    # example, labels = prepare_example(text, summary, chapter_id)

    # print(example, labels)

    # assert len(labels) == len(example['inputs'])
    

    # save_output(outputs_dir, example, labels, model)
        

def preprocess_part(input_dir, books_dir, outputs_dir, model, procs=16):

    # Open JSON file
    outputs_dir.mkdir(exist_ok=True, parents=True)

    f = open(books_dir, encoding='utf-8')
    
    def data_iter():
        for line in f:
            content = json.loads(line)
            # yield the content of the JSON
            yield content, outputs_dir, input_dir, model

    pool = multiprocessing.Pool(procs)
    count = 0
    for i, is_good in enumerate(pool.imap(worker, data_iter()), 1):
        if is_good:
            count += 1
            print("{}".format(count), end="\r", flush=True)
    print()
    

def main(args):

    procs = 1
    test_path = pathlib.Path("../booksum/alignments/book-level-summary-alignments/book_summaries/book_summaries_aligned_test.jsonl")

    if args.model == "presumm":
        print("The threshold set is {}.".format(presumm_threshold))

    preprocess_part(
        args.input,
        test_path,
        pathlib.Path("processed_data") / args.model / "booksum_book" / "test",
        args.model,
        procs=procs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    main(args)