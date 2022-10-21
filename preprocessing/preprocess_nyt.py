# process NYT corpus
# Print 

## Code Modified from GitHub: kedz/summarization-datasets
## and nlpyang/Presumm

import argparse
import json
import multiprocessing
import pathlib
import re
import tarfile
from bs4 import BeautifulSoup
# import spacy
import stanza
import rouge_papier

bad_sections = set([
    "Style", "Home and Garden", "Paid Death Notices", "Automobiles",
    "Real Estate", "Week in Review", "Corrections", "The Public Editor",
    "Editors' Notes"])

def get_paths(root_dir):
    data_dir = root_dir / "data"
    paths = []
    years = [x for x in data_dir.glob("*")]
    years.sort()
    for year in years:
        months = [x for x in year.glob("*")]
        months.sort()
        for month in months:
            if month.name.endswith('tgz'):
                paths.append(month)
    return paths

def doc_iter(tar_path):
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            f = tar.extractfile(member)
            if f is None:
                continue
            content = f.read().decode("utf8")
            yield content

def get_article_text(xml):
    return "\n\n".join([p.get_text() for p in xml.find_all("p")])

def extract_doc(content):
    nyt_remove_words = ["(Photo)", "(Graph)", "(Chart)", "(Map)", "(Table)", "(Drawing)"]
    soup = BeautifulSoup(content, features="xml")
   
    sections = set() 
    for meta in soup.find_all("meta"):
        if meta["name"] == "online_sections":
            for section in meta["content"].split(";"):
                section = section.strip()
                sections.add(section)

    if len(sections.intersection(bad_sections)) > 0:
        return None

    article_xml = soup.find("block", {"class": "full_text"})
    if article_xml is None:
        return None

    article_text = get_article_text(article_xml)
    if len(article_text.split()) < 200:
        return None
      
    abstract_xml = soup.find("abstract")
    if abstract_xml is not None:
        abs_txt = abstract_xml.get_text()
    else:
        abs_txt = ""

    online_lead_xml = soup.find(
        "block", {"class": "online_lead_paragraph"})
    if online_lead_xml is not None:
        online_lead_txt = online_lead_xml.get_text()
    else: 
        online_lead_txt = ""
    if len(abs_txt.split()) + len(online_lead_txt.split()) < 100:
        return None
    doc_id = soup.find("doc-id")["id-string"]
    
    for ww in nyt_remove_words:
        abs_txt = abs_txt.replace(ww, '')

    return article_text, abs_txt, online_lead_txt, doc_id, sections

def prepare_example(article_text, abstract_text, ol_text, doc_id, sections):
    global nlp
    inputs = []
    article_text = article_text.replace("\n", " ")
    doc = nlp(article_text)
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

    summary_texts = []
    if len(abstract_text) > 0:
        summary_texts.append(abstract_text)
    input_texts = [inp["text"] if inp["word_count"] > 2 else "@@@@@"
                    for inp in inputs[:50]]

    # ROUGE 1 score.
    ranks, pairwise_ranks = rouge_papier.compute_extract(
        input_texts, summary_texts, mode="sequential", ngram=1,
        remove_stopwords=True, length=100)

    labels = [1 if r > 0 else 0 for r in ranks]
    if len(labels) < len(inputs):
        labels.extend([0] * (len(inputs) - len(labels)))
    example = {"id": doc_id, "inputs": inputs, "sections": sections}
    return example, labels, abstract_text, ol_text

def init_worker():
    global nlp
    nlp = stanza.Pipeline('en', processors='tokenize')

def worker(args):
    content, outputs_dir, model = args

    # Process xml to get document and summary text. 
    doc_data = extract_doc(content)
    if doc_data is None:
        return False
    article_text, abs_txt, online_lead_txt, doc_id, sections = doc_data
    if len(abs_txt) < 50:
        return False
    example, labels, abstract_text, ol_text = prepare_example(
        article_text, abs_txt, online_lead_txt, doc_id, sections)

    assert abstract_text == abs_txt
    assert online_lead_txt == ol_text

    assert len(labels) == len(example["inputs"])

    
    # if the amount of tokens up to the third index sentence is greater than 512, return false.
    label_locs = [i for i, value in enumerate(labels) if value == 1]
    token_count = 0
    for x in range(label_locs[2]+1):
        token_count += len(example["inputs"][x]["tokens"])
    token_count += (label_locs[2]+1)*2

    # print(label_locs)
    # print(example['id'], token_count)

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
        outputs_path = outputs_dir / "nyt.{}.test.json".format(example["id"])
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

def preprocess_part(tar_paths, outputs_dir, model, procs=16):

    outputs_dir.mkdir(exist_ok=True, parents=True)

    def data_iter():
        for tar_path in tar_paths:
            for content in doc_iter(tar_path):
                yield content, outputs_dir, model
    
    pool = multiprocessing.Pool(procs, initializer=init_worker)
    count = 0
    for i, is_good in enumerate(pool.imap(worker, data_iter()), 1):
        if is_good:
            count += 1
            print("{}".format(count), end="\r", flush=True)
    print()

def main(args):

    procs = min(multiprocessing.cpu_count(), 16)

    paths = get_paths(args.nyt)

    train_paths = paths[:-30]
    valid_paths = paths[-30:-18]
    test_paths = paths[-18:]
    print(train_paths[0], train_paths[-1])
    print(valid_paths[0], valid_paths[-1])
    print(test_paths[0], test_paths[-1])

    # Preprocess the validation data. 5000
    # preprocess_part(
    #     valid_paths, 
    #     args.data_dir / args.model / "nyt" / "valid",
    #     procs=procs)

    preprocess_part(
        test_paths, 
        args.data_dir / args.model / "nyt" / "test",
        args.model,
        procs=procs)

    # preprocess_part(
    #     train_paths, 
    #     args.data_dir / args.model / "nyt" / "train",
    #     procs=procs)

if __name__ == "__main__":
    stanza.download('en')
    parser = argparse.ArgumentParser()
    parser.add_argument("--nyt", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    main(args)