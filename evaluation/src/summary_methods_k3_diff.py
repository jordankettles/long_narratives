import json
import os
import pathlib
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt

def get_chapter_data(line):
	chapter = json.loads(line)
	return {
        'book_id': chapter['book_id'].replace(" ", "_").lower(),
        'source': chapter['source']
    }

def sort_by_score(x):
		return x["prediction"]

def get_text_and_scores(result_path):
	df = pd.read_csv(result_path, sep='\t', engine='python')
	text = df["text"].astype(str)
	predictions = df["score"].astype(float)
	assert len(predictions) == len(text)
	# print(predictions)
	# print(text)
	return {"predictions": list(predictions), "text": list(text), "filename": result_path}

def get_human_summary(summary_path):
    with open("../booksum/scripts/" + summary_path, encoding='utf-8') as f:
        summary_json = json.load(f)
        return summary_json["summary"]

def generate_concat_summary(related_files, k):
	cat_related_files = []
	for related_file in related_files:
		cat_related_files.extend([{"prediction": related_file["predictions"][idx],\
			 "text": related_file["text"][idx]} for idx in range(len(related_file["predictions"]))])
	cat_related_files.sort(key=sort_by_score, reverse=True)
	cat_related_files = cat_related_files[:((k)*len(related_files))]
	print(k*len(related_files))
	generated_summary = [sentence["text"] for sentence in cat_related_files]
	return generated_summary

def generate_each_summary(related_files, k):
	generated_summary_list = []
	related_files_list = []
	for related_file in related_files:
		related_files_list.append([{"prediction": related_file["predictions"][idx],\
			 "text": related_file["text"][idx]} for idx in range(len(related_file["predictions"]))])
	for related_file in related_files_list:
		related_file.sort(key=sort_by_score, reverse=True)
		generated_summary_list.extend(related_file[:(k)])
	generated_summary = [sentence["text"] for sentence in generated_summary_list]
	return generated_summary

def calculate_F1():
	global chapters_data
	#text_and_scores = [get_text_and_scores(result_path) for result_path in input_path_list]
	used_files = []
	summary_lens = [] # int
	only_each_lens = [] # int
	only_concat_lens = [] # int
	shared_sentences_lens = []  # int

	k = 3

	f = open(pathlib.Path("../booksum/alignments/book-level-summary-alignments/book_summaries_aligned_all.jsonl"), \
		encoding='utf-8')
	
	book_count = 0

	for line in f:
		# print(line)
		content = json.loads(line)
		summary_path = content['summary_path']
		book_title = content['title']
		source = content['source']
		# Format title.
		book_title = book_title.replace(':', "").replace(" ", "_").lower()

		try:
			human_summary = get_human_summary(summary_path)
			# print("Summary Length : {}".format(len(human_summary)))
		except:
			human_summary = None
			print("Could not find summary.")
		
		# pull out all files related to that summary from data.
		# Modify to use the same lsting method as booksum_bookify.
		# related_files = 
		chapters = [x for x in chapters_data if x['book_id'].split('.')[0] == book_title and x['source'] == source]
		paths = [pathlib.Path("../PreSumm/results/booksum_summaries/20000/").joinpath( \
			"ext_bertbooksum_summaries." + chapter['book_id'].strip() + "." + chapter['source'].strip() + '.tsv') for chapter in chapters]
		related_files = []
		for path in paths:
			try:
				related_files.append(get_text_and_scores(path))
			except:
				pass
		used_files.extend(related_files)
		
		if len(related_files) == 0:
			continue

		elif (human_summary is None):
			print("Files exist but no summary.")
			print(summary_path)

		# Generate extractive summaries.
		extractive_concat_summary = generate_concat_summary(related_files, k)
		extractive_each_summary = generate_each_summary(related_files, k)

		# try:
		# There are some chapter summaries that are less than 3 sentences long, so some each summaries are shorter than concatenated summaries.
		# 	assert len(extractive_concat_summary) == len(extractive_each_summary)
		# except:
		# 	print("Could not match {}, {}. len each: {}, len concat: {}".format(book_title, source, len(extractive_each_summary), len(extractive_concat_summary)))
		if len(extractive_concat_summary) > len(extractive_each_summary):
			summary_lens.append(len(extractive_concat_summary))
		else:
			summary_lens.append(len(extractive_each_summary))

		extractive_concat_summary.sort()
		extractive_each_summary.sort()

		only_concat = list(set(extractive_concat_summary) - set(extractive_each_summary))
		only_each = list(set(extractive_each_summary) - set(extractive_concat_summary))
		shared_sentences = list(set(extractive_concat_summary) & set(extractive_each_summary))
		only_concat_lens.append(len(only_concat))
		only_each_lens.append(len(only_each))
		shared_sentences_lens.append(len(shared_sentences))

		

		# if (book_count == 3 or book_count == 4):
		# 	print("Concat Only: ")
		# 	print(only_concat)
		# 	print()
		# 	print("Each Only: ")
		# 	print(only_each)
		# 	print()
		# 	print("Shared Sentences: ")
		# 	print(shared_sentences)


		book_count += 1

		# results.append(curr_result)
	print("k: {}".format(k+1))
	print("Books covered: {}".format(book_count))
	print()

	mean_only_concat_len = sum(only_concat_lens) / len(only_concat_lens)
	mean_only_each_len = sum(only_each_lens) / len(only_each_lens)
	mean_shared_sentences_len = sum(shared_sentences_lens) / len(shared_sentences_lens)

	print("Mean Concat Only Length: {}".format(mean_only_concat_len))
	print("Mean Each Only Length: {}".format(mean_only_each_len))
	print("Mean Shared Sentences Length: {}".format(mean_shared_sentences_len))

	return {"summary_lens": summary_lens, "only_concat_lens": only_concat_lens, "only_each_lens": only_each_lens, "shared_sentences_lens": shared_sentences_lens}

def draw_graph(max_k, results): # TODO edit
	print("Drawing graph...")
	fig, axes = plt.subplots(2, 5, sharex=True, sharey=True)
	for k, ax in zip(range(max_k), axes.flatten()):
		diff_data = [x[1] for x in results[k]["diff_data"]]
		ax.hist(diff_data, alpha=0.5, bins=20)
		ax.set_title("k = {}".format(k+1))
	plt.suptitle("Distribution of Differences Between Summary Methods")
	fig.supxlabel("Difference in ROUGE-1 F1 Score")
	fig.supylabel("Frequency")
	plt.xlim([-0.2, 0.1])
	# plt.savefig("../graphs/booksum_summaries_dists.pdf")


# def save_file(results):
# 	print("Saving results...")
# 	results_to_save = [result for k in results for result in k]
# 	df = pd.DataFrame(results_to_save, columns=["k", "diff value", "summary path"])
# 	# df.to_csv("../csv_results/booksum_summaries/booksumsummdiffs-" + "-results.csv") # Save file.

chapters_test_split = open(pathlib.Path("../booksum/alignments/chapter-level-summary-alignments/chapter_summary_aligned_all_split.jsonl"), \
     encoding='utf-8')
chapters_data = [get_chapter_data(line) for line in chapters_test_split]

def main():
	input_path_list = glob(os.path.normpath("../PreSumm/results/booksum_summaries/20000/") + "/*")
	print("Evaluation results from {} files".format(len(input_path_list)))
	
	results = calculate_F1()
	# draw_graph(args.k, results)
	# save_file(results)

if __name__ == "__main__":
	main()