import json
import pathlib
import pandas as pd
import rouge
import nltk

def get_human_summary(summary_path):
	try:
		with open("../../booksum/scripts/" + summary_path, encoding='utf-8') as f:
			summary_json = json.load(f)
			return summary_json["summary"]
	except Exception as e:
		print("Failed to read summary file: {}".format(e))
		return None

def calculate_F1():
	summaries_count = 0
	data = []
	used_files = []
	unique_books = set()
	unique_used_books = set()
	human_summaries = dict()

	chapter_data = pd.read_csv(
        "../csv_results/booksum_summaries/chapter-level-sum-comparison-results.csv")

	chapter_result_books = set()

	for index, row in chapter_data.iterrows():
		chapter_result_books.add(row["source"] + row['chapter-title'].split('.')[0])

	f = open(pathlib.Path("../../booksum/alignments/book-level-summary-alignments/book_summaries_aligned_all.jsonl"), \
		encoding='utf-8')

	for line in f:
		content = json.loads(line)
		text = get_human_summary(content['summary_path'])
		if text is not None:
			human_summaries[content['summary_path']] = {
				"book_title": content['title'],
				"source": content['source'],
				"summary_text": text,
				}

	print("Evaluating {} summary documents...".format(len(human_summaries)))

	for summary_path, summary in human_summaries.items():

		# Check if the summary is in the chapter-level results.
		# if summary['source'] + summary['book_title'] not in chapter_result_books:
		# 	continue
		# Get all related summary documents.
		unique_books.add(summary['book_title'])
		# Special case for Around the World in Eighty (80) Days
		if summary['book_title'] == "Around the World in Eighty Days":
			related_summaries = list(filter(lambda curr_summary: curr_summary['book_title'] == 'Around the World in 80 Days', human_summaries.values()))

		elif summary['book_title'] == "Around the World in 80 Days":
			related_summaries = list(filter(lambda curr_summary: curr_summary['book_title'] == 'Around the World in Eighty Days', human_summaries.values()))

		else:
			related_summaries = list(filter(lambda curr_summary: curr_summary['book_title'] == summary['book_title'] and curr_summary['source'] != summary['source'], human_summaries.values()))
		used_files.extend(related_summaries) # Remember which files have been used.

		# if there are no related summary documents, then just print.
		if len(related_summaries) == 0:
			print("No related summary documents were found for {}.".format(summary['book_title']))
			continue

		# # Run the ROUGE command using the current summary as the reference and the related summaries as hypotheses.
		# # Print the scores and save them.
		related_summary_texts = [curr_summary['summary_text'] for curr_summary in related_summaries]

		evaluator = rouge.Rouge(metrics=['rouge-n'],
					max_n=1,
					limit_length=False)
		scores = evaluator.get_scores(summary['summary_text'], related_summary_texts)

		# print(scores['rouge-1'])
		data.append([scores['rouge-1']['f'], summary_path])
		unique_used_books.add(summary['book_title'])
		summaries_count += 1
		

	print("Unique books covered: {}".format(len(unique_books)))
	print("Unique books used: {}".format(len(unique_used_books)))
	ROUGE_list = [data_item[0] for data_item in data]
	ROUGE_mean = sum(ROUGE_list) / len(ROUGE_list)
	print("Mean ROUGE-1 F1: {}".format(ROUGE_mean))
	print()

	# # Comment these out to avoid saving the csv files.
	df = pd.DataFrame(data, columns=["ROUGE-1 F1", "summary path"])
	df.to_csv("../csv_results/booksum_summaries/lb-sum-comparison-results.csv") # Save file.
	# # Create graph.
	# x = [i + 1 for i in range(max_k)]
	# for mode in modes:
	# 	plt.plot(x, results[mode]["means"], marker='.')
	# plt.title("Generated Summary Methods Compared with a Human Summary")
	# plt.xlabel("Number of Sentences Per Chapter Summary (k)")
	# plt.locator_params(axis="x", integer=True)
	# plt.ylabel("Mean ROUGE-1 F1 Score")
	# plt.legend(labels=["Top k Sentence(s) per Chapter Summary", "Top k Sentence(s) over all Chapter Summaries"])
	# plt.savefig("booksum_summaries.pdf")
	# plt.show(block=True)
	# # TODO change labels.

if __name__ == "__main__":
	nltk.download('punkt')
	calculate_F1()