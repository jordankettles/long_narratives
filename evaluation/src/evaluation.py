from itertools import repeat
import os
import argparse
import random
import pandas as pd
from glob import glob
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, accuracy_score

def calculate_mean_score_diff(input_path_list):
	pos_score_list = []
	neg_score_list = []
	for file_path in input_path_list:
		df = pd.read_csv(file_path, sep='\t', engine='python')
		for _, series in df[["label", "score"]].iterrows():
			if series["label"] == 1:
				pos_score_list.append(series["score"])
			else:
				neg_score_list.append(series["score"])
	return sum(pos_score_list) / len(pos_score_list), sum(neg_score_list)/len(neg_score_list)

def count_labels(result_file_path):
	df = pd.read_csv(result_file_path, sep='\t', engine='python')

	labels = df["label"].astype(int)

	return len(labels)

def get_labels_and_scores(result_file_path):
	df = pd.read_csv(result_file_path, sep='\t', engine='python')
	labels = df["label"].astype(int)
	predictions = df["score"].astype(float)
	# print(scores)
	# print(df)

	return predictions, labels

def calculate_labels(input_path_list):
	count = 0
	for result_file_path in input_path_list:
			count += count_labels(result_file_path)
	return count

def calculate_scores(predictions, labels, threshold, is_random):

	def binary_labeller(x, threshold):
		if is_random:
			x = random.random()
		if x < 0:
			x = 0.0
		if x >= threshold:
			return 1
		else:
			return 0
	
	scores = list(map(binary_labeller, predictions, repeat(threshold)))

	precision = precision_score(labels, scores)
	recall = recall_score(labels, scores)
	F1_score = f1_score(labels, scores)
	accuracy = accuracy_score(labels, scores)
	
	return precision, recall, F1_score, accuracy

def calculate_F1(input_path_list, output_name, is_random): # TODO Rewrite this function.
	data = []
	predictions_list = []
	labels_list = []

	thresholds = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4] # Default. 28/07

	for result_file_path in input_path_list:
			predictions, labels  = get_labels_and_scores(result_file_path)
			assert len(predictions) == len(labels)
			predictions_list.extend(predictions[:-1])
			labels_list.extend(labels[:-1])
	
	for threshold in thresholds:

		precision, recall, F1_score, accuracy = calculate_scores(predictions_list, labels_list, threshold, is_random)

		data.append([threshold, precision, recall, F1_score, accuracy])

		print("Threshold: {}".format(threshold))
		print("Precision: {}".format(precision))
		print("Recall: {}".format(recall))
		print("F1: {}".format(F1_score))
		print("Accuracy: {}".format(accuracy))

	random_f1 = sum([x[3] for x in data]) / len(data)
	if random:
		print("Random: {}".format(random_f1))
	else:
		# Create filename
		output_path = "./" + output_name + "-results.csv"
		df = pd.DataFrame(data, columns=["threshold", "precision", "recall", "f1", "accuracy"])
		# Save file.
		df.to_csv(output_path)

def calculate_AP_for_story(result_file_path):
		
	df = pd.read_csv(result_file_path, sep='\t', engine='python')

	gold_label = df["label"].astype(int)
	scores = df["score"].apply(lambda x: x[0].replace("\n", ""))
	scores = df["score"].astype(float)
	is_bad = False
	if(gold_label.to_list().count(1) < 3):
		print(result_file_path)
		is_bad = True
	ap = average_precision_score(gold_label, scores)
	spearman_corr = df[["label", "score"]].corr(method="spearman").iloc[0, 1]


	return ap, spearman_corr, is_bad

def calculate_MeanAP(input_path_list):

	ap_score_list = []
	spearman_corr_list = []

	for result_file_path in input_path_list:
		ap_score, spearman_corr, is_bad = calculate_AP_for_story(result_file_path)
		if is_bad:
			continue
		ap_score_list.append(ap_score)
		spearman_corr_list.append(spearman_corr)

	mean_AP = sum(ap_score_list) / len(ap_score_list)
	mean_spearman_corr = sum(spearman_corr_list) / len(spearman_corr_list)

	return mean_AP, mean_spearman_corr


def main(args):
	input_path_list = glob(os.path.normpath(args.input) + "/*")
	print("Evaluation results from {} files".format(len(input_path_list)))
	calculate_F1(input_path_list, args.output, args.random)
	# print(calculate_labels(input_path_list))
	# print("Mean Average Precision (proposed): {}".format(calculate_MeanAP(input_path_list)[0]))
	
	# print("Average score for label 1: {}".format(calculate_mean_score_diff(input_path_list)[0]))
	# print("Average score for label 0: {}".format(calculate_mean_score_diff(input_path_list)[1]))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--input', '-i', type=str, help='directory path for input files')
	parser.add_argument("--output", type=str, help="name of output file")
	parser.add_argument('--random', '-r', type=str, default=False, help='Calcuate the score for random guesses of the same files.')

	args = parser.parse_args()
	main(args)