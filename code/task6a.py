from util import Util
from collections import OrderedDict
import numpy as np
import pandas as pd
import constants
from data_extractor import DataExtractor
import pickle

class Task6a():
	def __init__(self):
		self.ut = Util()
		self.img_ids = []
		self.adj_matrix = []
		self.img_feature_matrix = []
		self.label_img_matrix = dict()
		self.input_image_label_pairs = OrderedDict()

	def generate_img_img_adj_matrix(self):
		""" Method: generate image-image similarity matrix and stash in pickle file"""
		print("computing adj matrix...")
		data_extractor = DataExtractor()
		loc_mapping = data_extractor.location_mapping()
		self.img_feature_matrix = data_extractor.prepare_dataset_for_task1(loc_mapping)
		self.img_ids = list(self.img_feature_matrix.keys())
		# for id1 in self.img_ids:
		# 	img_matrix = []
		# 	for id2 in self.img_ids:
		# 		img_matrix.append(self.get_euclidean_similarity(self.img_feature_matrix[id1], self.img_feature_matrix[id2]))
		# 	self.adj_matrix.append(img_matrix)
		# print('pickling the adj matrix...')
		# with open('adj_mat_6a.pickle', 'wb') as file_handle:
		# 	pickle.dump(self.adj_matrix, file_handle, protocol=pickle.HIGHEST_PROTOCOL)
		# print(np.shape(np.asarray(adj_matrix)))

	def get_img_img_adj_matrix(self):
		""" Method: read image-image similarity matrix from pickle file"""
		with open('adj_mat_6a.pickle', 'rb') as file_handle:
			self.adj_matrix = pickle.load(file_handle)

	def get_euclidean_similarity(self, features_image1, features_image2):
		""" Method: image-image similarity computation"""
		return self.ut.compute_euclidean_distance(np.array(features_image1), np.array(features_image2))

	def read_input_labels_pairs(self):
		""" Method: read input image-label pairs"""
		input_image_label_pairs_df = pd.read_csv('../input/task_6a.txt')
		input_images = list(input_image_label_pairs_df['image'])
		input_labels = list(input_image_label_pairs_df['label'])
		self.input_image_label_pairs = OrderedDict(zip(input_images, input_labels))
		for i in range(0, len(input_labels)):
			if input_labels[i] in self.label_img_matrix:
				self.label_img_matrix[input_labels[i]].append(input_images[i])
			else:
				self.label_img_matrix[input_labels[i]] = [input_images[i]]
		print('initial fianl matrix ', self.label_img_matrix)

	def classify_images(self):
		""" Method: Classify all images based on given image-label pairs"""
		labelled_image_ids = list(self.input_image_label_pairs.keys())
		input_image_labels = list(self.input_image_label_pairs.values())
		for image in self.img_ids:
			label_similarity_dict = dict()
			for i, labelled_image in enumerate(labelled_image_ids):
				if input_image_labels[i] in label_similarity_dict:
					label_similarity_dict[input_image_labels[i]] += self.get_euclidean_similarity(self.img_feature_matrix[image], self.img_feature_matrix[str(labelled_image)])
				else:
					label_similarity_dict[input_image_labels[i]] = self.get_euclidean_similarity(self.img_feature_matrix[image], self.img_feature_matrix[str(labelled_image)])
			for label in label_similarity_dict:
				label_similarity_dict[label] = label_similarity_dict[label]/input_image_labels.count(label)
			min_sim = min(label_similarity_dict.values())
			for label, sim in label_similarity_dict.items():
				if sim == min_sim:
					self.label_img_matrix[label].append(image)

	def pretty_print(self):
		op = open(constants.TASK6a_OUTPUT_FILE, "w")

		for label, image_ids in self.label_img_matrix.items():
			op.write("Label " + label + "\n")
			for temp in image_ids:
				op.write(str(temp) + "\n")
			op.write("####\n")

			# # similarity_image_dict = Dict(zip(similarity_matrix, input_image_labels))
			# image_similarity_dict = Dict(zip(input_image_labels, similarity_matrix))
			# label_sum_dict = dict()
			# min_sum = 999999999999
			# min_label = ''
			# for label in image_similarity_dict.keys():
			# 	sum_val = sum(image_similarity_dict[label])
			# 	label_sum_dict[label] = sum_val
			# 	if min_sum > sum_val:
			# 		min_sum = sum_val
			# 		min_label = label
			# 	else:


		pass

		
if __name__ == "__main__":
	task = Task6a()
	task.generate_img_img_adj_matrix()
	if '40222616' in task.img_feature_matrix:
		print('found')
	else:
		print('not found')
	# task.get_img_img_adj_matrix()

	# test this code..
	task.read_input_labels_pairs()
	task.classify_images()
	task.pretty_print()
