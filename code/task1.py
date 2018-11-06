"""
This module is the program for task 1.
"""
import constants
from data_extractor import DataExtractor
import numpy as np
from util import Util

class Task1():
	def __init__(self):
		self.ut = Util()
		self.data_extractor = DataExtractor()
		self.mapping = self.data_extractor.location_mapping()

	def generate_imgximg_edgelist(self, image_feature_map):
		""" Method: generate_imgximg_edgelist returns image to image similarity in form of an edge list """
		imgximg_edgelist = []
		image_list = list(image_feature_map.keys())
		for index1 in range(0, len(image_list)):
			local_edgelist = []
			for index2 in range(0, len(image_list)):
				image1 = image_list[index1]
				image2 = image_list[index2]
				features_image1 = image_feature_map[image1]
				features_image2 = image_feature_map[image2]
				score = self.compute_similarity(features_image1, features_image2)
				local_edgelist.append((image1, image2, score))
			imgximg_edgelist.append(local_edgelist)

		return imgximg_edgelist

	def calculate_similarity(self, features_image1, features_image2):
		""" Method: image-image similarity computation"""
		return self.ut.compute_euclidean_distance(np.array(features_image1), np.array(features_image2))

	def generate_top_k_edgelist(self, imgximg_edgelist, k):
		""" returns top k edgelists for each image """
		top_k_edgelist = []
		for image_edge_list in imgximg_edgelist:
			image_edge_list.sort(key=lambda value: value[2])
			top_k_edgelist.append(image_edge_list[0:k])
		return top_k_edgelist

	def pretty_print(self, top_k_edgelist):
		pass

	def runner(self):
		"""
		Method: runner implemented for all the tasks, takes user input, and prints desired results.
		"""
		try:
			k = int(input("Enter the value of k: "))
			image_feature_map = self.data_extractor.prepare_dataset_for_task1(self.mapping)
			imgximg_edgelist = self.generate_imgximg_edgelist(image_feature_map)
			top_k_edgelist = self.generate_top_k_edgelist(imgximg_edgelist, k)
			self.pretty_print(top_k_edgelist)

		except Exception as e:
			print(constants.GENERIC_EXCEPTION_MESSAGE + "," + str(type(e)) + "::" + str(e.args))