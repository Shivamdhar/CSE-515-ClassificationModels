"""
This module is the program for task 1.
"""
import constants
from data_extractor import DataExtractor
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pickle
from util import Util

class Task1():
	def __init__(self):
		self.ut = Util()
		self.data_extractor = DataExtractor()
		self.mapping = self.data_extractor.location_mapping()

	def generate_imgximg_edgelist(self, image_list1, image_list2, image_feature_map):
		""" Method: generate_imgximg_edgelist returns image to image similarity in form of an edge list """
		imgximg_edgelist = []
		for index1 in range(0, len(image_list1)):
			print("index1 : ", index1)
			local_edgelist = []
			for index2 in range(0, len(image_list2)):
				image1 = image_list1[index1]
				image2 = image_list2[index2]
				features_image1 = image_feature_map[image1]
				features_image2 = image_feature_map[image2]
				score = 1 / (1 + self.calculate_similarity(features_image1, features_image2))
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
			image_edge_list.sort(key=lambda value: value[2], reverse = True)
			top_k_edgelist.append(image_edge_list[0:k])
		return top_k_edgelist

	def pretty_print(self, top_k_edgelist, k):
		self.print_to_terminal(top_k_edgelist, k)
		self.print_to_file(top_k_edgelist)

	def print_to_terminal(self, top_k_edgelist, k):
		for iter1 in range(len(top_k_edgelist)):
			for iter2 in range(len(top_k_edgelist[int(k)-1])):
				image_id = top_k_edgelist[iter1][iter2][0]
		print("Top " + str(k) + " images of image ID " + str(image_id) + " are:")
		print(top_k_edgelist[iter1][:k])

	def print_to_file(self, top_k_edgelist):
		merged_image_list = []
		for row in top_k_edgelist:
			merged_image_list.extend(row)

		graph_file = open(constants.VISUALIZATIONS_DIR_PATH + "graph_file.txt", "w")
		for iter in merged_image_list:
			graph_file.write(str(iter[0]) + " " + str(iter[1]) + " " + str(iter[2]) + "\n")

	def visualise_graph(self):
		g = nx.read_edgelist(constants.VISUALIZATIONS_DIR_PATH + "graph_file.txt", nodetype=int, \
							data=(('weight',float),), create_using=nx.DiGraph())
		print("graph created")
		# nx.draw(g, node_color="g", node_size=1)
		# plt.show()

	def runner(self):
		"""
		Method: runner implemented for all the tasks, takes user input, and prints desired results.
		"""
		k = int(input("Enter the value of k: "))
		try:
			task1_pkl_file = open(constants.DUMPED_OBJECTS_DIR_PATH + "task1_img.pickle", "rb")
			""" when objects are dumped into pickle multiple times, we need to load the pickle multiple\
				times to fetch all the objects """
			combined_edgelist = []
			for iter in range(0,9):
				combined_edgelist += pickle.load(task1_pkl_file)[1][0]

			top_k_edgelist = []
			top_k_edgelist += self.generate_top_k_edgelist(combined_edgelist, k)
			self.pretty_print(top_k_edgelist, k)
			# self.visualise_graph()

		except(OSError, IOError) as e:
			image_feature_map = self.data_extractor.prepare_dataset_for_task1(self.mapping)
			image_list = list(image_feature_map.keys())
			# for dev set we'll have 4 bins
			task1_pkl_file = open(constants.DUMPED_OBJECTS_DIR_PATH + "task1_img.pickle", "wb")
			imgximg_edgelist = []

			for iter in range(0, len(image_list), 1000):
				imgximg_edgelist = self.generate_imgximg_edgelist(image_list[iter:iter + 1000], image_list, image_feature_map)
				pickle.dump(["Object"+ str(iter), imgximg_edgelist], task1_pkl_file)

			# imgximg_edgelist, label_dict = self.generate_imgximg_edgelist(image_feature_map)
			task1_pkl_file.close()
			task1_pkl_file = open(constants.DUMPED_OBJECTS_DIR_PATH + "task1_img.pickle", "rb")
			objects = pickle.load(task1_pkl_file)
			top_k_edgelist = []

			for iter in range(0, len(objects)):
				top_k_edgelist += self.generate_top_k_edgelist(objects[iter][1], k)
			task1_pkl_file.close()

			self.pretty_print(top_k_edgelist, k)
			self.visualise_graph()

		except Exception as e:
			print(constants.GENERIC_EXCEPTION_MESSAGE + "," + str(type(e)) + "::" + str(e.args))