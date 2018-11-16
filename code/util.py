"""
This module contains all functions used throughout the codebase.
"""
import constants
import numpy as np
import pickle

class Util():

	def __init__(self):
		pass

	""" Returns the euclidean distance between vector_one and vetor_two """
	def compute_euclidean_distance(self, vector_one, vector_two):
		return np.linalg.norm(vector_one - vector_two)

	def fetch_imgximg_graph(self):
		"""
		1. load pickle file and return the graph stored.
		"""
		pass

	def fetch_dict_graph(self):
		"""
		graph returned in this format -
		graph = [{(1,2): 0.8, (1,3): 0.7, ....},
				{(2,1): 0.8, (2,3): 0.75, ...}]
		"""
		graph_dict_file = open(constants.DUMPED_OBJECTS_DIR_PATH + "graph_dict.pickle", "rb")
		objects = pickle.load(graph_dict_file)

		return objects[1]

	def fetch_adjacency_matrix(self):
		graph_dict_file = open(constants.DUMPED_OBJECTS_DIR_PATH + "graph_dict.pickle", "rb")
		objects = pickle.load(graph_dict_file)

		import pdb
		# pdb.set_trace()
		graph = objects[1]
		adj_matrix = [[0]*len(graph)]*len(graph)
		for image_row in graph:
			for edge, score in image_row.items():
				adj_matrix[edge[0]][edge[1]] = 1

		return adj_matrix

	def create_adj_mat_from_red_file(self, initial_k):

		image_id_mapping_file = open(constants.DUMPED_OBJECTS_DIR_PATH + "image_id_mapping.pickle", "rb")
		image_id_mapping = pickle.load(image_id_mapping_file)[1]

		graph_file = open(constants.GRAPH_FILE, "r")
		edges = graph_file.readlines()

		graph_file_len = len(edges)
		size_of_graph = graph_file_len // initial_k
		adj_matrix = [[0] * size_of_graph] * size_of_graph

		for line in edges:
			temp = line.split(" ")
			if temp[0] == temp[1]:
				continue
			adj_matrix[image_id_mapping[temp[0]]][image_id_mapping[temp[1]]] = 1

		return adj_matrix

	def image_id_mapping(self):
		image_id_mapping_file = open(constants.DUMPED_OBJECTS_DIR_PATH + "image_id_mapping.pickle", "rb")
		return pickle.load(image_id_mapping_file)[1]

	def create_sim_adj_mat_from_red_file(self, initial_k):
		image_id_mapping_file = open(constants.DUMPED_OBJECTS_DIR_PATH + "image_id_mapping.pickle", "rb")
		image_id_mapping = pickle.load(image_id_mapping_file)[1]

		graph_file = open(constants.GRAPH_FILE, "r")
		edges = graph_file.readlines()

		graph_file_len = len(edges)
		size_of_graph = graph_file_len // initial_k
		adj_matrix = [[0] * size_of_graph] * size_of_graph

		for line in edges:
			temp = line.split(" ")
			if temp[0] == temp[1]:
				continue
			adj_matrix[image_id_mapping[temp[0]]][image_id_mapping[temp[1]]] = temp[2]

		return adj_matrix