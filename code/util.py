"""
This module contains all functions used throughout the codebase.
"""
import constants
import numpy as np
import pickle
from sklearn.decomposition import TruncatedSVD

class Util():

	def __init__(self):
		pass

	""" Returns the euclidean distance between vector_one and vetor_two """
	def compute_euclidean_distance(self, vector_one, vector_two):
		return np.linalg.norm(vector_one - vector_two)

	def fetch_dict_graph(self):
		"""
		graph returned in this format -
		graph = [{(1,2): 0.8, (1,3): 0.7, ....},
				{(2,1): 0.8, (2,3): 0.75, ...}]
		"""
		graph_dict_file = open(constants.DUMPED_OBJECTS_DIR_PATH + "entire_graph_dict.pickle", "rb")
		objects = pickle.load(graph_dict_file)

		return objects[1]

	def create_adj_mat_from_red_file(self, initial_k, similarity=False):

		image_id_mapping_file = open(constants.DUMPED_OBJECTS_DIR_PATH + "image_id_mapping.pickle", "rb")
		image_id_mapping = pickle.load(image_id_mapping_file)[1]

		graph_file = open(constants.GRAPH_FILE, "r")
		edges = graph_file.readlines()

		graph_file_len = len(edges)
		size_of_graph = graph_file_len // initial_k
		adj_matrix = []
		image1 = ""
		for line in edges:
			temp = line.split(" ")
			if image1 != image_id_mapping[temp[0]]:
				adj_matrix.append([0]*size_of_graph)
				image1 = image_id_mapping[temp[0]]
			else:
				img2 = image_id_mapping[temp[1]]
				if(similarity):
					adj_matrix[-1][img2] = float(temp[2])
				else:
					adj_matrix[-1][img2] = 1

		return adj_matrix

	def image_id_mapping(self):
		image_id_mapping_file = open(constants.DUMPED_OBJECTS_DIR_PATH + "image_id_mapping.pickle", "rb")
		return pickle.load(image_id_mapping_file)[1]

	def dim_reduce_SVD(self, input_arr, k):
		svd = TruncatedSVD(n_components=int(k))
		svd.fit(input_arr)

		return(svd.transform(input_arr))
