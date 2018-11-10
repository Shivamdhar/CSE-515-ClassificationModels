"""
This module contains all functions used throughout the codebase.
"""
import numpy as np

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
		pass
