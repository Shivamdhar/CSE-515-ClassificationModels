import pandas as pd
import numpy as np

# # L = input("Enter the number of layers")
# # k = input("Enter the number of hashes per layer")
# L = 3
# k = 2
class Task5aHashTable:
	"""
	Method Explanation:
		. Initializes properties for the current instance.
	Input(s):
		k_hash_size -- number of hash functions in any given layer.
		feature_count -- number of features in the dataset.
		hash_table -- the data structure for the current layer.
	"""
	def __init__(self, k_hash_size, feature_count, w_parameter = 4):
		self.k_hash_size = k_hash_size
		self.feature_count = feature_count
		self.hash_table = dict()
		self.projections = np.random.randn(self.k_hash_size, self.feature_count)
		self.w_parameter = w_parameter
		self.b_offset = np.random.uniform(0, self.w_parameter)
		
	def generate_hash(self, input_vector):
		total_hash = ''
		for row_data in self.projections:
			random_vector_transpose = row_data.transpose()
			current_hash = np.floor((np.dot(input_vector, random_vector_transpose) + self.b_offset)/self.w_parameter).astype('int')
			total_hash+= current_hash.astype('str')
		return total_hash

	def __setitem__(self, input_vector, label):
		# Generate a hash value based on random projection 
		hash_value = self.generate_hash(input_vector)
		# Get all the items from the bucket as a list and append the label to that list.
		self.hash_table[hash_value] = self.hash_table.get(hash_value, list()) + [label]

	def __getitem__(self, input_vector):
		hash_value = self.generate_hash(input_vector)
		print("hash_value",hash_value)
		return self.hash_table.get(hash_value, [])