import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from task5a_hash_table import Task5aHashTable

""" Ha,b(v) = |_(r.v + b)/w_| --- (1) """
class Task5aLSH:
	def __init__(self, L_layer_count, k_hash_functions_per_layer, w_parameter = 4):
		self.L_layer_count = L_layer_count # number of layers
		self.k_hash_functions_per_layer = k_hash_functions_per_layer # number of random projections per layer
		self.feature_count = 0 # feature_count -- temporarily reading a dataset
		self.data_matrix = []
		self.data_matrix_transpose = []
		self.image_ids = list()
		self.init_data() # Initializes the data matrix and also the feature count
		self.w_parameter = w_parameter # w in (1)
		self.hash_tables = list()
		# create L hash tables with k hash functions per layer
		for value in range(self.L_layer_count):
			print('Initializing Hash Table: ', value)
			self.hash_tables.append(Task5aHashTable(self.k_hash_functions_per_layer, self.feature_count, self.w_parameter))
		
		self.fill_all_hashtables()

	"""
	Method Explanation:
		Custom implementation of the getter that returns the list of images in the same bucket
		as the given image accross all hash tables.
	Input(s):
		input_vector -- The representation of the image in the form of a vector.
	"""
	def __getitem__(self, input_vector):
		label_list = list()
		for table in self.hash_tables:
			label_list.extend(table[input_vector])
		return list(set(label_list))

	"""
	Method Explanation:
		Custom implementation of the setter that sets the value of the image represented
		by the input_vector accross all hash tables.
	Input(s):
		input_vector -- The representation of the image in the form of a vector.
		label -- the label you want to give the image when it is placed in the bucket. In this case, the ImageID.
	"""
	def __setitem__(self, input_vector, label):
		for table in self.hash_tables:
			table[input_vector] = label

	"""
	Method Explanation:
		Temporary method. Ignore.
	Input(s):
		None
	"""
	# Temporary function to test stuff. Will be removed later.
	def init_data(self):
		# Read the data
		input_df = pd.read_csv("../../dataset/visual_descriptors/acropolis_athens CM.csv", header = None)
		self.image_ids = input_df[0]

		# Delete the imageIDs
		del input_df[0]
		input_df = input_df.reset_index(drop=True)

		row_count = input_df.shape[0]
		column_count = input_df.shape[1]
		# print(self.image_ids)
		# print(input_df)

		# MinMax normalization
		minmax_scaler = MinMaxScaler()
		input_df_scaled = minmax_scaler.fit_transform(input_df)
		input_df = input_df_scaled

		# Temporary
		self.data_matrix = input_df
		self.data_matrix_transpose = input_df.transpose()
		self.feature_count = column_count

	"""
	Method Explanation:
		. Helper method for fill_all_hashtables.
		. Takes care of indexing the data for one layer.
		. Generates hashes for each data point and places them into their corresponding buckets.
	Input(s):
		table_instance -- The object representing a single layer.
	"""
	def fill_the_hashtable(self, table_instance):
		print('Filling a hash table...')
		for index, image in enumerate(self.data_matrix):
			the_label = self.image_ids[index]
			table_instance.__setitem__(image, the_label)
			# the_hash = table_instance.generate_hash(image)
			# table_instance.hash_table[the_hash] = the_label

	"""
	Method Explanation:
		. Wrapper method over fill_the_hashtable helper.
		. Takes care of indexing the data for all layers.
	Input(s):
		None
	"""
	def fill_all_hashtables(self):
		for table in self.hash_tables:
			self.fill_the_hashtable(table)
