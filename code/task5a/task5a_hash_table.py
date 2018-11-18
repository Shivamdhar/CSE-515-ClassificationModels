import numpy as np

class Task5aHashTable:
	"""
	Method Explanation:
		. Initializes properties for the current instance.
	Input(s):
		k_hash_size -- number of hash functions in any given layer.
		feature_count -- number of features in the dataset.
		hash_table -- the data structure for the current layer.
	"""
	def __init__(self, k_hash_size, feature_count, w_parameter = 12):
		self.k_hash_size = k_hash_size
		self.feature_count = feature_count
		self.hash_table = dict()
		self.projections = self.init_projections()
		#print('Projections: ', self.projections)
		self.w_parameter = w_parameter
		self.b_offsets = self.init_b_offsets() # Initialize 'k' number of random shifts
		# self.b_offset = np.random.uniform(0, self.w_parameter)

	"""
	Method Explanation:
		. Initializes the projections matrix.
		. Takes the values from a normal distribution and scales them to unit norm.
	"""
	def init_projections(self):
		the_projections = np.random.randn(self.k_hash_size, self.feature_count)
		for index, row_data in enumerate(the_projections):
			the_norm = np.linalg.norm(row_data)
			the_projections[index] = np.true_divide(row_data, the_norm)
		
		return the_projections

	"""
	Method Explanation:
		. Initializes 'k' number of b_offsets sampled uniformly between 0 and w_parameter of the instance.
	"""
	def init_b_offsets(self):
		to_return = list()
		for index in range(self.k_hash_size):
			to_return.append(np.random.uniform(0, self.w_parameter))
		return to_return
	
	"""
	Method Explanation:
		. Generate a hash value based on the euclidean hash family formula.
		. Each hash function generates a hash value of 8 bits long.
		. For k hash functions, we get a hash code of 8*k bits.
	Input(s):
		input_vector -- The image represented as a vector.
	Output:
		The bit representation of the hash code that is generated cast to an integer representation comprising of 0s and 1s.
	"""
	def generate_hash(self, input_vector):
		hash_code = ''
		for index, row_data in enumerate(self.projections):
			random_vector_transpose = row_data.transpose()
			current_hash = np.floor((np.dot(input_vector, random_vector_transpose) + self.b_offsets[index])/self.w_parameter).astype('int')
			bit_representation = np.binary_repr(current_hash, 8) # "{:08b}".format(current_hash & 0xffffffff)
			hash_code+= bit_representation
		return hash_code

	"""
	Method Explanation:
		Custom setter for the layer.
	"""
	def __setitem__(self, input_vector, label):
		# Generate a hash value based on random projection 
		hash_value = self.generate_hash(input_vector)
		# Get all the items from the bucket as a list and append the label to that list.
		self.hash_table[hash_value] = self.hash_table.get(hash_value, list()) + [label]

	"""
	Method Explanation:
		Custom Getter for the layer.
	"""
	def __getitem__(self, input_vector):
		hash_value = self.generate_hash(input_vector)
		return self.hash_table.get(hash_value, [])

	"""
	Method Explanation:
		. A helper to get_item_for_reduced_k()
		. Gives a reduced representation of the hash with 8, 16, 24... bits subtracted from the end depending on the k_value
	Input(s):
		current_hash_code -- the current hash_code of size 8*self.k_hash_size
		k_value -- a value lesser than self.k_hash_size
	Output:
		A reduced representation of the current_hash_code.
	"""
	def get_reduced_hash_code(self, current_hash_code, k_value):
		if (k_value == self.k_hash_size) or (k_value == 0) or (self.k_hash_size - k_value < 0):
			return current_hash_code
		return current_hash_code[:(len(current_hash_code)-8*(self.k_hash_size - k_value))]
	
	"""
	Method Explanation:
		. A helper for nearest neighbor search where if enough candidates are not retrieved for the current k_value, we reduce k iteratively
		  to get all candidate nearest neighbors.
	Input(s):
		input_vector -- The query image represented as a vector.
		k_value -- A value of the number of hash functions that is lesser than self.k_hash_size
	Output(s):
	"""
	def get_item_for_reduced_k(self, input_vector, k_value):
		result_list = list()
		hash_code_key = self.generate_hash(input_vector)
		reduced_hash_code_key = self.get_reduced_hash_code(hash_code_key, k_value) # hash_code_key[:(len(hash_code_key)-8*(self.k_hash_size - k_value))]
		#print("reduced_hash_code_key",reduced_hash_code_key)

		for hash_code, imageid_list in self.hash_table.items():
			reduced_hash_code = self.get_reduced_hash_code(hash_code, k_value)
			#print("reduced_hash_code",reduced_hash_code)
			if reduced_hash_code_key == reduced_hash_code:
				result_list.extend(imageid_list)
		
		return result_list
