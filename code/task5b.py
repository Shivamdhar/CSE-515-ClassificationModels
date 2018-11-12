import constants
import numpy as np
import os
#from task5a import Task5a_LSH
from util import Util

class Task5b():
	def __init__(self):
		self.ut = Util()

	# def fill_all_hashtables(self,data_matrix):
	# 	"""
	# 	This method i temporary, should be directly exposed from task5a
	# 	"""
	# 	for table in self.hash_tables:
	# 		self.fill_the_hashtable(data_matrix,table)

	# def fill_the_hashtable(self,data_matrix,table_instance):
	# 	"""
	# 	This method i temporary, should be directly exposed from task5a
	# 	"""
	#     for index, image in data_matrix.values():
	# 		the_label = index
	#     	the_hash = table_instance.generate_hash(image)
	#         table_instance.hash_table[the_hash] = the_label

	def index_images(self,image_feature_matrix):
		"""
		Testing function
		"""
		# sample images for test to check the bucket distribution
		L_layer_count = 6
		k_hash_size = 6
		lsh = Task5aLSH(L_layer_count,k_hash_size,self.feature_count)

		self.hash_tables = lsh.hash_tables

		self.fill_all_hashtables(image_feature_matrix)

		images = ['1429326778','9792811116','11733052316','1465972308','2199742801']

		final_imgs = {}
		for table in self.hash_tables:
			for image in images:
				if image in final_imgs:
					final_imgs[image].append(table[image_feature_matrix[image]])
				else:
					final_imgs[image] = [table[image_feature_matrix[image]]]


	def get_image_features_dataset(self):
		visual_dir_path = constants.PROCESSED_VISUAL_DESCRIPTORS_DIR_PATH
		list_of_files = os.listdir(visual_dir_path)
		image_feature_matrix = {}
		"""
		color models options
		color_models = ["LBP3x3", "CM3x3", "GLRLM"]
		color_models = ["CN3x3", "HOG", "CM3x3","CSD"]
		color_models = ["GLRLM", "GLRLM3x3"]
		"""
		#color_models = ["CM", "CM3x3", "CN", "CN3x3", "CSD"]
		#color_models = ["LBP3x3", "CM3x3", "GLRLM"]
		#color_models = ["GLRLM", "GLRLM3x3"]
		color_models = ["CN3x3", "HOG", "CM3x3","CSD"]
		#color_models = ["CN3x3", "HOG", "CM3x3","CSD","LBP"]
		#color_models = ["CN3x3", "HOG", "CM3x3"]
		#color_models = ["CM", "CM3x3", "CN", "CN3x3", "CSD"]
		#color_models = constants.MODELS
		for filename in list_of_files:
			model = filename.split(" ")[1].replace(".csv","")
			if model in color_models:
				with open(os.path.join(visual_dir_path,filename)) as file:
					for row in file:
						row_data = row.strip().split(",")
						feature_values = list(map(float, row_data[1:]))
						image_id = row_data[0]
						if image_id in image_feature_matrix:
							image_feature_matrix[image_id] += feature_values
						else:
							image_feature_matrix[image_id] = feature_values
		return image_feature_matrix

	def get_top_5_similar_images(self,image_feature_matrix,query_image_id):
		similar_images = []

		query_image_vector = image_feature_matrix[query_image_id]

		for image_id,image_vector in image_feature_matrix.items():
			image_feature_vector = np.array(image_vector)
			sim_distance = self.ut.compute_euclidean_distance(query_image_vector,image_feature_vector)
			score = 1 / (1+sim_distance)
			similar_images.append((image_id,score))

		return sorted(similar_images,key = lambda x:x[1],reverse=True)[:5]

	def runner(self):
		image_id = input("Enter the query image")
		t = int(input("Enter the value of t (number of similar images): "))
		image_feature_matrix = self.get_image_features_dataset()

		if image_id not in image_feature_matrix:
			raise ValueError(constants.IMAGE_ID_KEY_ERROR)

		self.feature_count = len(image_feature_matrix[image_id])

		similar_images = self.get_top_5_similar_images(image_feature_matrix,image_id)
		print(similar_images)

		'''
		Asesss the similar images to analyse the visual model combination 
		'''