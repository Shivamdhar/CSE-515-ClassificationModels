import constants
from data_extractor import DataExtractor
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from task5a.task5a_LSH import Task5aLSH
from util import Util

class Task5b():
	def __init__(self):
		self.ut = Util()
		self.data_extractor = DataExtractor()

	def get_indexed_image_candidates(self,image_feature_matrix,query_image_id):
		"""
		Testing function
		"""
		# sample images for test to check the bucket distribution
		L_layer_count = 3
		k_hash_size = 3
		#lsh = Task5aLSH(L_layer_count,k_hash_size,self.feature_count)

		lsh = Task5aLSH(L_layer_count,k_hash_size,image_feature_matrix,w_parameter=2,feature_count=self.feature_count)

		self.hash_tables = lsh.hash_tables

		k_count = 0
		max_key_size = -1
		# for i,table in enumerate(lsh.hash_tables):
		# 	print("For Hash table",i)
		# 	for key, val in table.hash_table.items():
		# 		if len(key) < k_hash_size:
		# 			k_count+=1
		# 		if len(key) > max_key_size:
		# 			max_key_size = len(key)
		# 		print('Key/Hash: ', key, ' Value: ', val)
		# 		print('-----------------\n')

		# print("K count",k_count)
		# print("max key size",max_key_size)

		#self.fill_all_hashtables(image_feature_matrix)

		images = [1429326778,9792811116,11733052316,1465972308,2199742801]

		print("LSH bucket images")
		#print(lsh.__getitem__(image_feature_matrix[int(images[0])]))


		lsh_images = []

		#count = 0
		for image,vector in image_feature_matrix.items():
			# if count > 5:
			# 	break
			lsh_images.append(lsh.__getitem__(vector))
			#count+=1

		# for image in images:
		# 	vector = image_feature_matrix[image]
		# 	lsh_images.append(lsh.__getitem__(vector))

		# print("LSH images",lsh_images,len(lsh_images))

		#lsh_images = lsh.__getitem__(image_feature_matrix[query_image_id])

		count = 0
		for i,v in enumerate(lsh_images):
			if len(v) == 1:
				print("index",i,v)
				count+=1
		print("Singular images",count)

		lsh_images = lsh.__getitem__(image_feature_matrix[query_image_id])
		print("Imge count",len(lsh_images))

		return lsh_images

	# def get_top_5_similar_images(self,image_feature_matrix,query_image_id):
	# 	similar_images = []

	# 	query_image_vector = image_feature_matrix[query_image_id]

	# 	for image_id,image_vector in image_feature_matrix.items():
	# 		image_feature_vector = np.array(image_vector)
	# 		sim_distance = self.ut.compute_euclidean_distance(query_image_vector,image_feature_vector)
	# 		score = 1 / (1+sim_distance)
	# 		similar_images.append((image_id,score))

	# 	return sorted(similar_images,key = lambda x:x[1],reverse=True)[:5]

	def get_top_t_similar_images(self,query_image_id,image_feature_matrix,image_candidates,t):
		similar_images = []
		query_image_vector = np.array(image_feature_matrix[query_image_id])

		for candidate in image_candidates:
			image_feature_vector = np.array(image_feature_matrix[candidate])
			sim_distance = self.ut.compute_euclidean_distance(query_image_vector,image_feature_vector)
			score = 1 / (1+sim_distance)
			similar_images.append((candidate,score))

		return sorted(similar_images,key = lambda x:x[1],reverse=True)[:t]

	def get_computed_latent_semantics(self,image_feature_semantics):
		entity_ids = list(self.image_feature_matrix.keys())
		k_semantics_map = {}
		for entity_id,value in zip(entity_ids,image_feature_semantics):
			k_semantics_map[entity_id] = value
		return k_semantics_map

	def runner(self):
		image_id = int(input("Enter the query image: "))
		t = int(input("Enter the value of t (number of similar images): "))
		#image_feature_matrix = self.get_image_features_dataset()

		image_feature_matrix = self.data_extractor.prepare_dataset_for_task5b()

		if image_id not in image_feature_matrix:
			raise ValueError(constants.IMAGE_ID_KEY_ERROR)

		#Impl SVD on image_feature_matrix
		data = np.array([value for value in image_feature_matrix.values()])
		image_feature_semantics = self.ut.dim_reduce_SVD(data,256)

		self.image_feature_matrix = image_feature_matrix

		computed_image_feature_matrix = self.get_computed_latent_semantics(image_feature_semantics)

		self.feature_count = 256

		indexed_image_candidates = self.get_indexed_image_candidates(computed_image_feature_matrix,
			int(image_id))

		#similar_images = self.get_top_5_similar_images(image_feature_matrix,image_id)

		similar_images = self.get_top_t_similar_images(image_id,computed_image_feature_matrix,
			indexed_image_candidates,t)
		print(similar_images)

if __name__== '__main__':
	runner()