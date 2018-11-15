import constants
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from task5a.task5a_LSH import Task5aLSH
from util import Util

class Task5b():
	def __init__(self):
		self.ut = Util()

	def get_indexed_image_candidates(self,image_feature_matrix,query_image_id):
		"""
		Testing function
		"""
		# sample images for test to check the bucket distribution
		L_layer_count = 3
		k_hash_size = 2
		#lsh = Task5aLSH(L_layer_count,k_hash_size,self.feature_count)

		lsh = Task5aLSH(L_layer_count,k_hash_size,image_feature_matrix,w_parameter=2,feature_count=self.feature_count)

		self.hash_tables = lsh.hash_tables

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

		# final_imgs = {}
		# for table in self.hash_tables:
		# 	for image in images:
		# 		if image in final_imgs:
		# 			final_imgs[image].append(table[image_feature_matrix[image]])
		# 		else:
		# 			final_imgs[image] = [table[image_feature_matrix[image]]]


	# def get_image_features_dataset(self):
	# 	visual_dir_path = constants.PROCESSED_VISUAL_DESCRIPTORS_DIR_PATH
	# 	list_of_files = os.listdir(visual_dir_path)
	# 	image_feature_matrix = {}
	# 	"""
	# 	color models options
	# 	"""
	# 	#color_models = ["LBP3x3", "CM3x3", "GLRLM"]
	# 	#color_models = ["CN3x3", "HOG", "CM3x3","CSD"] #last current
	# 	#color_models = ["GLRLM", "GLRLM3x3"]
	# 	#color_models = ["CM", "CM3x3", "CN", "CN3x3", "CSD"]
	# 	#color_models = ["CN3x3", "HOG", "CM3x3","CSD","LBP"]
	# 	#color_models = ["CN3x3", "HOG", "CM3x3"]
	# 	#color_models = ["CM", "CM3x3", "CN", "CN3x3", "CSD"]
	# 	color_models = constants.MODELS
	# 	for filename in list_of_files:
	# 		model = filename.split(" ")[1].replace(".csv","")
	# 		if model in color_models:
	# 			with open(os.path.join(visual_dir_path,filename)) as file:
	# 				for row in file:
	# 					row_data = row.strip().split(",")
	# 					feature_values = np.array(list(map(float, row_data[1:])))
	# 					feature_values = list(np.interp(feature_values, (feature_values.min(), feature_values.max()), (0, 1)))
	# 					image_id = int(row_data[0])
	# 					if image_id in image_feature_matrix:
	# 						image_feature_matrix[image_id] += feature_values
	# 					else:
	# 						image_feature_matrix[image_id] = feature_values
	# 	return image_feature_matrix

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


	def preprocess_image_dataset(self):
		"""
		1) Combine all models and apply min max to entire combined dataset of a given model
		2) Construct a map of keys as image id and combine all the features from each model.
		"""
		visual_dir_path = constants.PROCESSED_VISUAL_DESCRIPTORS_DIR_PATH
		list_of_files = os.listdir(visual_dir_path)
		"""
		color models options
		"""
		#color_models = ["LBP3x3", "CM3x3", "GLRLM"]
		#color_models = ["CN3x3", "HOG", "CM3x3","CSD"] #last current
		#color_models = ["GLRLM", "GLRLM3x3"]
		#color_models = ["CM", "CM3x3", "CN", "CN3x3", "CSD"]
		#color_models = ["CN3x3", "HOG", "CM3x3","CSD","LBP"]
		#color_models = ["CN3x3", "HOG", "CM3x3"]
		#color_models = ["CM", "CM3x3", "CN", "CN3x3", "CSD"]
		color_models = constants.MODELS
		model_map = {}
		for filename in list_of_files:
			model_name = filename.split(" ")[1].replace(".csv","")
			df = pd.read_csv(os.path.join(visual_dir_path,filename), header = None)
			if model_name in model_map:
				#concat the dataframe with earlier df (prev location)
				pass
				model_map[model_name] += [df]
			else:
				model_map[model_name] = [df]

		image_ids = []
		for k in model_map.keys():
			model_map[k] = pd.concat(model_map[k])
			if image_ids == []:
				image_ids = model_map[k].iloc[:,0].tolist()

		final_features = []
		for df in model_map.values():
			# min scale each df for a given model and store in the list
			#image_ids = df.iloc[:,0].to_frame
			features = df.iloc[:,1:]
			minmax_scaler = MinMaxScaler()
			features_scaled = minmax_scaler.fit_transform(features)
			final_features.append(features_scaled)

		combined_features = np.concatenate(final_features,axis=1)

		minmax_scaler = MinMaxScaler()
		combined_features = minmax_scaler.fit_transform(combined_features)

		image_feature_matrix = {}

		# Representing image ids as integer
		image_ids = list(map(int,image_ids))

		for image_id,vector in zip(image_ids,combined_features):
			image_feature_matrix[image_id] = vector

		return image_feature_matrix

	def runner(self):
		image_id = int(input("Enter the query image: "))
		t = int(input("Enter the value of t (number of similar images): "))
		#image_feature_matrix = self.get_image_features_dataset()

		image_feature_matrix = self.preprocess_image_dataset()

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