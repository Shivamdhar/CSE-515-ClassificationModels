"""
This module contains data parsing methods.
"""
from collections import OrderedDict
import constants
from functools import reduce
import numpy as np
import os
import operator
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import xml.etree.ElementTree as et

class DataExtractor(object): 
	def location_mapping(self):
		#parse the xml file of the locations
		tree = et.parse(constants.DEVSET_TOPICS_DIR_PATH)
		#get the root tag of the xml file
		doc = tree.getroot()
		mapping = OrderedDict({})
		#map the location id(number) with the location name
		for topic in doc:
			mapping[topic.find("number").text] = topic.find("title").text

		return mapping

	def prepare_dataset_for_task1(self, mapping):
		"""
		Method: Combining all the images across locations.
		"""
		locations = list(mapping.values())
		image_feature_map = OrderedDict({})
		model = constants.VISUAL_DESCRIPTOR_MODEL_FOR_GRAPH_CREATION

		for location in locations:
			location_model_file = location + " " + model + ".csv"
			data = open(constants.PROCESSED_VISUAL_DESCRIPTORS_DIR_PATH + location_model_file, "r").readlines()

			for row in data:
				row_data = row.strip().split(",")
				feature_values = list(map(float, row_data[1:]))
				image_id = row_data[0]
				image_feature_map[image_id] = feature_values

		return image_feature_map


	def prepare_dataset_for_task5b(self):
		"""
		1) Combine all models and apply min max to entire combined dataset of a given model
		2) Construct a map of keys as image id and combine all the features from each model.
		"""
		visual_dir_path = constants.PROCESSED_VISUAL_DESCRIPTORS_DIR_PATH
		list_of_files = os.listdir(visual_dir_path)
		list_of_files.sort()
		"""
		color models options
		"""
		#color_models = ["LBP3x3", "CM3x3", "GLRLM"]
		color_models = ["CN3x3", "HOG", "CM3x3","CSD"] #last current
		#color_models = ["GLRLM", "GLRLM3x3"]
		#color_models = ["CM", "CM3x3", "CN", "CN3x3", "CSD"]
		#color_models = ["CN3x3", "HOG", "CM3x3","CSD","LBP"]
		#color_models = ["CN3x3", "HOG", "CM3x3"]
		#color_models = ["CM", "CM3x3", "CN", "CN3x3", "CSD"]
		#color_models = constants.MODELS
		model_map = {}
		image_ids = []
		for filename in list_of_files:
			model_name = filename.split(" ")[1].replace(".csv","")
			df = pd.read_csv(os.path.join(visual_dir_path,filename), header = None)
			image_ids = df.iloc[:,0].tolist()
			features = df.iloc[:,1:]
			minmax_scaler = MinMaxScaler()
			features_scaled = minmax_scaler.fit_transform(features)


			if model_name in model_map:
				#concat the dataframe with earlier df (prev location)
				model_map[model_name]['features'] += [pd.DataFrame(features_scaled)]
				if image_ids not in model_map[model_name]['image_ids']:
					model_map[model_name]['image_ids'].append(image_ids)
			else:
				#image_ids += df.iloc[:,0].tolist()
				model_map[model_name] = {'features' : [pd.DataFrame(features_scaled)],'image_ids':[image_ids]}
# =======
# 			if model_name in color_models:
# 				df = pd.read_csv(os.path.join(visual_dir_path,filename), header = None)
# 				image_ids = df.iloc[:,0].tolist()
# 				features = df.iloc[:,1:]
# 				# minmax_scaler = MinMaxScaler()
# 				# features_scaled = minmax_scaler.fit_transform(features)


# 				if model_name in model_map:
# 					#concat the dataframe with earlier df (prev location)
# 					model_map[model_name]['features'] += [features]
# 					#model_map[model_name]['features'] += [pd.DataFrame(features_scaled)]
# 					if image_ids not in model_map[model_name]['image_ids']:
# 						model_map[model_name]['image_ids'].append(image_ids)
# 				else:
# 					#image_ids += df.iloc[:,0].tolist()
# 					model_map[model_name] = {'features' : [features],'image_ids':[image_ids]}
# 					#model_map[model_name] = {'features' : [pd.DataFrame(features_scaled)],'image_ids':[image_ids]}
# >>>>>>> Stashed changes

		image_ids = []
		for k in model_map.keys():
			model_map[k]['features'] = pd.concat(model_map[k]['features'])
			if image_ids == []:
				image_ids = reduce(operator.concat, model_map[k]['image_ids'])

			#image_ids += model_map[k]['image_ids']
				#image_ids = model_map[k].iloc[:,0].tolist()

		final_features = []
		for df in model_map.values():
			# min scale each df for a given model and store in the list
			#image_ids = df.iloc[:,0].to_frame
			#features = df.iloc[:,1:]
			minmax_scaler = MinMaxScaler()
			features_scaled = minmax_scaler.fit_transform(df['features'])
			final_features.append(features_scaled)

		combined_features = np.concatenate(final_features,axis=1)

		minmax_scaler = MinMaxScaler()
		combined_features = minmax_scaler.fit_transform(combined_features)

		# image_df = pd.DataFrame.from_records(np.array(image_ids))

		# feature_frame = pd.DataFrame.from_records(combined_features)

		# print(np.array(image_ids).shape)
		# print(combined_features.shape)

		# final_frame = np.concatenate([np.array(image_ids),combined_features],axis=1)

		np.savetxt('final_process_file.txt', combined_features, delimiter = ',')

		# final_frame.to_csv('final_process_file.csv', sep = ',', \
		# 	encoding = 'utf-8', index = False, header = False)

		image_feature_matrix = {}

		# Representing image ids as integer
		image_ids = list((map(int,image_ids)))

		np.savetxt('final_image_ids.txt', image_ids, fmt="%i")

		for image_id,vector in zip(image_ids,combined_features):
			image_feature_matrix[image_id] = vector

		return image_feature_matrix