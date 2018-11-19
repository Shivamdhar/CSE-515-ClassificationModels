"""
ALGORITHM FOR PREPROCESSING:
---------------------------

reference_model = 'CM'
CM_df = pd.DataFrame()
df_list = list(model_count-1) # since we are not considering CM for it
data_dict = dict()

. for every location:
	. df = location's CM model file
	. df = MinMaxScaler(df) OR StandardScaler(df)
	. CM_df = CM_df.append(df)
  . Remove duplicate Image IDs from the CM_df

. Build the data_dict where key = image_ids of the CM_df and values are a list of features of the images from the CM model.

. for every index, model other than CM:
	. for every location:
		. df = location's <model> model file
		. df = MinMaxScaler(df) OR StandardScaler(df)
		. df_list[index] = df_list[index].append (df)
	. Remove duplicate ImageIDs from the df_list[index] df
	. Go over the df_list[index] and for each image_id, if and only if it already exists in the data_dict, append the feature values of the current row to the values of the key in the data_dict
	  This ensures duplicates aren't inserted accross locations.
"""

from data_extractor import DataExtractor
import constants
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

pd.options.mode.chained_assignment = None  # default='warn'

class Task5PreProcessor:

	def __init__(self):
		self.data_extractor = DataExtractor()
		self.mapping = self.data_extractor.location_mapping()
		self.location_names = list(self.mapping.values())
		self.reference_model = 'CM'
		self.model_list = self.init_model_list()
		self.reference_df = pd.DataFrame()
		self.df_list = self.init_df_list()
		self.data_dict = dict()
		self.minmax_scaler = MinMaxScaler()
		self.standard_scaler = StandardScaler()

	def init_model_list(self):
		"""
		Method Explanation:
			. Initializes the model_list as every model name other than the reference model.
		"""
		models = constants.MODELS
		if self.reference_model in models:
			models.remove(self.reference_model)
		return models
	
	def init_df_list(self):
		"""
		Method Explanation:
			. Initializes the df_list comprising of the dataframes.
		"""
		to_return = list()
		for model in self.model_list:
			to_return.append(pd.DataFrame())
		return to_return

	def compute_latent_semantics(self, feature_matrix, k):
		"""
		Method Explanation:
			. Returns the latent semantic representation of the feature_matrix with 'k' concepts.
		Input(s):
			feature_matrix -- the list of all features of all image IDs on top of which SVD would be done.
			k -- an integer representing the number of concepts desired. (Can be chosen based on analysis of
			     eigen values as well, but currently it's driven by an input of k)
		Output:
			The concept mapping of the feature_matrix in 'k' dimensions/concepts.
		"""
		svd = TruncatedSVD(n_components=k)
		svd.fit(feature_matrix)

		return (svd.transform(feature_matrix))

	def preprocess_StandardScaler(self):
		"""
		Method Explanation:
			. Refer to the top of the file for the algorithm for preprocessing.
			. Uses the StandardScaling for the standardization of datapoints with mean=0 and variance=1.
		"""
		self.data_dict.clear()

		for location in self.location_names:
			current_df = pd.read_csv("../dataset/visual_descriptors/" + location + " " + self.reference_model + ".csv", header = None)
			self.reference_df = self.reference_df.append(current_df, ignore_index = True)

		self.reference_df = self.reference_df.drop_duplicates(subset=[0], keep = 'first') # drop duplicate image ID rows and keep the first one.
		columns_to_normalize = np.arange(1, self.reference_df.shape[1], 1) # the column indices to which Standardization will be applied to.
		self.reference_df[columns_to_normalize] = self.standard_scaler.fit_transform(self.reference_df[columns_to_normalize]) # MinMax normalization

		self.data_dict = self.reference_df.set_index(0).T.to_dict('list') # Filling the data dict

		temp_dict = dict()
		for index, model in enumerate(self.model_list):
			for location in self.location_names:
				current_df = pd.read_csv("../dataset/visual_descriptors/" + location + " " + model + ".csv", header = None)
				df_to_modify = self.df_list[index]
				df_to_modify = df_to_modify.append(current_df, ignore_index = True)
				self.df_list[index] = df_to_modify
			model_df = self.df_list[index]

			model_df = model_df.drop_duplicates(subset=[0], keep='first') # drop duplicate image ID rows and keep the first one.
			columns_to_normalize = np.arange(1, model_df.shape[1], 1) # the column indices to which Standardization will be applied to.
			model_df[columns_to_normalize] = self.standard_scaler.fit_transform(model_df[columns_to_normalize])
			self.df_list[index] = model_df

			temp_dict = self.df_list[index].set_index(0).T.to_dict('list')

			for key, val in temp_dict.items():
				if key in self.data_dict:
					current_list = self.data_dict[key]
					current_list.extend(temp_dict[key])
					self.data_dict[key] = current_list # insert into data dict only if image id is already present
			
			temp_dict.clear() # clear the dictionary
			
		return self.data_dict
		
	def preprocess_MinMaxScaler(self):
		"""
		Method Explanation:
			. Refer to the top of the file for the algorithm for preprocessing.
			. Uses the MinMaxScaling for the normalization of data between 0 and 1.
		"""
		self.data_dict.clear()

		for location in self.location_names:
			current_df = pd.read_csv("../dataset/visual_descriptors/" + location + " " + self.reference_model + ".csv", header = None)
			self.reference_df = self.reference_df.append(current_df, ignore_index = True)

		self.reference_df = self.reference_df.drop_duplicates(subset=[0], keep = 'first') # drop duplicate image ID rows and keep the first one.
		columns_to_normalize = np.arange(1, self.reference_df.shape[1], 1) # the column indices to which MinMax normalization will be applied to.
		self.reference_df[columns_to_normalize] = self.minmax_scaler.fit_transform(self.reference_df[columns_to_normalize]) # MinMax normalization

		self.data_dict = self.reference_df.set_index(0).T.to_dict('list') # Filling the data dict

		temp_dict = dict()
		for index, model in enumerate(self.model_list):
			print('Current model being processed: ', model, '...')
			for location in self.location_names:
				# print('\tLocation being processed: ', location, '...')
				current_df = pd.read_csv("../dataset/visual_descriptors/" + location + " " + model + ".csv", header = None)
				df_to_modify = self.df_list[index]
				df_to_modify = df_to_modify.append(current_df, ignore_index = True)
				self.df_list[index] = df_to_modify
			model_df = self.df_list[index]

			model_df = model_df.drop_duplicates(subset=[0], keep='first') # drop duplicate image ID rows and keep the first one.
			columns_to_normalize = np.arange(1, model_df.shape[1], 1) # the column indices to which MinMax normalization will be applied to.
			model_df[columns_to_normalize] = self.minmax_scaler.fit_transform(model_df[columns_to_normalize])
			self.df_list[index] = model_df

			temp_dict = self.df_list[index].set_index(0).T.to_dict('list')

			for key, val in temp_dict.items():
				if key in self.data_dict:
					current_list = self.data_dict[key]
					current_list.extend(temp_dict[key])
					self.data_dict[key] = current_list # insert into data dict only if image id is already present
			
			temp_dict.clear() # clear the dictionary

		return self.data_dict



