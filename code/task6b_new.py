from collections import OrderedDict
import constants
import numpy as np
import pickle
import re
import scipy.sparse as sparse
from task4 import Task4
from util import Util


class Task6b():
	def __init__(self):
		self.ut = Util()
		self.ppr = Task4()
		self.ppr.d = 0.95
		image_id_mapping_file = open(constants.DUMPED_OBJECTS_DIR_PATH + "image_id_mapping.pickle", "rb")
		self.image_id_mapping = pickle.load(image_id_mapping_file)[1]


	def get_normalized_seed_vector(self,seed_vector,graph_len):
		"""
		Returns the normalized seed vector such the norm of the vector is 1
		"""
		#normalize u
		seed_vector = self.ppr.initialize_vq(seed_vector,graph_len)
		seed_vector = [i/np.linalg.norm(seed_vector) for i in seed_vector]

		return np.array(seed_vector)

	def ppr_classifier(self, graph, image_label_map):
		"""
		Algorithm to classify images based on PPR
		1. Fetch seeds to compute personalized pagerank for each label
		2. Find pagerank vector for each label
		3. Argmax for finding label for the unclassified instace
		"""
		seed_label_map = self.fetch_seeds(image_label_map)
		label_ppr_map = {}
		graph_len = len(graph)
		M = self.ppr.normalize_M(graph)
		for label, seed_list in seed_label_map.items():
			seed_vector = self.get_normalized_seed_vector(seed_list,graph_len)
			rank_vector = seed_vector
			rank_vector = self.ppr.converge(rank_vector, seed_vector, M)
			label_ppr_map[label] = np.array(rank_vector)
			#label_ppr_map[label] = np.array(self.ppr.personalised_pagerank(graph, seed_list))
		classified_image_label_map = self.classify(label_ppr_map)
		return classified_image_label_map

	def fetch_seeds(self,image_label_map):
		seed_list_map = {}
		for key, value in image_label_map.items():
			if value in seed_list_map.keys():
				seed_list_map[value].append(key)
			else:
				seed_list_map[value] = [key]
		return seed_list_map

	def get_label_index_map(self,labels):
		label_index_map = {label:i for i,label in enumerate(labels)}
		return label_index_map

	def get_labels_from_indexes(self,label_indexes,index_label_map):
		label_list = []

		for iter in label_indexes:
			label_list.append(index_label_map[iter])

		return label_list


	def classify(self, label_ppr_map):
		label_list = label_ppr_map.keys()
		pagerank_vectors = label_ppr_map.values()
		pagerank_matrix = np.vstack(pagerank_vectors)
		label_index_map = self.get_label_index_map(label_list)
		index_label_map = dict((v,k) for k,v in label_index_map.items())

		image_label_matrix = pagerank_matrix.T
		classified_image_label_map = {}

		for i,v in enumerate(image_label_matrix):
			label_indexes = np.argwhere(v == np.amax(v)).flatten().tolist()
			if np.max(v) == 0:
				#Assigning the first label where 0 has occured.
				label_indexes = [label_indexes[0]]
			computed_labels = self.get_labels_from_indexes(label_indexes,index_label_map)
			classified_image_label_map[i] = computed_labels

		#classified_image_label_map = {image:reverse_label_index_map[image] for i,index in enumerate(image_label_indexes)}

		return classified_image_label_map

	def pretty_print(self,label_image_map):
		op = open(constants.TASK6b_OUTPUT_FILE, "w")
		count = 0

		for label, image_ids in label_image_map.items():
			count += 1
			print("Label " + str(count) + "\n ########################## \n")
			op.write("Label " + label + "\n")

			id_image_mapping = { y:x for x,y in self.image_id_mapping.items() }

			ids = [id_image_mapping[image_id] for image_id in image_ids]
			for temp in ids:
				op.write(temp + "\n")
			op.write("####\n")

	def runner(self):
		#try:
		image_label_map = OrderedDict({})

		f = open("PPR_input2.txt")
		file_content = f.readlines()[2:]

		for row in file_content:
			row_entry = row.split(" ")
			row_entry = [item.strip() for item in row_entry if re.match('\\W?\\w+', item)]
			image_id = self.image_id_mapping[row_entry[0]]
			label = row_entry[1]
			image_label_map[image_id] = label

		initial_k = self.ut.validate_and_get_correct_k()
		graph = self.ut.create_adj_mat_from_red_file(initial_k, True)
		classified_image_label_map = self.ppr_classifier(graph, image_label_map)

		singular_image_count = 0
		singular_images = []

		label_image_map = {}

		for k,v in classified_image_label_map.items():
			if len(v) == 1:
				singular_image_count+=1 
				singular_images.append((k,v[0]))

		#print(classified_image_label_map)

		for k,v in classified_image_label_map.items():
			for i in v:
				if i in label_image_map:
					label_image_map[i].append(k)
				else:
					label_image_map[i] = [k]

		# print(singular_image_count)
		# print(singular_images)

		# print(label_image_map)

		self.pretty_print(label_image_map)

		# except Exception as e:
		# 	print(constants.GENERIC_EXCEPTION_MESSAGE + "," + str(type(e)) + "::" + str(e.args))