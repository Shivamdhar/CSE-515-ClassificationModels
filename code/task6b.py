import numpy as np
from numpy import linalg as LA
from util import Util
import networkx as nx
import constants
import pandas as pd
class Task6b():
	def __init__(self):
		self.ut = Util()

	def execute_multirank_walk(self,G,xl,yl,xl_yl_map):
		"""
		Method: Returns Labels Yu for unlabeled nodes Xu

		Algorithm for multirank walk follows
		A) For each class c
			1) Set ui(transportation vector) -> 1 such that yl for i = c
			2) ||u|| = 1(Normalize u)
			3) Rc -> RandomWalk(G,u,d)
		B) For each instance
			Set Xiu -> argmax(Rci)
		"""
		#creates a list of V zeros
		u_vector = np.array([0]* len(G))

		reverse_xl_yl_map = dict((v,k) for k,v in xl_yl_map)

		for label in yl:
			x_indexes = [xi for v in reverse_xl_yl_map.values() if v == label]
			x_indexes.sort()
			# Set ui(transportation vector) -> 1 such that yl for i = c
			np.put(u_vector,x_indexes,([1]*len(x_indexes)))

			#normalize u
			u_vector = [i/np.linalg.norm(u_vector) for i in u_vector]



	def runner(self):
		seed_images = []

		image_label_map = []

		f = open("PPR_input1.txt")
		file_content = f.readlines()[2:]

		for row in file_content:
			row_entry = row.split(" ")
			image_id = int(row_entry[0]) #check on iint
			label = row_entry[1].replace("\n","") if row_entry[1] else row_entry[-1].replace("\n","")
			image_label_map[image_id] = label


		yl = list(set(list(image_label_map.values())))
		xl = list(image_label_map.keys())

		image_id_mapping_file = open(constants.DUMPED_OBJECTS_DIR_PATH + "image_id_mapping.pickle", "rb")
		image_id_mapping = pickle.load(image_id_mapping_file)[1]

		# Getting the k value for which reduced file was generated
		graph_file = open(constants.GRAPH_FILE, "r")
		edges = graph_file.readlines()
		size_of_graph = len(image_id_mapping)
		initial_k = graph_file_len // size_of_graph

		graph = self.ut.create_adj_mat_from_red_file(initial_k)

		#should return the map of all the the images with atleast one label
		self.execute_multirank_walk(graph,xl,yl,image_label_map)
