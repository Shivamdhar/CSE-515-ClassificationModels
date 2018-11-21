import numpy as np
from numpy import linalg as LA
from util import Util
import networkx as nx
import constants
import pandas as pd
import pickle
from task3 import Task3
from task3_iterative import Task3_iterative
class Task6b():
	def __init__(self):
		self.ut = Util()
		image_id_mapping_file = open(constants.DUMPED_OBJECTS_DIR_PATH + "image_id_mapping.pickle", "rb")
		self.image_id_mapping = pickle.load(image_id_mapping_file)[1]
		self.task3 = Task3()
		self.task3_new = Task3_iterative()

	def get_weighted_transition_matrix(self,graph):
		"""
		Method : Returns the weighted transition matrix(column stochastic) with each column having sum as 1
		and each individual value being reciprocal of degree of the node of the graph
		Wij = 1/degree(vi)
		"""
		out_degree_list = self.task3_new.calculate_node_outdegree(graph)
		pointing_nodes_list = self.task3_new.derive_pointing_nodes_list(graph.T)
		#degree_vi = len(out_degree_list) + len(pointing_nodes_list)
		n = len(graph)
		W = np.array([[0] * n] * n)

		#TODO VEctorize this , think about j can't do like this
		for i in range(0,n):
			#for j in range(0,n):
			W[i,:] = out_degree_list[i] + len(pointing_nodes_list[i])

		#compute W matrix using degree	
		#W = self.task3.compute_M(graph)
		return np.array(W)
		pass


	def get_normalized_teleportation_vector(self,u_vector,xl,yl,image_label_map,label):

		image_ids = [k for k,v in image_label_map.items() if v == label]

		image_indexes = [self.image_id_mapping[i] for i in image_ids]
		#image_indexes.sort()

		# Set ui(transportation vector) -> 1 such that yl for i = c
		np.put(u_vector,image_indexes,([1]*len(image_indexes)))

		#assert u_vector.non_zero()[0].tolist() == image_indexes

		#normalize u
		u_vector = [i/np.linalg.norm(u_vector) for i in u_vector]

		return np.array(u_vector)

	def get_ranking_vector(self,graph,u):
		"""
		Get the ranking vector
		Computed using the expression - (1-d)*(I - dW)^-1*u
		"""
		W = self.get_weighted_transition_matrix(graph)
		d = 0.85

		n = len(graph)

		left_operand = (1-d) * u
		ri = (np.linalg.inv(np.eye(n) - d*W) @ u)
		right_operand = (d*W) @ ri

		rvector_expression = left_operand + right_operand

		w, v = sparse.linalg.eigs(rvector_expression)
		return 

	def execute_multirank_walk(self,G,xl_yl_map):
		"""
		Method: Returns Labels Yu for unlabeled nodes Xu

		Algorithm for multirank walk follows
		A) For each class c
			1) Set ui(teleportation_vector) -> 1 such that yl for i = c
			2) ||u|| = 1(Normalize u)
			3) Rc -> RandomWalk(G,u,d)
		B) For each instance
			Set Xiu -> argmax(Rci)
		"""
		#creates a list of V zeros

		yl = list(set(list(image_label_map.values())))
		xl = list(image_label_map.keys())

		u_vector = np.array([0]* len(G))

		for label in yl:
			u_vector_normalized = self.get_normalized_teleportation_vector(u_vector,xl,yl,
				image_label_map,label)
			rc = self.get_ranking_vector(graph)


	def classifier(self):
		pass


	def runner(self):
		seed_images = []
		image_label_map = {}

		f = open("PPR_input1.txt")
		file_content = f.readlines()[2:]

		for row in file_content:
			row_entry = row.split(" ")
			image_id = int(row_entry[0]) #check on iint
			label = row_entry[1].replace("\n","") if row_entry[1] else row_entry[-1].replace("\n","")
			image_label_map[image_id] = label

		# Getting the k value for which reduced file was generated
		initial_k = self.ut.validate_and_get_correct_k()

		graph = self.ut.create_adj_mat_from_red_file(initial_k)

		graph = np.array(graph)

		#should return the map of all the the images with atleast one label
		self.execute_multirank_walk(graph,xl,yl,image_label_map)
