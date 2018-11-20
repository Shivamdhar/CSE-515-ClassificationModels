import constants
import numpy as np
import pickle
import scipy.sparse as sparse
from util import Util

class Task4():
	def __init__(self):
		self.ut = Util()
		self.d = 0.85

	def pagerank(self, graph, K, seeds=[]):
		"""
		Algorithm to compute PPR
		1. Let vq=0, for all its N entries, except a ’1’ for the q-th entry.
		2. Normalize the adjacency matrix of A(graph), by column. That is, make each column sum to 1.
		3. Initialize uq=vq
		4. while(uq has not converged)
		4.1 uq = (1-c)*A*uq + c*vq
		"""
		vq = self.initialize_vq(seeds, len(graph))
		uq = vq
		M = self.normalize_M(graph)
		uq = self.converge(uq, vq, M)
		self.top_k(uq, K)

	def initialize_vq(self, seeds, graph_len):
		vq = [0]*graph_len
		for iter in seeds:
			vq[iter] = 1.0
		return vq

	def normalize_M(self, graph):
		graph_transpose = np.transpose(graph)
		for row in graph_transpose:
			row = row / sum(row)
		return np.transpose(graph_transpose)

	def converge(self, uq, vq, M):
		uq = self.compute_uq(uq, vq, M)
		uq_list = []
		uq_list.append(uq)
		converged = False
		while(!converged):
			uq = self.compute_uq(uq_list[-1], vq, M)
			if(uq == uq_list[-1]):
				converged = True
		return uq_list[-1]

	def compute_uq(self, uq, vq, M):
		left_operand = np.multiply(M, uq)
		right_operand = np.multiply(vq, self.d)
		uq = np.multiply((1-self.d), left_operand) + right_operand
		return uq

	def top_k(self, pagerank_score, K):
		image_id_mapping_file = open(constants.DUMPED_OBJECTS_DIR_PATH + "image_id_mapping.pickle", "rb")
		image_id_mapping = pickle.load(image_id_mapping_file)[1]

		image_id_score_mapping = {}

		for iter in range(0, len(pagerank_score)):
			for image_id, index in image_id_mapping.items():
				if index == iter:
					image_id_score_mapping[image_id] = pagerank_score[iter]
		print("Top K images based on pagerank score\n")
		if(self.personalised == False):
			op = open(constants.TASK3_OUTPUT_FILE, "w")
			op.write("K most dominant images are:\n")
			for image_id, score in sorted(image_id_score_mapping.items(), key=lambda x: x[1], reverse=True)[:K]:
				op.write(str(image_id))
				op.write("\n")
		else:
			op = open(constants.TASK4_OUTPUT_FILE, "w")
			op.write("K most dominant images are:\n")
			for image_id, score in sorted(image_id_score_mapping.items(), key=lambda x: x[1], reverse=True)[:K]:
				op.write(str(image_id))
				op.write("\n")

		print(sorted(image_id_score_mapping.items(), key=lambda x: x[1], reverse=True)[:K])

	def runner(self):
		try:
			image_id_mapping_file = open(constants.DUMPED_OBJECTS_DIR_PATH + "image_id_mapping.pickle", "rb")
			image_id_mapping = pickle.load(image_id_mapping_file)[1]
			seeds = []
			K = int(input("Enter the value of K: "))
			initial_k = int(input("Enter the initial value of k: "))
			if self.personalised:
				print("Enter three image ids to compute PPR:\n")
				image_id1 = input("Image id1:")
				seeds.append(image_id_mapping[image_id1])
				image_id2 = input("Image id2:")
				seeds.append(image_id_mapping[image_id2])
				image_id3 = input("Image id3:")
				seeds.append(image_id_mapping[image_id3])

			graph = self.ut.create_adj_mat_from_red_file(initial_k)
			k_dominant_images = self.pagerank(graph, K, seeds)

		except Exception as e:
			print(constants.GENERIC_EXCEPTION_MESSAGE + "," + str(type(e)) + "::" + str(e.args))