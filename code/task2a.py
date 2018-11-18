import constants
import scipy
from util import Util

class Task2a():
	def __init__(self):
		self.ut = Util()

	def anglular_clustering(self, graph, c):
		"""
		1. perform SVD on the adjacency matrix.
		2. find top k singular vectors corresponding to the largest eigen values
		3. k eigen vectors form the clusters
		4. Now, assign each node to the cluster by finding max value in the row for these vectors.
		"""
		top_singular_vector_matrix = self.fetch_singular_vectors(graph, c)
		c_clusters = self.partition(top_singular_vector_matrix)
		return c_clusters
	
	def fetch_singular_vectors(self, graph, c):
		sparse_matrix = scipy.sparse.csc_matrix(graph, dtype=float)
		u, s, vt = scipy.sparse.linalg.svds(graph, k=c)
		return u

	def partition(self, top_singular_vector_matrix):
		clusters = {}
		for iter in range(len(top_singular_vector_matrix[0])):
			clusters[iter] = []

		for node in range(len(top_singular_vector_matrix)):
			index = top_singular_vector_matrix[node].index(max(top_singular_vector_matrix[node]))
			clusters[index].append(node)

		return clusters

	def pretty_print(self, c_clusters):
		print c_clusters

	def runner(self):
		try:
			initial_k = int(input("Enter the initial value of k: "))
			c = int(input("Enter the value of c (number of clusters): "))
			graph = self.ut.create_adj_mat_from_red_file(initial_k, True)
			c_clusters = self.anglular_clustering(graph, c)
			self.pretty_print(c_clusters)
		except Exception as e:
			print(constants.GENERIC_EXCEPTION_MESSAGE + "," + str(type(e)) + "::" + str(e.args))