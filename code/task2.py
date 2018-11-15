import constants
import pickle
import random
from util import Util
from task1 import Task1
import pdb

class Task2():
	def __init__(self):
		self.ut = Util()
		self.task1  = Task1()

	def k_means(self, graph, c):
		"""
		1. find random c points in graph -> initialize them as cluster centroids
		2. while centroids do not converge
				assign each image to the cluster that has closest cluster centroid.
				if all images have been assigned, then recalculate the cluster centroids.
		3. return c clusters
		"""
		initial_cluster_centroids = self.get_cluster_centroids(graph, c)
		initial_clusters = self.form_clusters(graph, initial_cluster_centroids)

		c_clusters = self.converge(c, initial_clusters, graph)

		return c_clusters

	def get_cluster_centroids(self, graph, c):
		return [random.randint(0,len(graph)) for iter in range(0,c)]

	def form_clusters(self, graph, initial_cluster_centroids):
		clusters = {}
		for iter in initial_cluster_centroids:
			clusters[iter] = []

		N = len(graph)
		for i in range(N):
			node_centroid_sim = []
			for centroid in initial_cluster_centroids:
				if i not in initial_cluster_centroids:
					node_sim_list = graph[i]
					for node_node_weight in node_sim_list:
						if node_node_weight[1] == centroid:
							node_centroid_sim.append(node_node_weight[2])

			if len(node_centroid_sim) > 0:
				cluster_centroid_index_with_max_sim = node_centroid_sim.index(max(node_centroid_sim))
				clusters[initial_cluster_centroids[cluster_centroid_index_with_max_sim]].append(i)

		print(clusters)
		return clusters

	def converge(self, c, clusters, graph, default=50):
		"""
		loop until cluster centroids found in consecutive iterations are closer to each other
		"""
		iterations = 0
		while(True):
			cluster_centroids = self.get_cluster_centroids(graph, c)
			c_clusters = self.form_clusters(graph, cluster_centroids)
			iterations += 1
			if self.check_for_convergence(clusters, c_clusters):
				return c_clusters
			elif iterations == default:
				print("****** default iterations ******")
				return c_clusters

	def check_for_convergence(self, clustering1, clustering2):
		cluster1_objects = []
		cluster2_objects = []

		for key1, value1 in clustering1.items():
			cluster1_objects.append(value1.append(key1))
		for key2, value2 in clustering2.items():
			cluster2_objects.append(value2.append(key2))
		
		cluster1_objects = list(clustering1.values())
		cluster2_objects = list(clustering2.values())

		count = 0
		for iter1 in cluster1_objects:
			for iter2 in cluster2_objects:
				union_of_sets = set(iter1).union(set(iter2))
				if len(union_of_sets) == len(iter1) and len(union_of_sets) == len(iter2):
					count += 1

		if count == len(cluster1_objects):
			return True
		else:
			return False

	def pretty_print(self, clusters):
		print(clusters)

	def runner(self):
		try:
			c = int(input("Enter the value of c (number of clusters): "))

			list_of_list_graph_file = open(constants.DUMPED_OBJECTS_DIR_PATH + "list_of_list_graph.pickle", "rb")
			graph = pickle.load(list_of_list_graph_file)
			list_of_list_graph_file.close()

			# pdb.set_trace()
			c_clusters = self.k_means(graph, c)
			self.pretty_print(c_clusters)

		except Exception as e:
			print(constants.GENERIC_EXCEPTION_MESSAGE + "," + str(type(e)) + "::" + str(e.args))