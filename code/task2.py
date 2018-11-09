import random
class Task2():
	def __init__(self):
		self.ut = Util()

	def k_means(self, graph, c):
		"""
		1. find random c points in graph -> initialize them as cluster centroids
		2. while centroids do not converge
				assign each image to the cluster that has closest cluster centroid.
				if all images have been assigned, then recalculate the cluster centroids.
		3. return c clusters
		"""
		initial_cluster_centroids, edge_set = self.get_initial_cluster_centroids(graph, c)
		c_clusters = self.converge(graph, edge_list, initial_cluster_centroids)

		return c_clusters

	def get_initial_cluster_centroids(self, graph, c):
		edge_set = [edge for edge_list in graph for edge in edge_list]
		return [random.randint(0,len(edge_set)) for iter in range(0,c)], edge_list
		#return [random.choice(edge_set) for cluster_no in range(0, c)]

	def converge(self, graph, edge_list, initial_cluster_centroids):
		"""
		1. loop until cluster centroids found in consecutive iterations are closer to each other
		"""
		clusters = {}
		# edge_set = [edge for edge_list in graph for edge in edge_list]

	def runner(self):
		try:
			c = int(input("Enter the value of c (number of clusters): "))
			graph = self.ut.fetch_graph()
			c_clusters = k_means(graph, c)

		except Exception as e:
			print(constants.GENERIC_EXCEPTION_MESSAGE + "," + str(type(e)) + "::" + str(e.args))