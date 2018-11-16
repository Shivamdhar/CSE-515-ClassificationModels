import pickle
import random
from util import Util

class Task2b():
	def __init__(self):
		self.ut = Util()

	def max_a_min_partitioning(self, graph, c):
		"""
		1. fix random object as leader 1
		2. select c-1 farthest points from these leaders
		3. perform pass iteration - add images closest to a leader to cluster represented by the leader
		"""
		leaders = self.leader_selection(graph, c)
		clusters_heads = self.leader_fixation(leaders)
		final_clusters = self.pass_iteration(clusters_heads, graph)

	def leader_selection(self, graph, c):
		leaders = []
		leaders.append(random.randint(0,len(graph)))
		for i in range(c-1):
			leaders.append(graph[leaders[-1]].index(min(graph[leaders[-1]])))
		return leaders

	def leader_fixation(self, leaders):
		clusters = {}
		for leader in leaders:
			clusters[leader] = []
		return clusters

	def pass_iteration(self, clusters, graph):
		leaders = list(clusters.keys())
		for image_iter in range(len(graph)):
			image_out_links = graph[image_iter]
			leader_image_sim_list = [image_out_links[leader] for leader in leaders]
			clusters[leader].append(leader_image_sim_list.index(max(leader_image_sim_list)))
		return clusters

	def runner(self):
		try:
			c = int(input("Enter the value of c (number of clusters): "))
			graph = self.ut.create_sim_adj_mat_from_red_file(7)
			c_clusters = self.max_a_min_partitioning(graph, c)

		except Exception as e:
			print(constants.GENERIC_EXCEPTION_MESSAGE + "," + str(type(e)) + "::" + str(e.args))