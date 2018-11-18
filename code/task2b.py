import pickle
import random
from util import Util
import constants

class Task2b():
	def __init__(self):
		self.ut = Util()

	def max_a_min_partitioning(self, graph, c, k):
		"""
		1. fix random object as leader 1
		2. select c-1 farthest points from these leaders
		3. perform pass iteration - add images closest to a leader to cluster represented by the leader
		"""
		leaders = self.leader_selection(graph, c, k)
		clusters_heads = self.leader_fixation(leaders)
		final_clusters = self.pass_iteration(clusters_heads, graph)
		return final_clusters

	def leader_selection(self, graph, c_, k):
		"""
		FIXED: ensured distinct leaders in the leader set
		"""
		import pdb
		# pdb.set_trace()
		leaders = []
		#initialse first leader
		leaders.append(random.randint(0,len(graph)))
		for i in range(c_ - 1):
			temp = sorted(graph[leaders[i]], reverse=True)
			leader = graph[leaders[i]].index(temp[k-2])
			if leader in leaders:
				leader = graph[leaders[i]].index(temp[k-3])
			leaders.append(leader)

		return leaders

	def leader_fixation(self, leaders):
		clusters = {}
		for leader in leaders:
			clusters[leader] = []
		return clusters

	def pass_iteration(self, clusters, graph):
		leaders = list(clusters.keys())
		dict_graph = self.ut.fetch_dict_graph()
		for image_iter in range(len(graph)):
			image_out_links = graph[image_iter]
			leader_image_sim_list = [dict_graph[image_iter][(image_iter, leader)] for leader in leaders]

			max_img = leader_image_sim_list.index(max(leader_image_sim_list))
			clusters[leaders[max_img]].append(image_iter)
		return clusters

	def pretty_print(self, c_clusters):
		image_id_mapping_file = open(constants.DUMPED_OBJECTS_DIR_PATH + "image_id_mapping.pickle", "rb")
		image_id_mapping = pickle.load(image_id_mapping_file)[1]
		id_image_mapping = {y:x for x,y in image_id_mapping.items()}
		count = 0

		op = open(constants.TASK2b_OUTPUT_FILE, "w")
		# op.write("K clusters are:\n")
	
		for cluster, image_ids in c_clusters.items():
			count += 1
			print("Cluster " + str(count) + "\n ########################## \n")
			op.write("Cluster " + str(count) + "\n")

			ids = [id_image_mapping[image_id] for image_id in image_ids]
			for temp in ids:
				op.write(temp + "\n")
			op.write("####\n")
			
			print("Cluster head: " + str(id_image_mapping[cluster]) + "\n" + "Clustering: " + str(ids) + "\n")

	def runner(self):
		try:
			initial_k = int(input("Enter the initial value of k: "))
			c = int(input("Enter the value of c (number of clusters): "))
			graph = self.ut.create_adj_mat_from_red_file(initial_k, True)
			c_clusters = self.max_a_min_partitioning(graph, c, initial_k)
			self.pretty_print(c_clusters)
		except Exception as e:
			print(constants.GENERIC_EXCEPTION_MESSAGE + "," + str(type(e)) + "::" + str(e.args))