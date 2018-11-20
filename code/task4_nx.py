import numpy as np
from numpy import linalg as LA
from util import Util
import networkx as nx
import constants
class Task4_nx():

	def PageRank(self, G, alpha=0.85, personalization=None, 
			 max_iter=100, tol=1.0e-6, nstart=None, weight='weight', 
			 dangling=None): 
	
		if len(G) == 0: 
			return {} 

		if not G.is_directed(): 
			D = G.to_directed() 
		else: 
			D = G 

		# Create a copy in (right) stochastic form 
		W = nx.stochastic_graph(D, weight=weight) 
		N = W.number_of_nodes() 

		# Choose fixed starting vector if not given 
		if nstart is None: 
			x = dict.fromkeys(W, 1.0 / N) 
		else: 
			# Normalized nstart vector 
			s = float(sum(nstart.values())) 
			x = dict((k, v / s) for k, v in nstart.items()) 

		if personalization is None: 

			# Assign uniform personalization vector if not given 
			p = dict.fromkeys(W, 1.0 / N) 
		else: 
			missing = set(G) - set(personalization) 
			if missing: 
				raise NetworkXError('Personalization dictionary '
									'must have a value for every node. '
									'Missing nodes %s' % missing) 
			s = float(sum(personalization.values())) 
			p = dict((k, v / s) for k, v in personalization.items()) 

		if dangling is None: 

			# Use personalization vector if dangling vector not specified 
			dangling_weights = p 
		else: 
			missing = set(G) - set(dangling) 
			if missing: 
				raise NetworkXError('Dangling node dictionary '
									'must have a value for every node. '
									'Missing nodes %s' % missing) 
			s = float(sum(dangling.values())) 
			dangling_weights = dict((k, v/s) for k, v in dangling.items()) 
		dangling_nodes = [n for n in W if W.out_degree(n, weight=weight) == 0.0] 

		# power iteration: make up to max_iter iterations 
		for _ in range(max_iter): 
			xlast = x 
			x = dict.fromkeys(xlast.keys(), 0) 
			danglesum = alpha * sum(xlast[n] for n in dangling_nodes) 
			for n in x: 

				# this matrix multiply looks odd because it is 
				# doing a left multiply x^T=xlast^T*W 
				for nbr in W[n]: 
					x[nbr] += alpha * xlast[n] * W[n][nbr][weight] 
				x[n] += danglesum * dangling_weights[n] + (1.0 - alpha) * p[n] 

			# check convergence, l1 norm 
			err = sum([abs(x[n] - xlast[n]) for n in x]) 
			if err < N*tol: 
				return x 
		raise NetworkXError('pagerank: power iteration failed to converge '
							'in %d iterations.' % max_iter)

	def runner(self):

		#create graph using networkx
		G = nx.read_edgelist(constants.GRAPH_FILE, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
		personalised_dict = {}

		#take the seeds as input
		seed_1 = int(input("Enter seed1: "))
		seed_2 = int(input("Enter seed2: "))
		seed_3 = int(input("Enter seed3: "))
		value_of_k = int(input("Enter value of K: "))

		#initialise personalised_dict with zeroes
		for each in set(G):
			personalised_dict[each]=0

		#update the dict with seed values as 0.33 for each of the 3 seeds
		for key, value in personalised_dict.items():
			if(key == seed_1):
				personalised_dict[key] = 0.33
			elif(key == seed_2):
				personalised_dict[key] = 0.33
			elif(key == seed_3):
				personalised_dict[key] = 0.33
				
		page_rank_values = self.PageRank(G, personalization = personalised_dict)
		sorted_by_value = sorted(page_rank_values.items(), key=lambda kv: kv[1], reverse = True)[:value_of_k]
		op = open(constants.TASK4NX_OUTPUT_FILE, "w")
		op.write("K most dominant images are:\n")
		for image in sorted_by_value:
			op.write(str(list(image)[0]))
			op.write("\n")
		print(sorted_by_value)