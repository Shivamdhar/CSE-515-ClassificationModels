from task5_hash_table import Task5HashTable
from task5_LSH import Task5LSH

class Task5Driver:
	def __init__(self):
		self.L = int(input('Enter the number of layers (L): '))
		self.k = int(input('Enter the number of Hashes per layer (k): '))
		# Needs a little bit of a change in logic for running 5a and 5b, will accomodate it once we get sample inputs for 5a
		task = '5b'
		self.lsh = Task5LSH(self.k, self.L, task)
		self.query_imageid = ''
		self.t = int()

	def runner(self):
		print('Initialized!')
		for table_instance in self.lsh.hash_tables:
			print('Number of hash codes/buckets for the given layer: ', len(list(table_instance.hash_table.keys()))) #, ' Max size of any given bucket: ', max(list(table_instance.hash_table.values())))
			print('------------\n\n')
		
		while True:
			self.query_imageid = int(input('Enter the image ID: ')) # 10045482563
			self.t = int(input('Enter the number of nearest neighbors desired (t): '))
			result_list = self.lsh.get_atleast_t_candidate_nearest_neighbors(self.query_imageid, self.t)
			print('Candidate list length: ', len(result_list))
			# print('Returned: ', result_list)
			nearest_neighbors = self.lsh.get_t_nearest_neighbors(self.query_imageid, result_list, self.t)
			print('T Nearest neighbors: ', nearest_neighbors)
			runagain = input('Run again? (Y/N)')
			if runagain == 'N':
				break
		

t = Task5Driver()
t.runner()