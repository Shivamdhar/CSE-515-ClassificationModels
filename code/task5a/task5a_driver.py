from task5a_hash_table import Task5aHashTable
from task5a_LSH import Task5aLSH
import numpy as np
import pandas as pd

k = 5
L = 10

class Task5aDriver:
	def runner(self):
		k = input('Enter the number of Hashes per layer')
		L = input('Enter the number of layers')

		
lsh = Task5aLSH(L, k)

# Printing the first hash table for now...
# print('The first hash table:\n')
# print(lsh.hash_tables[0])

for table_instance in lsh.hash_tables:
	for key, val in table_instance.hash_table.items():
		print('Key/Hash: ', key) # ' Value: ', val)
		print('-----------------')
	print('-----------------\n\n')

sample = lsh.data_matrix[0]
print(lsh.__getitem__(sample), "Size: ", len(lsh.__getitem__(sample)))
# print(lsh.__getitem__(lsh.data_matrix[0]))