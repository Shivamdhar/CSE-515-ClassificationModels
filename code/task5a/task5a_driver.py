from task5a_hash_table import Task5aHashTable
from task5a_LSH import Task5aLSH
import numpy as np
import pandas as pd

k = 5
L = 10

#k = input('number of layers')
k = input('Enter the number of Hashes per layer')
L = input('Enter the number of layers')

class Task5aDriver:
	def runner(self):
		k = input('Enter the number of Hashes per layer')
		L = input('Enter the number of layers')

		
lsh = Task5aLSH(int(L), int(k))

# Printing the first hash table for now...
# print('The first hash table:\n')
# # print(lsh.hash_tables[0])
# for i,table in enumerate(lsh.hash_tables):
#     print("For Hash table",i)
#     for key, val in table.hash_table.items():
#     	print('Key/Hash: ', key, ' Value: ', val)
#     	print('-----------------\n')

# sample = lsh.data_matrix[0]
# print(lsh.__getitem__(sample))
print(lsh.__getitem__(lsh.data_matrix[0]))