"""
This module is a driver program which selects a particular task.
"""
from task1 import Task1
from task2a import Task2a
from task2b import Task2b
from task3 import Task3
from task3_iterative import Task3_iterative
from task4_nx import Task4

class Driver():

	def input_task_num(self):
		task_num = input("Enter the Task no.: 1, 2a(Spectral), 2b(Leader), 3(PageRank), 4(PPR), 5a, 5b, 6a, 6b\t")
		self.select_task(task_num)

	def select_task(self, task_num):
		# Plugin class names for each task here
		tasks = { "1": Task1(), "2a": Task2a(),  "2b": Task2b(), "3": Task3_iterative(), "4": Task3_iterative(personalised = True), "4nx": Task4()}
		# Have a runner method in all the task classes
		tasks.get(task_num).runner()

flag = True
while(flag):
	choice = int(input("Enter your choice:\t1) Execute tasks \t2) Exit\n"))

	if choice == 2:
		flag = False
	else:
		t = Driver()
		t.input_task_num()