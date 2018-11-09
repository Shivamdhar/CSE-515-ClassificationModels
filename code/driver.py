"""
This module is a driver program which selects a particular task.
"""
from task1 import Task1
from task2 import Task2

class Driver():

	def input_task_num(self):
		task_num = input("Enter the Task no.: 1, 2, 3, 4, 5a, 5b, 6a, 6b\t")
		self.select_task(task_num)

	def select_task(self, task_num):
		# Plugin class names for each task here
		tasks = { "1": Task1(), "2": Task2() }
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