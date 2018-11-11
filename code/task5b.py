import os

class Task5b():
	def __init__(self):
		self.ut = Util()

	def get_image_features_dataset(self):
		list_of_files = os.listdir(constants.PROCESSED_VISUAL_DESCRIPTORS_DIR_PATH)
		image_feature_matrix = {}
		color_models = ["CM", "CM3x3", "CN", "CN3x3", "CSD"]
		for filename in list_of_files:
			model = filename.split(" ")[1].replace(".csv","")
			if model in color_models:
				with open(os.path.join(visual_dir_path,filename)) as file:
					for row in file:
						row_data = row.strip().split(",")
						feature_values = list(map(float, row_data[1:]))
						image_id = row_data[0]
						if image_id in image_feature_matrix:
							image_feature_matrix[image_id] += feature_values
						else:
							image_feature_matrix[image_id] = feature_values
		return image_feature_matrix

	def runner(self):
		try:
			image_id = input("Enter the query image")
			t = int(input("Enter the value of t (number of similar images): "))
			image_feature_matrix = self.get_image_features_dataset()