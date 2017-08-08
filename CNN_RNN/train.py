import numpy as np
from utility import feature_scaling, root_dir
import matplotlib.pyplot as plt
import sys
import CNN_RNN_config
from CNN_RNN import CNN_RNN, CNN_3D, RNN, concurrent_CNN_RNN
import os
sys.path.append(root_dir)
import data_utility as du
from multi_task_data import Prepare_Task_Data

# root_dir = '/home/mldp/ML_with_bigdata'

def print_Y_array(Y_array):
	print('Y array shape:{}'.format(Y_array.shape))
	plot_y_list = []
	for i in range(148):
		for j in range(Y_array.shape[1]):
			print(Y_array[i, j, 5, 10])
			plot_y_list.append(Y_array[i, j, 5, 10])
	plt.figure()
	plt.plot(plot_y_list, marker='.')
	plt.show()


def train():
	# X_array, Y_array = get_X_and_Y_array(task_num=5)
	TK = Prepare_Task_Data('./npy/final')
	X_array, Y_array = TK.Task_max_min_avg(generate_data=False)
	data_len = X_array.shape[0]
	X_array = X_array[: 9 * data_len // 10, :, :, :, -1, np.newaxis]
	Y_array = Y_array[: 9 * data_len // 10, :, :, :, 2:]
	# X_array = X_array[:, :, 10:15, 10:15, :]
	Y_array = Y_array[:, :, 10:13, 10:13, :]
	X_array, scaler = feature_scaling(X_array)
	Y_array, _ = feature_scaling(Y_array, scaler)

	# X_array_2, Y_array_2 = get_X_and_Y_array(task_num=6)
	# Y_array_2 = Y_array_2[:, :, 10:13, 10:13, :]
	# parameter
	input_data_shape = [X_array.shape[1], X_array.shape[2], X_array.shape[3], X_array.shape[4]]
	output_data_shape = [Y_array.shape[1], Y_array.shape[2], Y_array.shape[3], 1]
	result_path = './result/temp/'
	# result_path = os.path.join(result_path,'report')
	model_path = {
		'reload_path': './output_model/CNN_RNN_test.ckpt',
		'save_path': './output_model/CNN_RNN_test.ckpt',
		'result_path': result_path
	}
	hyper_config = CNN_RNN_config.HyperParameterConfig()
	hyper_config.read_config(file_path=os.path.join(root_dir, 'CNN_RNN/result/random_search_0609/_85/config.json'))

	neural = concurrent_CNN_RNN(input_data_shape, output_data_shape, hyper_config)

	neural.create_MTL_task(X_array, Y_array[:, :, :, :, 0, np.newaxis], 'min_traffic')
	neural.create_MTL_task(X_array, Y_array[:, :, :, :, 1, np.newaxis], 'avg_traffic')
	neural.create_MTL_task(X_array, Y_array[:, :, :, :, 2, np.newaxis], 'max_traffic')
	del X_array, Y_array
	# neural.create_MTL_task(X_array_2, Y_array_2[:, :, :, :, 0, np.newaxis], '10_mins', 'cross_entropy')
	# del X_array_2, Y_array_2

	# neural.start_train(model_path, reload=False)
	neural.start_MTL_train(model_path, reload=False)


if __name__ == '__main__':
	train()

