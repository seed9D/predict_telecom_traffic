import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('/home/mldp/ML_with_bigdata')
import data_utility as du

method_1_result_path = '/home/qiuhui/processed_data/'
method_CNN_RNN_result_path = '/home/mldp/ML_with_bigdata/CNN_RNN/result/Y_real_prediction.npy'

plot_grid_id_list = [4258, 4456, 4457]


def get_quihui_result(search_grid_id):
	min_file = os.path.join(method_1_result_path, str(search_grid_id) + '_min.npy')
	avg_file = os.path.join(method_1_result_path, str(search_grid_id) + '_avg.npy')
	max_file = os.path.join(method_1_result_path, str(search_grid_id) + '_max.npy')

	min_array = du.load_array(min_file)[-100:]
	avg_array = du.load_array(avg_file)[-100:]
	max_array = du.load_array(max_file)[-100:]
	
	min_array = np.reshape(min_array, (100, 1, 1), order='C')
	avg_array = np.reshape(avg_array, (100, 1, 1), order='C')
	max_array = np.reshape(max_array, (100, 1, 1), order='C')

	qiuhui_array = np.concatenate((min_array, avg_array, max_array), axis=-1)
	# print(qiuhui_array.shape)
	return qiuhui_array


def get_CNN_RNN_result(search_grid_id):
	def search(search_array, search_grid_id):
		for row in range(CNN_RNN_result.shape[2]):
			for col in range(CNN_RNN_result.shape[3]):
				grid_id_ = CNN_RNN_result[0, 0, row, col, 0]
				if int(search_grid_id) == int(grid_id_):
					return CNN_RNN_result[:, :, row, col, :]

	CNN_RNN_result = du.load_array(method_CNN_RNN_result_path)

	return search(CNN_RNN_result, search_grid_id)


def plot_result(info_real, CNN_RNN, qiuhui):
	def plot_task(task_name, task_info, task_real, task_cnn_rnn, task_quihui):
		fig = plt.figure()
		plt.xlabel('time sequence')
		plt.ylabel('activity strength')

		plt.plot(task_real, label='real', marker='.')
		plt.plot(task_cnn_rnn, label='CNN_RNN', marker='.')
		plt.plot(task_quihui, label='??', marker='.')
		plt.title(task_name + ': grid_id ' + str(task_info[0]))
		plt.grid()
		plt.legend()

	plot_task('min', info_real[0, 0, (0, 1)], info_real[:, 0, 2], CNN_RNN[:, 0, 0], qiuhui[:, 0, 0])
	plot_task('avg', info_real[0, 0, (0, 1)], info_real[:, 0, 3], CNN_RNN[:, 0, 1], qiuhui[:, 0, 1])
	plot_task('max', info_real[0, 0, (0, 1)], info_real[:, 0, 4], CNN_RNN[:, 0, 2], qiuhui[:, 0, 2])

	plt.show()

for each_grid_id in plot_grid_id_list:
	CNN_RNN = get_CNN_RNN_result(each_grid_id)
	info_real = CNN_RNN[:, :, :5]
	CNN_RNN = CNN_RNN[:, :, 5:]
	qiuhui = get_quihui_result(each_grid_id)
	plot_result(info_real, CNN_RNN, qiuhui)
