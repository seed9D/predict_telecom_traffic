import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('/home/mldp/ML_with_bigdata')
import data_utility as du

root_dir = '/home/mldp/ML_with_bigdata'

def get_data():
	filelist = du.list_all_input_file(root_dir + '/npy/hour_max/X')
	filelist.sort()
	X_list = []
	Y_list = []
	for i, filename in enumerate(filelist):
		data_array = du.load_array(root_dir + '/npy/hour_max/X/' + filename)
		data_array = data_array[:, :, 50:53, 50:53, :]
		X_list.append(data_array)

	filelist = du.list_all_input_file(root_dir + '/npy/hour_max/Y')
	filelist.sort()
	for i, filename in enumerate(filelist):
		data_array = du.load_array(root_dir + '/npy/hour_max/Y/' + filename)
		data_array = data_array[:, :, 50:53, 50:53, :]
		Y_list.append(data_array)

	X_array = np.concatenate(X_list, axis=0)
	Y_array = np.concatenate(Y_list, axis=0)
	X_array = X_array[:-1]  # important
	Y_array = Y_array[1:]  # important
	print('X data array shape', X_array.shape)
	print('Y data array shape', Y_array.shape)
	return X_array, Y_array


def plot_grid(input_data):
	plt.figure()
	input_shape = input_data.shape
	for row in range(input_shape[2]):
		for col in range(input_shape[3]):
			grid_id = input_data[0, 0, row, col, 0]
			plt.plot(input_data[:, 0, row, col, -1], label=str(grid_id), marker='.')
	plt.legend()
	plt.grid()
	plt.show()


def check_plot(input_x, input_Y):
	plt.figure()
	input_shape = input_x.shape

	plt.plot(input_x[20, :, 0, 0, -1], marker='.')
	plt.plot(input_Y[19, :, 0, 0, -1], marker='.')
	plt.legend()
	plt.grid()
	plt.show()

X_data, Y_data = get_data()
check_plot(X_data[:120], Y_data[:120])
