import numpy as np
from sklearn import preprocessing
from CNN_RNN import CNN_RNN
import sys
import os
sys.path.append('/home/mldp/ML_with_bigdata')
import data_utility as du

root_dir = '/home/mldp/ML_with_bigdata'


def prepare_training_data():
	'''
	input_dir_list = [
			"/home/mldp/big_data/openbigdata/milano/SMS/11/data_preproccessing_10/",
			"/home/mldp/big_data/openbigdata/milano/SMS/12/data_preproccessing_10/"]

	for input_dir in input_dir_list:
			filelist = list_all_input_file(input_dir)
			filelist.sort()
			load_data_format(input_dir, filelist)
	'''
	def task_1():
		'''
		X: past one hour
		Y: next hour's max value
		'''
		x_target_path = './npy/final/hour_max/training/X'
		y_target_path = './npy/final/hour_max/training/Y'
		if not os.path.exists(x_target_path):
			os.makedirs(x_target_path)
		if not os.path.exists(y_target_path):
			os.makedirs(y_target_path)

		filelist = du.list_all_input_file(root_dir + '/npy/hour_max/X')
		filelist.sort()
		for i, filename in enumerate(filelist):
			if filename != 'training_raw_data.npy':
				data_array = du.load_array(root_dir + '/npy/hour_max/X/' + filename)
				data_array = data_array[:, :, 40:65, 40:65, -1, np.newaxis]  # only network activity
				print('saving  array shape:{}'.format(data_array.shape))
				du.save_array(
					data_array, x_target_path + '/hour_max_' + str(i))

		filelist = du.list_all_input_file(root_dir + '/npy/hour_max/Y')
		filelist.sort()
		for i, filename in enumerate(filelist):
			max_array = du.load_array(root_dir + '/npy/hour_max/Y/' + filename)
			max_array = max_array[:, :, 40:65, 40:65, -1, np.newaxis]  # only network activity
			du.save_array(max_array, y_target_path + '/hour_max_' + str(i))

	def task_2():
		'''
		rolling 10 minutes among timeflows
		X: past one hour
		Y: next 10 minutes value
		'''
		x_target_path = './npy/final/roll_10/training/X'
		y_target_path = './npy/final/roll_10/training/Y'
		if not os.path.exists(x_target_path):
			os.makedirs(x_target_path)
		if not os.path.exists(y_target_path):
			os.makedirs(y_target_path)

		filelist_X = du.list_all_input_file(root_dir + '/npy/npy_roll/X/')
		filelist_Y = du.list_all_input_file(root_dir + '/npy/npy_roll/Y/')
		filelist_X.sort()
		filelist_Y.sort()
		for i, filename in enumerate(filelist_X):
			data_array = du.load_array(root_dir + '/npy/npy_roll/X/' + filename)
			data_array = data_array[:, :, 40:65, 40:65, -1, np.newaxis]  # only network activity
			print('saving  array shape:{}'.format(data_array.shape))
			du.save_array(data_array, x_target_path + '/X_' + str(i))

		for i, filename in enumerate(filelist_Y):
			data_array = du.load_array(root_dir + '/npy/npy_roll/Y/' + filename)
			data_array = data_array[:, :, 40:65, 40:65, -1, np.newaxis]  # only network activity
			print('saving  array shape:{}'.format(data_array.shape))
			du.save_array(data_array, y_target_path + '/Y_' + str(i))

	task_2()


def get_X_and_Y_array():

	def task_1():
		'''
		X: past one hour
		Y: next hour's max value
		'''
		x_dir = './npy/final/hour_max/training/X/'
		y_dir = './npy/final/hour_max/training/Y/'
		x_data_list = du.list_all_input_file(x_dir)
		x_data_list.sort()
		y_data_list = du.list_all_input_file(y_dir)
		y_data_list.sort()

		X_array_list = []
		for filename in x_data_list:
			X_array_list.append(du.load_array(x_dir + filename))

		X_array = np.concatenate(X_array_list, axis=0)
		del X_array_list

		Y_array_list = []
		for filename in y_data_list:
			Y_array_list.append(du.load_array(y_dir + filename))
		Y_array = np.concatenate(Y_array_list, axis=0)
		del Y_array_list
		X_array = feature_scaling(X_array)
		Y_array = feature_scaling(Y_array)
		X_array = X_array[0: -1]  # important!!
		Y_array = Y_array[1:]  # important!! Y should shift 10 minutes
		return X_array, Y_array

	def task_2():
		'''
		rolling 10 minutes among timeflows
		return :
			X_array: past one hour [?, 6, row, col, 1]
			Y_array: next 10 minutes value [?, 1, row, col, 1]

		'''
		x_dir = './npy/final/roll_10/training/X/'
		y_dir = './npy/final/roll_10/training/Y/'
		X_file_list = du.list_all_input_file(x_dir)
		Y_file_list = du.list_all_input_file(y_dir)
		X_file_list.sort()
		Y_file_list.sort()
		X_array_list = []
		Y_array_list = []
		# X array
		for filename in X_file_list:
			X_array_list.append(du.load_array(x_dir + filename))
		X_array = np.concatenate(X_array_list, axis=0)
		del X_array_list
		# Y array
		for filename in Y_file_list:
			Y_array_list.append(du.load_array(y_dir + filename))
		Y_array = np.concatenate(Y_array_list, axis=0)
		del Y_array_list
		X_array = feature_scaling(X_array)
		Y_array = feature_scaling(Y_array)
		return X_array, Y_array

	return task_2()


def feature_scaling(input_datas):
	input_shape = input_datas.shape
	input_datas = input_datas.reshape(input_shape[0], -1)
	min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 255))
	output = min_max_scaler.fit_transform(input_datas)
	output = output.reshape(input_shape)
	return output


def print_Y_array(Y_array):
	for i in range(200):
		for j in range(Y_array.shape[1]):
			print(Y_array[i, j, 10, 10])

if __name__ == '__main__':
	# prepare_training_data()
	X_array, Y_array = get_X_and_Y_array()
	# print_Y_array(Y_array)
	# parameter
	data_shape = [X_array.shape[1], X_array.shape[2], X_array.shape[3], X_array.shape[4]]

	cnn_rnn = CNN_RNN(*data_shape)
	cnn_rnn.set_training_data(X_array, Y_array)
	del X_array, Y_array
	cnn_rnn.start_train()
