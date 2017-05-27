import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import sys
import CNN_RNN_config
import os
sys.path.append('/home/mldp/ML_with_bigdata')
import data_utility as du

root_dir = '/home/mldp/ML_with_bigdata'


def prepare_training_data(task_num=2):
	grid_start = 45
	grid_stop = 60
	'''
	input_dir_list = [
			"/home/mldp/big_data/openbigdata/milano/SMS/11/data_preproccessing_10/",
			"/home/mldp/big_data/openbigdata/milano/SMS/12/data_preproccessing_10/"]

	for input_dir in input_dir_list:
			filelist = list_all_input_file(input_dir)
			filelist.sort()
			load_data_format(input_dir, filelist)
	'''
	def _task_1():
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
				data_array = data_array[:, :, grid_start:grid_stop, grid_start:grid_stop, -1, np.newaxis]  # only network activity
				print('saving  array shape:{}'.format(data_array.shape))
				du.save_array(
					data_array, x_target_path + '/hour_max_' + str(i))

		filelist = du.list_all_input_file(root_dir + '/npy/hour_max/Y')
		filelist.sort()
		for i, filename in enumerate(filelist):
			max_array = du.load_array(root_dir + '/npy/hour_max/Y/' + filename)
			max_array = max_array[:, :, grid_start:grid_stop, grid_start:grid_stop, -1, np.newaxis]  # only network activity
			du.save_array(max_array, y_target_path + '/hour_max_' + str(i))

	def _task_2():
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
			data_array = data_array[:, :, grid_start:grid_stop, grid_start:grid_stop, -1, np.newaxis]  # only network activity
			print('saving  array shape:{}'.format(data_array.shape))
			du.save_array(data_array, x_target_path + '/X_' + str(i))

		for i, filename in enumerate(filelist_Y):
			data_array = du.load_array(root_dir + '/npy/npy_roll/Y/' + filename)
			data_array = data_array[:, :, grid_start:grid_stop, grid_start:grid_stop, -1, np.newaxis]  # only network activity
			print('saving  array shape:{}'.format(data_array.shape))
			du.save_array(data_array, y_target_path + '/Y_' + str(i))

	def _task_3():
		'''
			X: past one hour
			Y: next hour's average value
		'''
		x_target_path = './npy/final/hour_avg/training/X'
		y_target_path = './npy/final/hour_avg/training/Y'
		if not os.path.exists(x_target_path):
			os.makedirs(x_target_path)
		if not os.path.exists(y_target_path):
			os.makedirs(y_target_path)

		filelist = du.list_all_input_file(root_dir + '/npy/hour_avg/X')
		filelist.sort()

		for i, filename in enumerate(filelist):
			if filename != 'training_raw_data.npy':
				data_array = du.load_array(root_dir + '/npy/hour_avg/X/' + filename)
				data_array = data_array[:, :, grid_start:grid_stop, grid_start:grid_stop, -1, np.newaxis]  # only network activity
				print('saving array shape:{}'.format(data_array.shape))
				du.save_array(data_array, x_target_path + '/hour_avg_' + str(i))

		filelist = du.list_all_input_file(root_dir + '/npy/hour_avg/Y')
		filelist.sort()
		for i, filename in enumerate(filelist):
			avg_array = du.load_array(root_dir + '/npy/hour_avg/Y/' + filename)
			avg_array = avg_array[:, :, grid_start:grid_stop, grid_start:grid_stop, -1, np.newaxis]  # only network activity
			du.save_array(avg_array, y_target_path + '/hour_avg_' + str(i))

	def _task_4():
		'''
			X: past one hour
			Y: next hour's min value
		'''
		x_target_path = './npy/final/hour_min/training/X'
		y_target_path = './npy/final/hour_min/training/Y'
		if not os.path.exists(x_target_path):
			os.makedirs(x_target_path)
		if not os.path.exists(y_target_path):
			os.makedirs(y_target_path)

		filelist = du.list_all_input_file(root_dir + '/npy/hour_min/X')
		filelist.sort()

		for i, filename in enumerate(filelist):
			if filename != 'training_raw_data.npy':
				data_array = du.load_array(root_dir + '/npy/hour_min/X/' + filename)
				data_array = data_array[:, :, grid_start:grid_stop, grid_start:grid_stop, -1, np.newaxis]  # only network activity
				print('saving array shape:{}'.format(data_array.shape))
				du.save_array(data_array, x_target_path + '/hour_min_' + str(i))

		filelist = du.list_all_input_file(root_dir + '/npy/hour_min/Y')
		filelist.sort()
		for i, filename in enumerate(filelist):
			min_array = du.load_array(root_dir + '/npy/hour_min/Y/' + filename)
			min_array = min_array[:, :, grid_start:grid_stop, grid_start:grid_stop, -1, np.newaxis]  # only network activity
			du.save_array(min_array, y_target_path + '/hour_min_' + str(i))

	def _task_5():
		'''
			X: past one hour
			Y: next hour's min avg max internet traffic
			for multi task learning
		'''

		_task_4()
		_task_3()
		_task_1()
		x_target_path = './npy/final/hour_min_avg_max/training/X'
		y_target_path = './npy/final/hour_min_avg_max/training/Y'
		if not os.path.exists(x_target_path):
			os.makedirs(x_target_path)
		if not os.path.exists(y_target_path):
			os.makedirs(y_target_path)

		max_X, max_Y = get_X_and_Y_array(task_num=1)
		min_X, min_Y = get_X_and_Y_array(task_num=4)
		avg_X, avg_Y = get_X_and_Y_array(task_num=3)
		min_avg_max_Y = np.zeros([max_Y.shape[0], max_Y.shape[1], max_Y.shape[2], max_Y.shape[3], 3])

		for i in range(max_Y.shape[0]):
			for j in range(max_Y.shape[1]):
				for row in range(max_Y.shape[2]):
					for col in range(max_Y.shape[3]):
						# print('min:{} avg:{} max:{}'.format(min_Y[i, j, row, col, 0], avg_Y[i, j, row, col, 0], max_Y[i, j, row, col, 0]))
						min_avg_max_Y[i, j, row, col, 0] = min_Y[i, j, row, col, 0]
						min_avg_max_Y[i, j, row, col, 1] = avg_Y[i, j, row, col, 0]
						min_avg_max_Y[i, j, row, col, 2] = max_Y[i, j, row, col, 0]

		du.save_array(max_X, x_target_path + '/min_avg_max_X')
		du.save_array(min_avg_max_Y, y_target_path + '/min_avg_max_Y')

	def _task_6():
		'''
			X: past one hour
			Y: next 10 minutes traffic level
		'''
		def num_2_vector(input_Y, class_number=10):
			def sigmoid(input_array):
				return 1 / (1 + np.exp(input_array))
			# input_Y_shape = input_Y.shape
			# print(input_Y_shape)
			input_Y = feature_scaling(input_Y, feature_range=(0.1, 255))
			input_Y = sigmoid(input_Y)
			input_Y = feature_scaling(input_Y, feature_range=(0, 10))
			input_Y = np.floor(input_Y)
			print_Y_array(input_Y)
			return input_Y

		# _task_2()
		x_target_path = './npy/final/10_minutes_level/training/X'
		y_target_path = './npy/final/10_minutes_level/training/Y'
		if not os.path.exists(x_target_path):
			os.makedirs(x_target_path)
		if not os.path.exists(y_target_path):
			os.makedirs(y_target_path)

		X, Y = get_X_and_Y_array(task_num=2)
		Y = num_2_vector(Y)

		# du.save_array(X, x_target_path + '/10_minutes_X')
		# du.save_array(Y, y_target_path + '/10_minutes_Y')

	_task_6()


def get_X_and_Y_array(task_num=1):

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
		# X_array = feature_scaling(X_array)
		# Y_array = feature_scaling(Y_array)
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
		# X_array = feature_scaling(X_array)
		# Y_array = feature_scaling(Y_array)
		return X_array, Y_array

	def task_3():
		'''
		X: past one hour
		Y: next hour's avg value
		'''
		x_dir = './npy/final/hour_avg/training/X/'
		y_dir = './npy/final/hour_avg/training/Y/'
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
		# X_array = feature_scaling(X_array)
		# Y_array = feature_scaling(Y_array)
		X_array = X_array[0: -1]  # important!!
		Y_array = Y_array[1:]  # important!! Y should shift 10 minutes
		return X_array, Y_array

	def task_4():
		'''
		X: past one hour
		Y: next hour's min value
		'''
		x_dir = './npy/final/hour_min/training/X/'
		y_dir = './npy/final/hour_min/training/Y/'
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
		# X_array = feature_scaling(X_array)
		# Y_array = feature_scaling(Y_array)
		X_array = X_array[0: -1]  # important!!
		Y_array = Y_array[1:]  # important!! Y should shift 10 minutes
		return X_array, Y_array

	def task_5():
		'''
			X: past one hour
			Y: next hour's min avg max network traffic
			for multi task learning
		'''
		x_dir = './npy/final/hour_min_avg_max/training/X/'
		y_dir = './npy/final/hour_min_avg_max/training/Y/'
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
		# X_array = feature_scaling(X_array)
		# Y_array = feature_scaling(Y_array)
		X_array = X_array[0: -1]  # important!!
		Y_array = Y_array[1:]  # important!! Y should shift 10 minutes
		return X_array, Y_array

	def task_6():
		'''
			X: past one hour
			Y: next 10 minutes traffic level
		'''
		x_dir = './npy/final/10_minutes_level/training/X/'
		y_dir = './npy/final/10_minutes_level/training/Y/'
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
		# X_array = feature_scaling(X_array)
		# Y_array = feature_scaling(Y_array)
		return X_array, Y_array

	if task_num == 1:
		func = task_1()
	elif task_num == 2:
		func = task_2()
	elif task_num == 3:
		func = task_3()
	elif task_num == 4:
		func = task_4()
	elif task_num == 5:
		func = task_5()
	elif task_num == 6:
		func = task_6()
	else:
		func = None

	return func


def feature_scaling(input_datas, feature_range=(0.1, 255)):
	# print(input_datas.shape)
	input_shape = input_datas.shape
	input_datas = input_datas.reshape(-1, 1)
	# print(np.amin(input_datas))
	min_max_scaler = preprocessing.MinMaxScaler(feature_range=feature_range)
	output = min_max_scaler.fit_transform(input_datas)
	output = output.reshape(input_shape)
	return output


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
	X_array, Y_array = get_X_and_Y_array(task_num=5)
	Y_array = Y_array[:, :, 10:13, 10:13, :]
	X_array = feature_scaling(X_array)
	Y_array = feature_scaling(Y_array)

	# X_array_2, Y_array_2 = get_X_and_Y_array(task_num=6)
	# Y_array_2 = Y_array_2[:, :, 10:13, 10:13, :]
	# parameter
	input_data_shape = [X_array.shape[1], X_array.shape[2], X_array.shape[3], X_array.shape[4]]
	output_data_shape = [Y_array.shape[1], Y_array.shape[2], Y_array.shape[3], 1]
	model_path = {
		'reload_path': '/home/mldp/ML_with_bigdata/CNN_RNN/output_model/CNN_RNN_test.ckpt',
		'save_path': '/home/mldp/ML_with_bigdata/CNN_RNN/output_model/CNN_RNN_test.ckpt'
	}
	hyper_config = CNN_RNN_config.HyperParameterConfig()
	cnn_rnn = CNN_RNN(input_data_shape, output_data_shape, hyper_config)
	cnn_rnn.create_MTL_task(X_array, Y_array[:, :, :, :, 0, np.newaxis], 'min_traffic')
	cnn_rnn.create_MTL_task(X_array, Y_array[:, :, :, :, 1, np.newaxis], 'avg_traffic')
	cnn_rnn.create_MTL_task(X_array, Y_array[:, :, :, :, 2, np.newaxis], 'max_traffic')
	del X_array, Y_array
	# cnn_rnn.create_MTL_task(X_array_2, Y_array_2[:, :, :, :, 0, np.newaxis], '10_mins', 'cross_entropy')
	# del X_array_2, Y_array_2

	# cnn_rnn.start_train(model_path, reload=False)
	cnn_rnn.start_MTL_train(model_path, reload=False)


def grid_search():
	X_array, Y_array = get_X_and_Y_array(task_num=5)
	Y_array = Y_array[:, :, 10:13, 10:13, :]
	X_array = feature_scaling(X_array)
	Y_array = feature_scaling(Y_array)
	model_path = '/home/mldp/ML_with_bigdata/CNN_RNN/output_model/CNN_RNN.ckpt'
	gridsearcg = CNN_RNN_config.GridSearch(model_path, X_array, Y_array)
	gridsearcg.search_learning_rate()

if __name__ == '__main__':
	# prepare_training_data()
	# train()
	grid_search()
	
