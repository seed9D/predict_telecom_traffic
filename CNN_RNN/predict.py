from CNN_RNN import CNN_RNN
import CNN_RNN_config
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
from sklearn import preprocessing
import os
import sys
from datetime import datetime
sys.path.append('/home/mldp/ML_with_bigdata')
import data_utility as du

root_dir = '/home/mldp/ML_with_bigdata'
y_scalar = None

def set_time_zone(timestamp):

	UTC_timezone = pytz.timezone('UTC')
	Mi_timezone = pytz.timezone('Europe/Rome')
	date_time = datetime.utcfromtimestamp(float(timestamp))
	date_time = date_time.replace(tzinfo=UTC_timezone)
	date_time = date_time.astimezone(Mi_timezone)
	return date_time


def date_time_covert_to_str(date_time):
	return date_time.strftime('%m-%d %H')


def plot_predict_vs_real(real, predict):
	# print('real shape:{} predict shape:{}'.format(real.shape, predict.shape))
	x_ragne = real.shape[0]

	def __plot(plot_1, plot_2):
		x = range(len(plot_1['time_str']))
		fig = plt.figure()
		ax1 = fig.add_subplot(211)
		ax2 = fig.add_subplot(212)

		plt.xlabel('time sequence')
		plt.ylabel('activity strength')

		ax1.plot(x, plot_1['real'], label='real', marker='.')
		ax1.plot(x, plot_1['predict'], label='predict', marker='.')
		ax1.set_xticks(list(range(0, x_ragne, 12)))
		ax1.set_xticklabels(plot_1['time_str'][0:x_ragne:12], rotation=45)
		ax1.set_title(plot_1['grid_id'])
		ax1.grid()
		# ax1.set_ylabel('time sequence')
		# ax1.set_xlabel('activity strength')
		ax1.legend()

		ax2.plot(x, plot_2['real'], label='real', marker='.')
		ax2.plot(x, plot_2['predict'], label='predict', marker='.')
		ax2.set_xticks(list(range(0, x_ragne, 12)))
		ax2.set_xticklabels(plot_2['time_str'][0:x_ragne:12], rotation=45)
		ax2.set_title(plot_2['grid_id'])
		ax2.grid()
		# ax2.set_ylabel('time sequence')
		# ax2.set_xlabel('activity strength')
		ax2.legend()
		print('\ngrid id {}'.format(plot_1['grid_id']))
		compute_loss_rate(np.array(plot_1['real']), np.array(plot_1['predict']))
		print('\ngrid id {}'.format(plot_2['grid_id']))
		compute_loss_rate(np.array(plot_2['real']), np.array(plot_2['predict']))

	def __get_plot_data(real, predict, *plt_row_col):
		plot_1_row, plot_1_col, plot_2_row, plot_2_col = plt_row_col
		plot_1 = {
			'grid_id': int(real[0, 0, plot_1_row, plot_1_col, 0]),
			'predict': [],
			'real': [],
			'time_str': []
		}
		plot_2 = {
			'grid_id': int(real[0, 0, plot_2_row, plot_2_col, 0]),
			'predict': [],
			'real': [],
			'time_str': []
		}

		for i in range(x_ragne):
			for j in range(real.shape[1]):
				plot_1['real'].append(real[i, j, plot_1_row, plot_1_col, 2, np.newaxis])
				plot_1['predict'].append(predict[i, j, plot_1_row, plot_1_col])
				# print('id:{},real:{},predict:{}'.format(plot_1['grid_id'],real[i,j,10,10,2],predict[i,j,10,10]))
				plot_2['real'].append(real[i, j, plot_2_row, plot_2_col, 2, np.newaxis])
				plot_2['predict'].append(predict[i, j, plot_2_row, plot_2_col])
				# print('id: {},real: {},predict: {}'.format(plot_2['grid_id'], real[i, j, plot_2_row, plot_2_col, 2], predict[i, j, plot_2_row, plot_2_col]))

				data_time = set_time_zone(int(real[i, j, plot_1_row, plot_1_col, 1]))
				plot_1['time_str'].append(date_time_covert_to_str(data_time))
				data_time = set_time_zone(int(real[i, j, plot_2_row, plot_2_col, 1]))
				plot_2['time_str'].append(date_time_covert_to_str(data_time))

		__plot(plot_1, plot_2)
	__get_plot_data(real, predict, 0, 0, 0, 1)
	__get_plot_data(real, predict, 2, 1, 2, 2)

	plt.show()


def plot_error_distribution(error_array, error_name):
		error_array = error_array.flatten()
		error_array = pd.DataFrame({error_name: error_array})
		print(error_array.describe())

		plt.figure()
		plt.subplot(211)
		error_array.hist(ax=plt.gca())
		plt.subplot(212)
		error_array.plot(kind='kde', ax=plt.gca(), label='Resudual error')
		plt.legend()
		plt.title('Residual Forecast Error Plots')


def compute_loss_rate(real, predict):
	# print(real.shape, predict.shape)
	# ab_sum = (np.absolute(real - predict).sum()) / real.size
	# print('real shape {} predict shape {}'.format(real.shape, predict.shape))

	def AE(real, predict):
		AE = np.absolute(real - predict)
		AE_mean = AE.mean()
		print('AE:', AE_mean)
		# plot_error_distribution(AE, 'absoulute_error')

	def RMSE(real, predict):
		MSE = (real - predict) ** 2
		RMSE = np.sqrt(MSE.mean())

		print('RMSE:', RMSE)
		# plot_error_distribution(MSE, 'square error')

	def MAPE(real, predict):
		''' # find the zero index
		zero_tuple = np.where(real == 0)
		zero_array = np.array(zero_tuple)
		zero_array = np.transpose(zero_array, (1, 0))
		for _zero_array in zero_array:
			real[_zero_array] = 1
		'''
		mean_real = real.mean()
		AE = np.absolute(real - predict)
		MAPE = np.divide(AE, mean_real)
		MAPE_mean = MAPE.mean()
		print('Mean accuracy:{:.4f} MAPE:{:.4f}'.format(1 - MAPE_mean, MAPE_mean))
		# plot_error_distribution(MAPE, 'mean absoulute percentage error')

	AE(real, predict)
	RMSE(real, predict)
	MAPE(real, predict)


def prepare_predict_data():
	'''
		generate the predict data from original npy format
	'''
	grid_start = 45
	grid_stop = 60

	def _task_1():
		'''
			X: past one hour
			Y: next hour's max value
		'''
		x_target_path = './npy/final/hour_max/testing/X'
		y_target_path = './npy/final/hour_max/testing/Y'
		if not os.path.exists(x_target_path):
			os.makedirs(x_target_path)
		if not os.path.exists(y_target_path):
			os.makedirs(y_target_path)

		filelist = du.list_all_input_file(root_dir + '/npy/hour_max/X')
		filelist.sort()
		for i, filename in enumerate(filelist):
			if filename != 'training_raw_data.npy':
				data_array = du.load_array(root_dir + '/npy/hour_max/X/' + filename)

				data_array = data_array[:, :, grid_start:grid_stop, grid_start:grid_stop, (0, 1, -1)]
				print('saving array shape:', data_array.shape)
				du.save_array(data_array, x_target_path + '/hour_max_' + str(i))

				# prepare y
				filelist = du.list_all_input_file(root_dir + '/npy/hour_max/Y')
				filelist.sort()
				for i, filename in enumerate(filelist):
					max_array = du.load_array(root_dir + '/npy/hour_max/Y/' + filename)

					max_array = max_array[:, :, grid_start:grid_stop, grid_start:grid_stop, (0, 1, -1)]  # only network activity
					du.save_array(max_array, y_target_path + '/hour_max_' + str(i))

	def _task_2():
		'''
		rolling 10 minutes among timeflows
			X: past one hour
			Y: next 10 minutes value
		'''
		# check target dir exist
		x_target_path = './npy/final/roll_10/testing/X'
		y_target_path = './npy/final/roll_10/testing/Y'
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

			data_array = data_array[:, :, grid_start:grid_stop, grid_start:grid_stop, (0, 1, -1)]
			print('saving  array shape:{}'.format(data_array.shape))
			du.save_array(data_array, x_target_path + '/X_' + str(i))

		for i, filename in enumerate(filelist_Y):
			data_array = du.load_array(root_dir + '/npy/npy_roll/Y/' + filename)

			data_array = data_array[:, :, grid_start:grid_stop, grid_start:grid_stop, (0, 1, -1)]  # only network activity
			print(data_array[0, 0, 20, 20, 0])
			print('saving  array shape:{}'.format(data_array.shape))
			du.save_array(data_array, y_target_path + '/Y_' + str(i))

	def _task_3():
		'''
			X: past one hour
			Y: next hour's avg value
		'''
		x_target_path = './npy/final/hour_avg/testing/X'
		y_target_path = './npy/final/hour_avg/testing/Y'
		if not os.path.exists(x_target_path):
			os.makedirs(x_target_path)
		if not os.path.exists(y_target_path):
			os.makedirs(y_target_path)

		filelist = du.list_all_input_file(root_dir + '/npy/hour_avg/X')
		filelist.sort()
		for i, filename in enumerate(filelist):
			if filename != 'training_raw_data.npy':
				data_array = du.load_array(root_dir + '/npy/hour_avg/X/' + filename)

				data_array = data_array[:, :, grid_start:grid_stop, grid_start:grid_stop, (0, 1, -1)]
				print('saving array shape:', data_array.shape)
				du.save_array(data_array, x_target_path + '/hour_avg_' + str(i))

				# prepare y
				filelist = du.list_all_input_file(root_dir + '/npy/hour_avg/Y')
				filelist.sort()
				for i, filename in enumerate(filelist):
					avg_array = du.load_array(root_dir + '/npy/hour_avg/Y/' + filename)
					avg_array = avg_array[:, :, grid_start:grid_stop, grid_start:grid_stop, (0, 1, -1)]  # only network activity					avg_array = avg_array[:, :, grid_start:65, grid_start:65, (0, 1, -1)]  # only network activity
					du.save_array(avg_array, y_target_path + '/hour_avg_' + str(i))

	def _task_4():
		'''
			X: past one hour
			Y: next hour's min value
		'''
		x_target_path = './npy/final/hour_min/testing/X'
		y_target_path = './npy/final/hour_min/testing/Y'
		if not os.path.exists(x_target_path):
			os.makedirs(x_target_path)
		if not os.path.exists(y_target_path):
			os.makedirs(y_target_path)
		filelist = du.list_all_input_file(root_dir + '/npy/hour_min/X')
		filelist.sort()

		for i, filename in enumerate(filelist):
			if filename != 'training_raw_data.npy':
				data_array = du.load_array(root_dir + '/npy/hour_min/X/' + filename)
				data_array = data_array[:, :, grid_start:grid_stop, grid_start:grid_stop, (0, 1, -1)]  # only network activity
				print('saving array shape:{}'.format(data_array.shape))
				du.save_array(data_array, x_target_path + '/hour_min_' + str(i))

		filelist = du.list_all_input_file(root_dir + '/npy/hour_min/Y')
		filelist.sort()
		for i, filename in enumerate(filelist):
			min_array = du.load_array(root_dir + '/npy/hour_min/Y/' + filename)
			min_array = min_array[:, :, grid_start:grid_stop, grid_start:grid_stop, (0, 1, -1)]  # only network activity
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
		x_target_path = './npy/final/hour_min_avg_max/testing/X'
		y_target_path = './npy/final/hour_min_avg_max/testing/Y'
		if not os.path.exists(x_target_path):
			os.makedirs(x_target_path)
		if not os.path.exists(y_target_path):
			os.makedirs(y_target_path)

		max_X, max_Y = get_X_and_Y_array(task_num=1)
		min_X, min_Y = get_X_and_Y_array(task_num=4)
		avg_X, avg_Y = get_X_and_Y_array(task_num=3)
		min_avg_max_Y = np.zeros([max_Y.shape[0], max_Y.shape[1], max_Y.shape[2], max_Y.shape[3], 5])  # grid_id timestamp, min, avg, max

		for i in range(max_Y.shape[0]):
			for j in range(max_Y.shape[1]):
				for row in range(max_Y.shape[2]):
					for col in range(max_Y.shape[3]):
						# print('min:{} avg:{} max:{}'.format(min_Y[i, j, row, col, 0], avg_Y[i, j, row, col, 0], max_Y[i, j, row, col, 0]))
						min_avg_max_Y[i, j, row, col, 0] = min_Y[i, j, row, col, 0]
						min_avg_max_Y[i, j, row, col, 1] = min_Y[i, j, row, col, 1]

						min_avg_max_Y[i, j, row, col, 2] = min_Y[i, j, row, col, -1]
						min_avg_max_Y[i, j, row, col, 3] = avg_Y[i, j, row, col, -1]
						min_avg_max_Y[i, j, row, col, 4] = max_Y[i, j, row, col, -1]
		du.save_array(max_X, x_target_path + '/min_avg_max_X')
		du.save_array(min_avg_max_Y, y_target_path + '/min_avg_max_Y')

	def _task_6():
		'''
			X: past one hour
			Y: next 10 minutes traffic level
		'''
		_task_2()
		x_target_path = './npy/final/10_minutes_level/testing/X'
		y_target_path = './npy/final/10_minutes_level/testing/Y'
		if not os.path.exists(x_target_path):
			os.makedirs(x_target_path)
		if not os.path.exists(y_target_path):
			os.makedirs(y_target_path)

		X, Y = get_X_and_Y_array(task_num=2)
		Y = feature_scaling(Y, feature_range=(1, 10))  # 10 interval
		Y = np.floor(Y)  # 10 level
		du.save_array(X, x_target_path + '/10_minutes_X')
		du.save_array(Y, y_target_path + '/10_minutes_Y')
	_task_6()


def get_X_and_Y_array(task_num=1):

	def task_1():
		'''
		X: past one hour
		Y: next hour's max value
		'''
		x_dir = './npy/final/hour_max/testing/X/'
		y_dir = './npy/final/hour_max/testing/Y/'

		x_data_list = du.list_all_input_file(x_dir)
		x_data_list.sort()
		y_data_list = du.list_all_input_file(y_dir)
		y_data_list.sort()

		X_array_list = []
		for filename in x_data_list:
			X_array_list.append(du.load_array(x_dir + filename))
		X_array = np.concatenate(X_array_list, axis=0)
		# X_array = X_array[:, :, 0:21, 0:21, :]
		del X_array_list

		Y_array_list = []
		for filename in y_data_list:
			Y_array_list.append(du.load_array(y_dir + filename))
		Y_array = np.concatenate(Y_array_list, axis=0)
		del Y_array_list

		# new_X_array = feature_scaling(X_array[:, :, :, :, -1, np.newaxis])
		# new_Y_array = feature_scaling(Y_array[:, :, :, :, -1, np.newaxis])
		# X_array = _copy(X_array, new_X_array)
		# Y_array = _copy(Y_array, new_Y_array)
		X_array = X_array[0: -1]  # important!!
		Y_array = Y_array[1:]  # important!! Y should shift 10 minutes
		return X_array, Y_array

	def task_2():
		'''
		rolling 10 minutes among timeflows
			X: past one hour
			Y: next 10 minutes value
		'''
		x_dir = './npy/final/roll_10/testing/X/'
		y_dir = './npy/final/roll_10/testing/Y/'
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
		# new_X_array = feature_scaling(X_array)
		# new_Y_array = feature_scaling(Y_array)
		# X_array = _copy(X_array, new_X_array)
		# Y_array = _copy(Y_array, new_Y_array)
		return X_array, Y_array

	def task_3():
		'''
		X: past one hour
		Y: next hour's avg value
		'''
		x_dir = './npy/final/hour_avg/testing/X/'
		y_dir = './npy/final/hour_avg/testing/Y/'

		x_data_list = du.list_all_input_file(x_dir)
		x_data_list.sort()
		y_data_list = du.list_all_input_file(y_dir)
		y_data_list.sort()

		X_array_list = []
		for filename in x_data_list:
			X_array_list.append(du.load_array(x_dir + filename))

		X_array = np.concatenate(X_array_list, axis=0)
		# X_array = X_array[:, :, 0:21, 0:21, :]
		del X_array_list

		Y_array_list = []
		for filename in y_data_list:
			Y_array_list.append(du.load_array(y_dir + filename))
		Y_array = np.concatenate(Y_array_list, axis=0)
		del Y_array_list

		# new_X_array = feature_scaling(X_array[:, :, :, :, -1, np.newaxis])
		# new_Y_array = feature_scaling(Y_array[:, :, :, :, -1, np.newaxis])
		# X_array = _copy(X_array, new_X_array)
		# Y_array = _copy(Y_array, new_Y_array)
		X_array = X_array[0: -1]  # important!!
		Y_array = Y_array[1:]  # important!! Y should shift 10 minutes
		return X_array, Y_array

	def task_4():
		'''
		X: past one hour
		Y: next hour's min value
		'''
		x_dir = './npy/final/hour_min/testing/X/'
		y_dir = './npy/final/hour_min/testing/Y/'
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
		X_array = X_array[0: -1]  # important!!
		Y_array = Y_array[1:]  # important!! Y should shift 10 minutes
		return X_array, Y_array

	def task_5():
		'''
			X: past one hour
			Y: next hour's min avg max network traffic
			for multi task learning
		'''
		x_dir = './npy/final/hour_min_avg_max/testing/X/'
		y_dir = './npy/final/hour_min_avg_max/testing/Y/'
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
		x_dir = './npy/final/10_minutes_level/testing/X/'
		y_dir = './npy/final/10_minutes_level/testing/Y/'
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


def predict_train(cnn_rnn, X_array, Y_array, model_path):
	'''
	see overall performance
	'''
	print(Y_array.shape)

	_, predict_y = cnn_rnn.start_predict(
		X_array[:, :, :, :, 2, np.newaxis],
		Y_array[:, :, :, :, 2, np.newaxis],
		model_path)
	# compute_loss_rate(X_array[1:, :, :, :, 2, np.newaxis], predict_y)
	compute_loss_rate(Y_array[:, :, :, :, 2, np.newaxis], predict_y)
	plot_predict_vs_real(Y_array, predict_y)


def predict_MTL_train(cnn_rnn, X_array, Y_array, model_path):
	# print(Y_array.shape)

	prediction_min, prediction_avg, prediction_max = cnn_rnn.start_MTL_predict(
		X_array[:, :, :, :, 2:],
		Y_array[:, :, :, :, 2:],
		model_path)
	real_min = Y_array[:, :, :, :, 2, np.newaxis]
	real_avg = Y_array[:, :, :, :, 3, np.newaxis]
	real_max = Y_array[:, :, :, :, 4, np.newaxis]
	print(prediction_min.shape, real_min.shape)
	''' unfeature scaling'''
	predict_y = np.concatenate((prediction_min, prediction_avg, prediction_max, real_min, real_avg, real_max), axis=-1)
	predict_y = unfeature_scaling(predict_y)
	new_Y_array = unfeature_scaling(Y_array[:, :, :, :, 2:])
	Y_array = copy(Y_array, new_Y_array)

	print('-' * 20, 'task min:', '-' * 20)
	compute_loss_rate(Y_array[:, :, :, :, 2, np.newaxis], predict_y[:, :, :, :, 0, np.newaxis])
	plot_predict_vs_real(Y_array[:, :, :, :, (0, 1, 2)], predict_y[:, :, :, :, 0, np.newaxis])
	print('-' * 30)
	print('-' * 20, 'task avg:', '-' * 20)
	compute_loss_rate(Y_array[:, :, :, :, 3, np.newaxis], predict_y[:, :, :, :, 1, np.newaxis])
	plot_predict_vs_real(Y_array[:, :, :, :, (0, 1, 3)], predict_y[:, :, :, :, 1, np.newaxis])
	print('-' * 30)
	print('-' * 20, 'task max:', '-' * 20)
	compute_loss_rate(Y_array[:, :, :, :, 4, np.newaxis], predict_y[:, :, :, :, 2, np.newaxis])
	plot_predict_vs_real(Y_array[:, :, :, :, (0, 1, 4)], predict_y[:, :, :, :, 2, np.newaxis])
	print('-' * 30)


def unfeature_scaling(input_datas):
	input_shape = input_datas.shape
	input_datas = input_datas.reshape(-1, 1)
	output = y_scalar.inverse_transform(input_datas)
	output = output.reshape(input_shape)
	return output


def feature_scaling(input_datas):

	input_shape = input_datas.shape
	input_datas = input_datas.reshape(-1, 1)
	min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0.1, 255))
	output = min_max_scaler.fit_transform(input_datas)
	output = output.reshape(input_shape)
	return output, min_max_scaler


def copy(old, new):
	# print(old.shape, new.shape)
	for i in range(old.shape[0]):
		for j in range(old.shape[1]):
			for row in range(old.shape[2]):
				for col in range(old.shape[3]):
					old[i, j, row, col, 2:] = new[i, j, row, col]  # internet

	return old


def print_Y_array(Y_array):
	print('Y array shape:{}'.format(Y_array.shape))
	plot_y_list = []
	for i in range(100):
		for j in range(Y_array.shape[1]):
			print(Y_array[i, j, 2, 2])
			plot_y_list.append(Y_array[i, j, 2, 2])
	plt.figure()
	plt.plot(plot_y_list, marker='.')
	plt.show()

# prepare_predict_data()


X_array, Y_array = get_X_and_Y_array(task_num=5)
# print_Y_array(Y_array[:, :, :, :, 2:])

Y_array = Y_array[:, :, 10:13, 10:13, :]
new_X_array, _ = feature_scaling(X_array[:, :, :, :, 2:])
new_Y_array, y_scalar = feature_scaling(Y_array[:, :, :, :, 2:])
X_array = copy(X_array, new_X_array)
Y_array = copy(Y_array, new_Y_array)
del new_X_array, new_Y_array


X_array_train = X_array[0:120]   # should correspond to bathc size
Y_array_train = Y_array[0:120]	 # should correspond to bathc size
X_array_test = X_array[X_array.shape[0] - 120:]  # should correspond to bathc size
Y_array_test = Y_array[Y_array.shape[0] - 120:]  # should correspond to bathc size
del X_array, Y_array


input_data_shape = [X_array_train.shape[1], X_array_train.shape[2], X_array_train.shape[3], 1]
output_data_shape = [Y_array_train.shape[1], Y_array_train.shape[2], Y_array_train.shape[3], 1]
hyper_config = CNN_RNN_config.HyperParameterConfig()

cnn_rnn = CNN_RNN(input_data_shape, output_data_shape, hyper_config)
model_path = {
	'reload_path': '/home/mldp/ML_with_bigdata/CNN_RNN/output_model/CNN_RNN_test.ckpt',
	'save_path': '/home/mldp/ML_with_bigdata/CNN_RNN/output_model/CNN_RNN.ckpt'
}

# predict_train(cnn_rnn, X_array_train, Y_array_train, model_path)
# predict_train(cnn_rnn, X_array_test, Y_array_test, model_path)
cnn_rnn.create_MTL_task(X_array_test, Y_array_test[:, :, :, :, 2, np.newaxis], 'min_traffic')
cnn_rnn.create_MTL_task(X_array_test, Y_array_test[:, :, :, :, 3, np.newaxis], 'avg_traffic')
cnn_rnn.create_MTL_task(X_array_test, Y_array_test[:, :, :, :, 4, np.newaxis], 'max_traffic')
predict_MTL_train(cnn_rnn, X_array_test, Y_array_test, model_path)
predict_MTL_train(cnn_rnn, X_array_train, Y_array_train, model_path)


