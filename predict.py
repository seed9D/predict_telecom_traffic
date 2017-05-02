import CNN_autoencoder as cn
import matplotlib.pyplot as plt
import numpy as np
import pytz
from datetime import datetime
import data_utility as du
from sklearn import preprocessing
import os


def set_time_zone(timestamp):

	UTC_timezone = pytz.timezone('UTC')
	Mi_timezone = pytz.timezone('Europe/Rome')
	date_time = datetime.utcfromtimestamp(float(timestamp))
	date_time = date_time.replace(tzinfo=UTC_timezone)
	date_time = date_time.astimezone(Mi_timezone)
	return date_time


def date_time_covert_to_str(date_time):
	return date_time.strftime('%m-%d %H')


def random_generate_data():
	shape = [200, 6, 40, 40, 1]
	fake = np.random.randint(0, 100, size=shape)
	fake_point = np.random.randn(*shape)
	fake_zero = np.zeros_like(fake)

	return fake + fake_point
	# return np.random.rand(*shape)
	# return np.random.uniform(0, 100, size=shape)


def plot_predict_vs_real2(real, predict):
	print('real shape:{} predict shape:{}'.format(real.shape, predict.shape))
	plot_1_row = 9
	plot_1_col = 10
	plot_2_row = 3
	plot_2_col = 3
	plot_1 = {
		'grid_id': int(real[0, 0, plot_1_row, plot_1_col, 0]),
		'predict': [],
		'real': []
	}
	plot_2 = {
		'grid_id': int(real[0, 0, plot_2_row, plot_2_col, 0]),
		'predict': [],
		'real': []
	}
	x_length = 0
	for i in range(24):
		for j in range(real.shape[1]):
			plot_1['real'].append(real[i, j, plot_1_row, plot_1_col, 0])
			plot_1['predict'].append(predict[i, j, plot_1_row, plot_1_col])
			# print('id:{},real:{},predict:{}'.format(plot_1['grid_id'],real[i,j,10,10,2],predict[i,j,10,10]))
			plot_2['real'].append(real[i, j, plot_2_row, plot_2_col, 0])
			plot_2['predict'].append(predict[i, j, plot_2_row, plot_2_col])
			# print('id: {},real: {},predict: {}'.format(plot_2['grid_id'], real[i, j, plot_2_row, plot_2_col, 2], predict[i, j, plot_2_row, plot_2_col]))
			x_length += 1
	x = range(x_length)
	fig = plt.figure()
	ax1 = fig.add_subplot(211)
	ax2 = fig.add_subplot(212)

	plt.xlabel('time sequence')
	plt.ylabel('activity strength')

	ax1.plot(x, plot_1['real'], label='real', marker='.')
	ax1.plot(x, plot_1['predict'], label='predict', marker='.')
	ax1.grid()
	# ax1.set_ylabel('time sequence')
	# ax1.set_xlabel('activity strength')
	ax1.legend()

	ax2.plot(x, plot_2['real'], label='real', marker='.')
	ax2.plot(x, plot_2['predict'], label='predict', marker='.')
	ax2.grid()
	# ax2.set_ylabel('time sequence')
	# ax2.set_xlabel('activity strength')
	ax2.legend()
	plt.show()


def plot_predict_vs_real(real, predict):
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
				plot_1['real'].append(real[i, j, plot_1_row, plot_1_col, 2])
				plot_1['predict'].append(predict[i, j, plot_1_row, plot_1_col])
				# print('id:{},real:{},predict:{}'.format(plot_1['grid_id'],real[i,j,10,10,2],predict[i,j,10,10]))
				plot_2['real'].append(real[i, j, plot_2_row, plot_2_col, 2])
				plot_2['predict'].append(predict[i, j, plot_2_row, plot_2_col])
				# print('id: {},real: {},predict: {}'.format(plot_2['grid_id'], real[i, j, plot_2_row, plot_2_col, 2], predict[i, j, plot_2_row, plot_2_col]))

				data_time = set_time_zone(int(real[i, j, plot_1_row, plot_1_col, 1]))
				plot_1['time_str'].append(date_time_covert_to_str(data_time))
				data_time = set_time_zone(int(real[i, j, plot_2_row, plot_2_col, 1]))
				plot_2['time_str'].append(date_time_covert_to_str(data_time))

		__plot(plot_1, plot_2)
	__get_plot_data(real, predict, 20, 10, 20, 11)
	__get_plot_data(real, predict, 10, 5, 10, 10)
	

	plt.show()


def compute_loss_rate(real, predict):
	# print(real.shape, predict.shape)
	# ab_sum = (np.absolute(real - predict).sum()) / real.size
	# print('real shape {} predict shape {}'.format(real.shape, predict.shape))
	ab_sum = (np.absolute(real - predict).mean())
	print('AE:', ab_sum)

	rmse_sum = np.sqrt(((real - predict) ** 2).mean())
	print('RMSE:', rmse_sum)


def check_data(predict_array):
	"""
	just plot the data to see
	"""
	plot_1 = {
		'grid_id': int(predict_array[0, 0, 10, 20, 0]),
		'real': [],
		'time_': []
	}

	for i in range(100):
		for j in range(predict_array.shape[1]):

			grid = predict_array[i, j, 10, 20, 0]

			timestamp = predict_array[i, j, 10, 20, 1]
			data_time = set_time_zone(timestamp)
			date_string = date_time_covert_to_str(data_time)
			internet = predict_array[i, j, 10, 20, 2]

			plot_1['real'].append(internet)
			plot_1['time_'].append(date_string)
			print('id:{} time:{} internet:{}'.format(grid, date_string, internet))

	x = range(len(plot_1['time_']))
	plt.xlabel('time sequence')
	plt.ylabel('activity strength')
	plt.title(plot_1['grid_id'])
	# plt.xticks(x,plot_1['time_'])
	plt.grid()
	plt.plot(x, plot_1['real'])
	plt.show()


def prepare_predict_data():
	'''
		generate the predict data from original npy format
	'''
	def task_1():
		'''
			X: past one hour 
			Y: next hour's max value
		'''
		filelist = du.list_all_input_file('./npy/')
		filelist.sort()
		for i, filename in enumerate(filelist):
			if filename != 'training_raw_data.npy':
				data_array = du.load_array('./npy/' + filename)
				# print(' 0 grid id {}'.format(data_array[0, 0, 10, 10, 0]))
				# print(' 60 grid id {}'.format(data_array[0, 0, 70, 70, 0]))
				data_array = data_array[:, :, 40:65, 40:65, (0, 1, -1)]
				print('saving array shape:', data_array.shape)
				du.save_array(data_array, './npy/final/testing_raw_data/' + 'testing_' + str(i))

				# prepare y
				max_array = du.get_MAX_internet_array(data_array[:, :, :, :, -1, np.newaxis])
				new_max_array_shape = [max_array.shape[0], max_array.shape[1], max_array.shape[2], max_array.shape[2], 3]
				new_max_array = np.zeros(new_max_array_shape, dtype=np.float32)
				for index in range(max_array.shape[0]):
					for row in range(max_array.shape[2]):
						for col in range(max_array.shape[3]):
							new_max_array[index, 0, row, col, 0] = data_array[index, 0, row, col, 0]  # gird id
							new_max_array[index, 0, row, col, 1] = data_array[index, 3, row, col, 1]  # timestamp
							new_max_array[index, 0, row, col, 2] = max_array[index, 0, row, col, 0]  # internet
							'''
							print('id:{} date:{} internet:{}'.format(
								new_max_array[index, 0, row, col, 0],
								date_time_covert_to_str(set_time_zone(new_max_array[index, 0, row, col, 1])),
								new_max_array[index, 0, row, col, 2]))
							'''
				du.save_array(new_max_array, './npy/final/testing_raw_data/one_hour_max_value/one_hour_max' + '_' + str(i))
	def task_2():
		'''
		rolling 10 minutes among timeflows
			X: past one hour
			Y: next 10 minutes value
		'''
		# check target dir exist
		x_target_path = './npy/npy_roll/final/predict/X/'
		y_target_path = './npy/npy_roll/final/predict/Y/'
		if not os.path.exists(x_target_path):
			os.makedirs(x_target_path)
		if not os.path.exists(y_target_path):
			os.makedirs(y_target_path)
		filelist_X = du.list_all_input_file('./npy/npy_roll/X/')
		filelist_Y = du.list_all_input_file('./npy/npy_roll/Y/')
		filelist_X.sort()
		filelist_Y.sort()
		for i, filename in enumerate(filelist_X):
			data_array = du.load_array('./npy/npy_roll/X/' + filename)
			data_array = data_array[:, :, 40:65, 40:65, (0, 1, -1)]
			print('saving  array shape:{}'.format(data_array.shape))
			du.save_array(data_array, x_target_path + 'X_' + str(i))

		for i, filename in enumerate(filelist_Y):
			data_array = du.load_array('./npy/npy_roll/Y/' + filename)
			data_array = data_array[:, :, 40:65, 40:65, (0, 1, -1)]  # only network activity
			print(data_array[0, 0, 20, 20, 0])
			print('saving  array shape:{}'.format(data_array.shape))
			du.save_array(data_array, y_target_path + 'Y_' + str(i))
	task_2()

def get_X_and_Y_array():
	def _copy(old, new):
			# print(old.shape, new.shape)
			for i in range(old.shape[0]):
				for j in range(old.shape[1]):
					for row in range(old.shape[2]):
						for col in range(old.shape[3]):
							old[i, j, row, col, -1] = new[i, j, row, col, -1]  # internet

			return old
	def task_1():
		'''
		X: past one hour 
		Y: next hour's max value
		'''
		
		training_data_list = du.list_all_input_file('./npy/final/testing_raw_data/')
		training_data_list.sort()
		X_array_list = []
		for filename in training_data_list:
			X_array_list.append(du.load_array('./npy/final/testing_raw_data/' + filename))

		X_array = np.concatenate(X_array_list, axis=0)
		# X_array = X_array[:, :, 0:21, 0:21, :]
		del X_array_list

		Y_data_list = du.list_all_input_file('./npy/final/testing_raw_data/one_hour_max_value/')
		Y_data_list.sort()
		Y_array_list = []
		for filename in Y_data_list:
			Y_array_list.append(du.load_array('./npy/final/testing_raw_data/one_hour_max_value/' + filename))
		Y_array = np.concatenate(Y_array_list, axis=0)
		del Y_array_list

		new_X_array = feature_scaling(X_array[:, :, :, :, -1, np.newaxis])
		new_Y_array = feature_scaling(Y_array[:, :, :, :, -1, np.newaxis])
		X_array = _copy(X_array, new_X_array)
		Y_array = _copy(Y_array, new_Y_array)
		X_array = X_array[0: -1]  # important!!
		Y_array = Y_array[1:]  # important!! Y should shift 10 minutes
		return X_array, Y_array
	
	def task_2():
		'''
		rolling 10 minutes among timeflows
			X: past one hour
			Y: next 10 minutes value
		'''
		x_target_path = './npy/npy_roll/final/predict/X/'
		y_target_path = './npy/npy_roll/final/predict/Y/'
		X_file_list = du.list_all_input_file(x_target_path)
		Y_file_list = du.list_all_input_file(y_target_path)
		X_file_list.sort()
		Y_file_list.sort()
		X_array_list = []
		Y_array_list = []
		# X array
		for filename in X_file_list:
			X_array_list.append(du.load_array(x_target_path + filename))
		X_array = np.concatenate(X_array_list, axis=0)
		del X_array_list
		# Y array
		for filename in Y_file_list:
			Y_array_list.append(du.load_array(y_target_path + filename))
		Y_array = np.concatenate(Y_array_list, axis=0)
		del Y_array_list
		new_X_array = feature_scaling(X_array)
		new_Y_array = feature_scaling(Y_array)
		X_array = _copy(X_array, new_X_array)
		Y_array = _copy(Y_array, new_Y_array)
		return X_array, Y_array
	return task_2()
def predict_pre_train(CNN, X_array, model_path):
	'''
	see pre train performance 
	'''
	_, predict_y = CNN.predict_data(predict_array[:, :, :, :, 2, np.newaxis], model_path, 'pre_train')
	compute_loss_rate(predict_array[1:, :, :, :, 2, np.newaxis], predict_y)
	plot_predict_vs_real(predict_array[1:], predict_y)


def predict_train(CNN, X_array, Y_array, model_path):
	'''
	see overall performance
	'''
	print(Y_array.shape)

	_, predict_y = CNN.predict_data(
		X_array[:, :, :, :, 2, np.newaxis],
		Y_array[:, :, :, :, 2, np.newaxis],
		model_path,
		'train')
	# compute_loss_rate(X_array[1:, :, :, :, 2, np.newaxis], predict_y)
	compute_loss_rate(Y_array[:, :, :, :, 2, np.newaxis], predict_y)
	plot_predict_vs_real(Y_array, predict_y)


def feature_scaling(input_datas):
	input_shape = input_datas.shape
	input_datas = input_datas.reshape(input_shape[0], -1)
	min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 255))
	output = min_max_scaler.fit_transform(input_datas)
	output = output.reshape(input_shape)
	return output

# prepare_predict_data()
# fake_data = random_generate_data()

X_array, Y_array = get_X_and_Y_array()
X_array_train = X_array[0:200]
Y_array_train = Y_array[0:200]
X_array_test = X_array[X_array.shape[0] - 200:]
Y_array_test = Y_array[Y_array.shape[0] - 200:]
del X_array, Y_array 
# print(Y_array[0, 0
# , 10, 20, -1])
# predict_array = predict_array[0:400]
'''
# network_parameter = {'conv1': 128, 'conv2': 64, 'conv3': 32, 'fc1': 512, 'fc2': 512}
# network_parameter = {'conv1': 64, 'conv2': 64, 'conv3': 32, 'conv4': 32, 'fc1': 512}  # hidden layer
'''
network_parameter = {
		'conv1': 32,
		'conv2': 32,
		'conv3': 32,
		'conv4': 32,
		'conv5': 32,
		'conv6': 32,
		'conv7': 32,
		'fc1': 512}  # hidden layer
data_shape = [X_array_train.shape[1], X_array_train.shape[2], X_array_train.shape[3], 1]
predict_CNN = cn.CNN_autoencoder(*data_shape, **network_parameter)
model_path = {
	'pretrain_save': '/home/mldp/ML_with_bigdata/output_model/AE_pre_64_32_32_test.ckpt',
	'pretrain_reload': '/home/mldp/ML_with_bigdata/output_model/AE_pre_64_32_32_test.ckpt',
	'reload': '/home/mldp/ML_with_bigdata/output_model/roll_10_train_32_32_32_32_32_512_v2.ckpt',
	'save': '/home/mldp/ML_with_bigdata/output_model/train_64_64_32_32_32_512.ckpt'
}
predict_CNN.set_training_data(X_array_train[:, :, :, :, 2, np.newaxis], Y_array_train[:, :, :, :, 2, np.newaxis])  # internet traffic

# predict_pre_train(predict_CNN, predict_array, model_path)
predict_train(predict_CNN, X_array_train, Y_array_train, model_path)
predict_train(predict_CNN, X_array_test, Y_array_test, model_path)


# _, fake_predict_y = predict_CNN.predict_data(fake_data, model_path)
# plot_predict_vs_real2(fake_data[0:-1], fake_predict_y)
# compute_loss_rate(fake_data[0:-1], fake_predict_y)
# check_data(predict_array)
