import CNN_autoencoder as cn
import matplotlib.pyplot as plt
import numpy as np
import pytz
from datetime import datetime
import data_utility as du


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

	# grid_id_1 = real[0,0,10,10,0]
	# grid_id_2 = real[0,0,30,20,0]
	plot_1_row = 9
	plot_1_col = 5
	plot_2_row = 10
	plot_2_col = 20
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
	time_x = []
	for i in range(300):
		for j in range(real.shape[1]):
			plot_1['real'].append(real[i, j, plot_1_row, plot_1_col, 2])
			plot_1['predict'].append(predict[i, j, plot_1_row, plot_1_col])
			# print('id:{},real:{},predict:{}'.format(plot_1['grid_id'],real[i,j,10,10,2],predict[i,j,10,10]))
			plot_2['real'].append(real[i, j, plot_2_row, plot_2_col, 2])
			plot_2['predict'].append(predict[i, j, plot_2_row, plot_2_col])
			# print('id: {},real: {},predict: {}'.format(plot_2['grid_id'], real[i, j, plot_2_row, plot_2_col, 2], predict[i, j, plot_2_row, plot_2_col]))

			data_time = set_time_zone(int(real[i, j, plot_1_row, plot_1_col, 1]))
			time_x.append(date_time_covert_to_str(data_time))
	'''
	for time_str in time_x:
		print(time_str)
	'''
	x = range(len(time_x))
	fig = plt.figure()
	ax1 = fig.add_subplot(211)
	ax2 = fig.add_subplot(212)

	plt.xlabel('time sequence')
	plt.ylabel('activity strength')

	ax1.plot(x, plot_1['real'], label='real', marker='.')
	ax1.plot(x, plot_1['predict'], label='predict', marker='.')
	ax1.set_xticks(list(range(0, 300, 12)))
	ax1.set_xticklabels(time_x[0:300:12], rotation=45)
	ax1.set_title(plot_1['grid_id'])
	ax1.grid()
	# ax1.set_ylabel('time sequence')
	# ax1.set_xlabel('activity strength')
	ax1.legend()

	ax2.plot(x, plot_2['real'], label='real', marker='.')
	ax2.plot(x, plot_2['predict'], label='predict', marker='.')
	ax2.set_xticks(list(range(0, 300, 12)))
	ax2.set_xticklabels(time_x[0:300:12], rotation=45)
	ax2.set_title(plot_2['grid_id'])
	ax2.grid()
	# ax2.set_ylabel('time sequence')
	# ax2.set_xlabel('activity strength')
	ax2.legend()
	print('grid error:')
	compute_loss_rate(np.array(plot_1['real']), np.array(plot_1['predict']))
	compute_loss_rate(np.array(plot_2['real']), np.array(plot_2['predict']))
	plt.show()


def compute_loss_rate(real, predict):
	# print(real.shape, predict.shape)
	# ab_sum = (np.absolute(real - predict).sum()) / real.size
	print('real shape {} predict shape {}'.format(real.shape, predict.shape))
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


def get_X_and_Y_array():
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
	return X_array, Y_array


def predict_pre_train(CNN, predict_array, model_path):
	_, predict_y = CNN.predict_data(predict_array[:, :, :, :, 2, np.newaxis], model_path, 'pre_train')
	compute_loss_rate(predict_array[1:, :, :, :, 2, np.newaxis], predict_y)
	plot_predict_vs_real(predict_array[1:], predict_y)


def predict_train(CNN, predict_array, Y_array, model_path):
	_, predict_y = CNN.predict_data(
		predict_array[:, :, :, :, 2, np.newaxis],
		Y_array[:, :, :, :, 2, np.newaxis],
		model_path,
		'train')
	# compute_loss_rate(predict_array[1:, :, :, :, 2, np.newaxis], predict_y)
	compute_loss_rate(Y_array[1:, :, :, :, 2, np.newaxis], predict_y)
	plot_predict_vs_real(Y_array[1:], predict_y)


# prepare_predict_data()
# fake_data = random_generate_data()

predict_array, Y_array = get_X_and_Y_array()
predict_array = predict_array[0:400]
Y_array = Y_array[0:400]
# print(Y_array[0, 0, 10, 20, -1])
# predict_array = predict_array[0:400]

network_parameter = {'conv1': 128, 'conv2': 64, 'conv3': 32, 'fc1': 512, 'fc2': 512}
data_shape = [predict_array.shape[1], predict_array.shape[2], predict_array.shape[3], 1]
predict_CNN = cn.CNN_autoencoder(*data_shape, **network_parameter)
model_path = {
	'pretrain_save': '/home/mldp/ML_with_bigdata/output_model/AE_pre_64_32_32_test.ckpt',
	'pretrain_reload': '/home/mldp/ML_with_bigdata/output_model/AE_pre_64_32_32_test.ckpt',
	'reload': '/home/mldp/ML_with_bigdata/output_model/train_test2.ckpt',
	'save': '/home/mldp/ML_with_bigdata/output_model/train_test2.ckpt'
}
predict_CNN.set_training_data(predict_array[:, :, :, :, 2, np.newaxis], Y_array[:, :, :, :, 2, np.newaxis])

# predict_pre_train(predict_CNN, predict_array, model_path)
predict_train(predict_CNN, predict_array, Y_array, model_path)


# _, fake_predict_y = predict_CNN.predict_data(fake_data, model_path)
# plot_predict_vs_real2(fake_data[0:-1], fake_predict_y)
# compute_loss_rate(fake_data[0:-1], fake_predict_y)
# check_data(predict_array)
