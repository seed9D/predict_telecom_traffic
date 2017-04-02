import CNN_autoencoder as cn
import matplotlib.pyplot as plt
import numpy as np
import pytz
from datetime import datetime


def set_time_zone(timestamp):

	UTC_timezone = pytz.timezone('UTC')
	Mi_timezone = pytz.timezone('Europe/Rome')
	date_time = datetime.utcfromtimestamp(float(timestamp))
	date_time = date_time.replace(tzinfo=UTC_timezone)
	date_time = date_time.astimezone(Mi_timezone)
	return date_time


def date_time_covert_to_str(date_time):
	return date_time.strftime('%Y-%m-%d %H:%M:%S')


def plot_predict_vs_real(real, predict):

	# grid_id_1 = real[0,0,10,10,0]
	# grid_id_2 = real[0,0,30,20,0]

	plot_1 = {
		'grid_id': int(real[0, 0, 10, 10, 0]),
		'predict': [],
		'real': []
	}
	plot_2 = {
		'grid_id': int(real[0, 0, 2, 23, 0]),
		'predict': [],
		'real': []
	}
	time_x = []
	for i in range(24):
		for j in range(real.shape[1]):
			plot_1['real'].append(real[i, j, 10, 10, 2])
			plot_1['predict'].append(predict[i, j, 10, 10])
			# print('id:{},real:{},predict:{}'.format(plot_1['grid_id'],real[i,j,10,10,2],predict[i,j,10,10]))
			plot_2['real'].append(real[i, j, 2, 23, 2])
			plot_2['predict'].append(predict[i, j, 2, 23])
			print('id: {},real: {},predict: {}'.format(plot_2['grid_id'], real[i, j, 2, 23, 2], predict[i, j, 2, 23]))

			data_time = set_time_zone(int(real[i, j, 10, 10, 1]))
			time_x.append(date_time_covert_to_str(data_time))

	x = range(len(time_x))
	fig = plt.figure()
	ax1 = fig.add_subplot(211)
	ax2 = fig.add_subplot(212)

	plt.xlabel('time sequence')
	plt.ylabel('activity strength')

	ax1.plot(x, plot_1['real'], label='real', marker='.')
	ax1.plot(x, plot_1['predict'], label='predict', marker='.')
	ax1.set_xticks(x, time_x)
	ax1.set_title(plot_1['grid_id'])
	ax1.grid()
	# ax1.set_ylabel('time sequence')
	# ax1.set_xlabel('activity strength')
	ax1.legend()

	ax2.plot(x, plot_2['real'], label='real', marker='.')
	ax2.plot(x, plot_2['predict'], label='predict', marker='.')
	ax2.set_xticks(x, time_x)
	ax2.set_title(plot_2['grid_id'])
	ax2.grid()
	# ax2.set_ylabel('time sequence')
	# ax2.set_xlabel('activity strength')
	ax2.legend()

	plt.show()


def compute_loss_rate(real, predict):
	# print(real.shape, predict.shape)
	ab_sum = (np.absolute(real - predict).sum()) / real.size
	print('absolute average:', ab_sum)


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

	filelist = cn.list_all_input_file('./npy/')
	filelist.sort()
	for i, filename in enumerate(filelist):
		if filename != 'training_raw_data.npy':
			data_array = cn.load_array('./npy/' + filename)
			data_array = data_array[:, :, 0:40, 0:40, (0, 1, -1)]
			print('savin array shape:', data_array.shape)
			cn.save_array(data_array, './npy/final/testing_raw_data/' + 'testing_' + str(i))


# prepare_predict_data()

training_data_list = cn.list_all_input_file('./npy/final/')
training_data_list.sort()
X_array_list = []
for filename in training_data_list:
	X_array_list.append(cn.load_array('./npy/final/' + filename))
X_array = np.concatenate(X_array_list, axis=0)
del X_array_list


testing_data_list = cn.list_all_input_file('./npy/final/testing_raw_data/')
testing_data_list.sort()
predict_array_list = []
for filename in testing_data_list:
	predict_array_list.append(cn.load_array('./npy/final/testing_raw_data/' + str(filename)))
predict_array = np.concatenate(predict_array_list, axis=0)
del testing_data_list


predict_array = predict_array[predict_array.shape[0] - 200:]

network_parameter = {'conv1': 16, 'conv2': 32, 'conv3': 48}
data_shape = [X_array.shape[1], X_array.shape[2], X_array.shape[3], X_array.shape[4]]
predict_CNN = cn.CNN_autoencoder(*data_shape, **network_parameter)
predict_CNN.set_model_name('/home/mldp/ML_with_bigdata/output_model/CNN_autoencoder_onlyinternet_16_32_48.ckpt', '/home/mldp/ML_with_bigdata/output_model/CNN_autoencoder_onlyinternet_16_32_48.ckpt')
predict_CNN.set_training_data(X_array)
del X_array
_, predict_y = predict_CNN.predict_data(predict_array[:, :, :, :, 2, np.newaxis])

plot_predict_vs_real(predict_array[0:-1], predict_y)
compute_loss_rate(predict_array[0:-1, :, :, :, 2, np.newaxis], predict_y)

# check_data(predict_array)
