import numpy as np
import matplotlib.pyplot as plt
import os
import json
import sys
from sklearn import preprocessing
sys.path.append('/home/mldp/ML_with_bigdata')
import data_utility as du
from multi_task_data import Prepare_Task_Data
from CNN_RNN.CNN_RNN_config import HyperParameterConfig
from CNN_RNN.CNN_RNN import CNN_RNN
root_dir = '/home/mldp/ML_with_bigdata'


def compute_row_col(grid_id):
	grid_row_num = 100
	grid_column_num = 100
	row = 99 - int(grid_id / grid_row_num)  # row mapping to milan grid
	column = grid_id % grid_column_num - 1
	# print(row, column)
	return int(row), int(column)


def comput_grid_id(row, col):
	Y = 99 - row
	X = col + 1
	grid_id = Y * 100 + X
	return grid_id


class Internet_Traffic:

	def __init__(self):
		self.grid_row_num = 100
		self.grid_column_num = 100
		self.input_width = 15  # 15 X 15
		self.output_width = 3  # 3 X 3

		self.scaler = None
		self.X_train = None
		self.Y_train = None
		self.X_test = None
		self.Y_test = None

		# self.set_grid_id(grid_id)

	def get_grid_id(self, row, col):
		Y = 99 - row
		X = col + 1
		grid_id = Y * self.grid_row_num + X
		return grid_id

	def set_grid_id(self, grid_id):
		# row mapping to milan grid
		self.row = 99 - int(grid_id / self.grid_row_num)
		self.column = grid_id % self.grid_column_num - 1  # column mapping to milan grid
		self.input_pair_list, self.output_pair_lsit = self._get_input_and_output_pair(
			self.row, self.column)
		# print(self.row, self.column)
		self._prepare_data()
		'''
		for output_pair in self.output_pair_lsit:
			grid_id = self.get_grid_id(*output_pair)
			print(grid_id)
		'''

	def _get_input_and_output_pair(self, row, col):
		input_half = (self.input_width - 1) // 2
		input_row_range = list(
			range(self.row - input_half, self.row + input_half + 1))
		input_col_range = list(
			range(self.column - input_half, self.column + input_half + 1))

		output_half = (self.output_width - 1) // 2
		output_row_range = list(
			range(self.row - output_half, self.row + output_half + 1))
		output_col_range = list(
			range(self.column - output_half, self.column + output_half + 1))

		input_pair_list = []
		output_pair_lsit = []

		for row in (input_row_range):
			for col in (input_col_range):
				input_pair_list.append((row, col))

		for row in (output_row_range):
			for col in (output_col_range):
				output_pair_lsit.append((row, col))
		return input_pair_list, output_pair_lsit

	def _prepare_data(self):
		Y_center_square_start = (self.input_width - self.output_width) // 2 + 1
		Y_center_square_end = Y_center_square_start + self.output_width
		# print(Y_center_square_start, Y_center_square_end)
		TK = Prepare_Task_Data('./npy/final')

		row_grid_start = self.input_pair_list[0][0]
		row_grid_end = self.input_pair_list[-1][0] + 1
		col_grid_start = self.input_pair_list[0][1]
		col_grid_end = self.input_pair_list[-1][1] + 1
		grid_limit = [(row_grid_start, row_grid_end),
					  (col_grid_start, col_grid_end)]
		# print(grid_limit)
		X_array, Y_array = TK.Task_max_min_avg(grid_limit, generate_data=True)
		# print(X_array.shape, Y_array.shape)
		# print(grid_limit, X_array[0, 0, 0, 0, 0])
		self.X_info = X_array[:, :, :, :, 0:2]  # gird id, timestamp
		self.Y_info = Y_array[:, :, Y_center_square_start: Y_center_square_end,
							  Y_center_square_start: Y_center_square_end, 0:2]  # gird id, timestamp

		X_array, Y_array = X_array[:, :, :, :, -1, np.newaxis], Y_array[
			:, :, Y_center_square_start: Y_center_square_end, Y_center_square_start: Y_center_square_end, 2:]
		_, self.scaler = self._feature_scaling(X_array)
		_, _ = self._feature_scaling(Y_array, self.scaler)

		data_len = X_array.shape[0]
		self.X_train = X_array  # todo
		self.Y_train = Y_array  # todo
		# self.X_train = X_array[: 9 * data_len // 10]
		# self.Y_train = Y_array[: 9 * data_len // 10]

		self.X_test = X_array[9 * data_len // 10:]
		self.Y_test = Y_array[9 * data_len // 10:]
		# print(data_len, self.X_train.shape, self.X_test.shape)

	def _feature_scaling(self, input_data, scaler=None, feature_range=(0.1, 255)):
		input_shape = input_data.shape
		input_data = input_data.reshape(-1, 1)
		if scaler:
			output = scaler.transform(input_data)
		else:
			scaler = preprocessing.MinMaxScaler(feature_range=feature_range)
			output = scaler.fit_transform(input_data)
		return output.reshape(input_shape), scaler

	def _un_feature_scaling(self, input_data, scaler):
		input_shape = input_data.shape
		input_data = input_data.reshape(-1, 1)
		output = scaler.inverse_transform(input_data)
		output = output.reshape(input_shape)
		return output

	def set_NN_mode(self):
		self.hyper_config = hyper_config = HyperParameterConfig()
		self.hyper_config.read_config(file_path=os.path.join(
			root_dir, 'CNN_RNN/result/random_search_0609/_85/config.json'))
		hyper_config.iter_epoch = 2000
		# key_var = hyper_config.get_variable()
		input_data_shape = [self.X_train.shape[1], self.X_train.shape[
			2], self.X_train.shape[3], self.X_train.shape[4]]
		output_data_shape = [self.Y_train.shape[1],
							 self.Y_train.shape[2], self.Y_train.shape[3], 1]

		X_train, _ = self._feature_scaling(self.X_train, self.scaler)
		X_test, _ = self._feature_scaling(self.X_test, self.scaler)
		Y_train, _ = self._feature_scaling(self.Y_train, self.scaler)
		Y_test, _ = self._feature_scaling(self.Y_test, self.scaler)

		self.internet_traffic_model = CNN_RNN(
			input_data_shape, output_data_shape, hyper_config)
		self.internet_traffic_model.create_MTL_task(
			X_train, Y_train[:, :, :, :, 0, np.newaxis], 'min_traffic')
		self.internet_traffic_model.create_MTL_task(
			X_train, Y_train[:, :, :, :, 1, np.newaxis], 'avg_traffic')
		self.internet_traffic_model.create_MTL_task(
			X_train, Y_train[:, :, :, :, 2, np.newaxis], 'max_traffic')

	def run_train(self):
		result_path = '/home/mldp/ML_with_bigdata/offloading/result/temp/'
		model_path = {
			'reload_path': '/home/mldp/ML_with_bigdata/offloading/output_model/CNN_RNN_test.ckpt',
			'save_path': '/home/mldp/ML_with_bigdata/offloading/output_model/CNN_RNN_test.ckpt',
			'result_path': result_path
		}
		# hyper_config = HyperParameterConfig()
		# hyper_config.read_config(file_path=os.path.join(root_dir, 'CNN_RNN/result/random_search_0609/_85/config.json'))
		# cnn_rnn = CNN_RNN(input_data_shape, output_data_shape, hyper_config)

		self.internet_traffic_model.start_MTL_train(model_path, reload=False)

	def run_predict(self):
		def predcit_MTL_train(cnn_rnn, X_array, Y_array, model_path):
			prediction_min, prediction_avg, prediction_max = cnn_rnn.start_MTL_predict(
				X_array,
				Y_array,
				model_path)
			prediction_y = np.concatenate(
				(prediction_min, prediction_avg, prediction_max), axis=-1)
			return prediction_y

		def data_remainder(cnn_rnn, X_array, Y_array, model_path, batch_size):
			# print('batch_size', batch_size)
			array_len = X_array.shape[0]
			# n_batch = array_len // batch_size
			n_remain = array_len % batch_size
			# print('n remain:', n_remain)
			new_x = X_array[array_len - batch_size:]
			new_y = Y_array[array_len - batch_size:]
			# print(new_x.shape, new_y.shape)

			prediction = predcit_MTL_train(cnn_rnn, new_x, new_y, model_path)
			# print(prediction.shape)
			remainder_prediction = prediction[prediction.shape[0] - n_remain:]
			# print(remainder_prediction.shape)
			return remainder_prediction

		# hyper_config.read_config(file_path=os.path.join(root_dir, 'CNN_RNN/result/random_search_0609/_85/config.json'))
		X_train, _ = self._feature_scaling(self.X_train, self.scaler)
		X_test, _ = self._feature_scaling(self.X_test, self.scaler)
		Y_train, _ = self._feature_scaling(self.Y_train, self.scaler)
		Y_test, _ = self._feature_scaling(self.Y_test, self.scaler)

		key_var = self.hyper_config.get_variable()
		batch_size = key_var['batch_size']

		train_len = X_train.shape[0]
		test_len = X_test.shape[0]

		n_train_batch = train_len // batch_size
		n_test_batch = test_len // batch_size

		X_array_train = X_train[: n_train_batch * batch_size]
		Y_array_train = Y_train[: n_train_batch * batch_size]
		X_array_test = X_test[: n_test_batch * batch_size]
		Y_array_test = Y_test[: n_test_batch * batch_size]
		print(X_array_train.shape, Y_array_train.shape)
		print(X_array_test.shape, Y_array_test.shape)
		# input_data_shape = [X_array_train.shape[1], X_array_train.shape[2], X_array_train.shape[3], 1]
		# output_data_shape = [Y_array_train.shape[1], Y_array_train.shape[2], Y_array_train.shape[3], 1]
		# cnn_rnn = CNN_RNN(input_data_shape, output_data_shape, hyper_config)
		model_path = {
			'reload_path': '/home/mldp/ML_with_bigdata/offloading/output_model/CNN_RNN_test.ckpt',
			'save_path': '/home/mldp/ML_with_bigdata/CNN_RNN/output_model/CNN_RNN.ckpt'
		}
		# self.internet_traffic_model.create_MTL_task(X_array_test, Y_array_test[:, :, :, :, 0, np.newaxis], 'min_traffic')
		# self.internet_traffic_model.create_MTL_task(X_array_test, Y_array_test[:, :, :, :, 1, np.newaxis], 'avg_traffic')
		# self.internet_traffic_model.create_MTL_task(X_array_test, Y_array_test[:, :, :, :, 2, np.newaxis], 'max_traffic')

		train_y_prediction = predcit_MTL_train(
			self.internet_traffic_model, X_array_train, Y_array_train, model_path)
		train_remain_predcition = data_remainder(
			self.internet_traffic_model, X_train, Y_train, model_path, batch_size)
		train_y_prediction = np.concatenate(
			(train_y_prediction, train_remain_predcition), axis=0)
		self.train_y_prediction = self._un_feature_scaling(
			train_y_prediction, self.scaler)
		test_y_prediction = predcit_MTL_train(
			self.internet_traffic_model, X_array_test, Y_array_test, model_path)
		test_remain_predcition = data_remainder(
			self.internet_traffic_model, X_test, Y_test, model_path, batch_size)
		test_y_prediction = np.concatenate(
			(test_y_prediction, test_remain_predcition), axis=0)
		self.test_y_prediction = self._un_feature_scaling(
			test_y_prediction, self.scaler)
		# print(train_y_prediction.shape, test_y_prediction.shape)

		# self.plot(self.Y_train, self.train_y_prediction)
		return self.train_y_prediction, self.test_y_prediction

	def evaluation(self):
		pass

	def plot(self, real, prediction):
		x_range = 120
		predition_array = []
		real_array = []
		predition_array = []
		plt.figure()
		for i in range(x_range):
			real_array.append(real[i, 0, 1, 1, 0])
			predition_array.append(prediction[i, 0, 1, 1, 0])

		plt.xlabel('time sequence')
		plt.ylabel('activity strength')
		plt.plot(real_array, label='real', marker='.')
		plt.plot(predition_array, label='prediction', marker='.')
		plt.grid()
		plt.legend()
		plt.show()

	def fetch_real_prediction(self):
		# print(self.Y_train.shape, self.Y_test.shape)
		# print(self.train_y_prediction.shape, self.test_y_prediction.shape)
		'''
		real_Y = np.concatenate((self.Y_train, self.Y_test), axis=0)  # 1388 for training  149 for testing
		prediction_Y = np.concatenate((self.train_y_prediction, self.test_y_prediction), axis=0)
		'''
		real_Y = self.Y_train  # todo
		prediction_Y = self.train_y_prediction  # todo

		print(self.Y_info.shape, real_Y.shape, prediction_Y.shape)
		# gird_id, timestamp, real_min, real_avg, real_max, prediction_min, prediction_avg, prediction_max
		Internet_Traffic = np.concatenate(
			(self.Y_info, real_Y, prediction_Y), axis=-1)
		print(Internet_Traffic.shape)
		return Internet_Traffic


class Env_Config(object):
	def __init__(self):
		self.mean_load = 0.044
		self.B_LTE = 300
		self.B_UMTS = 14.4  # HSPA+
		self.B_GSM = 1.6

		self.avg_throughtput_LTE = 6  # 5MHz
		self.avg_throughtput_UMTS = 1.4  # HSPA
		self.avg_throughtput_GSM = 0.175  # EDGE

		self.saturation = 0.01
		self.average_demand_per_CDR = 0.87  # MB
		self.CDR_threshold = 0

		self.P_Macro_cst = 130  # W
		self.P_Small_cst = 4.8
		self.P_Macro_tx_power_factor = 4.7
		self.P_Small_tx_power_factor = 8
		self.P_Macro_tx = 20
		self.P_Small_tx = 0.05

		self.P_total_GSM_ = 1430
		self.p_total_UMTS = 1450
		self.P_total_LTE = 1040
		self.P_total_small_cell = 30


class CDR_to_Throughput(Env_Config):
	def __init__(self, cell_index):
		super().__init__()
		self._build(cell_index)

	def _get_cell_tower_grid_pair(self):
		cell_tower_with_grid = os.path.join(
			root_dir, 'cell_tower/cell_tower_with_grid.txt')
		with open(cell_tower_with_grid, 'r') as f:
			cell_grid = json.load(f)
		return cell_grid

	def _get_hour_CDR_internt_traffic(self, grid_list):
		source_path = os.path.join(root_dir, 'offloading/npy/real_prediction_traffic_array.npy')
		traffic_array = du.load_array(source_path)
		traffic_array = np.transpose(traffic_array, (2, 3, 0, 1, 4))
		traffic_list = []
		for search_grid_id in grid_list:
			# print('grid:{}'.format(grid))
			# row, column = compute_row_col(grid)
			# print(row, column)
			for row_index in range(traffic_array.shape[0]):
				for col_index in range(traffic_array.shape[1]):
					grid_id = traffic_array[row_index, col_index, 0, 0, 0]
					if search_grid_id == grid_id:
						grid_traffic = traffic_array[row_index, col_index, :, :]
			traffic_list.append(grid_traffic)

		traffic_array = np.stack(traffic_list)  # (grid_num, 1487, 1, 8)
		print('hour_CDR_internt_traffic shape', traffic_array.shape)
		return traffic_array

	def _get_10mins_CDR_internet_traffic(self, grid_list, reload=True):
		target_path = './npy/10min_CDR_internet_traffic_temp.npy'
		source_path = os.path.join(root_dir, 'offloading/npy/10min_CDR_internet_traffic.npy')
		if reload:
			# TK = Prepare_Task_Data('./npy/final/')
			# X_array, _ = TK.Task_max(grid_limit=[(0, 100), (0, 100)], generate_data=True)  # only need max here
			X_array = du.load_array(source_path)
			X_array = np.transpose(X_array, (2, 3, 0, 1, 4))
			array_list = []
			for search_grid_id in grid_list:
				# row, column = compute_row_col(grid_id)
				for row_index in range(X_array.shape[0]):
					for col_index in range(X_array.shape[1]):
						grid_id = X_array[row_index, col_index, 0, 0, 0]
						if search_grid_id == grid_id:
							new_x = X_array[row_index, col_index]
							new_x = new_x[:, :, (0, 1, -1)]
							array_list.append(new_x)  # grid_id, timestamp, internet traffic
			_10mins_CDR_internet_traffic = np.stack(array_list)
			print('_10mins_CDR_internet_traffic shape', _10mins_CDR_internet_traffic.shape)  # (grid_number, 1487, 6, 3)
			du.save_array(_10mins_CDR_internet_traffic, target_path)
		else:
			_10mins_CDR_internet_traffic = du.load_array(target_path)

		return _10mins_CDR_internet_traffic

	def _build(self, cell_index):
		cell_grid = self._get_cell_tower_grid_pair()
		cell_grid.sort(key=lambda x: x['index'])
		self.grid_list = cell_grid[cell_index]['grid']
		self.cell_info = cell_grid[cell_index]
		print('cell info', self.cell_info)
		traffic_hour_real_prediciton_CDR_array = self._get_hour_CDR_internt_traffic(self.grid_list)
		_10mins_CDR_internet_traffic = self._get_10mins_CDR_internet_traffic(self.grid_list, reload=True)

		# self.CDR_threshold = self._evaluate_CDR_threshold(_10mins_CDR_internet_traffic)
		# self._evaluate_saturation_factor()
		# self.hour_traffic_throughput_array = self._CDR_to_throghput(traffic_hour_real_prediciton_CDR_array, slice_range=(2, 8))
		# self.ten_minutes_traffic_throughput_array = self._CDR_to_throghput(_10mins_CDR_internet_traffic, slice_range=(2, 3))
		self._10mins_CDR_internet_traffic = self._combine_grid_CDR(_10mins_CDR_internet_traffic, slice_range=(2, 3))
		self.traffic_hour_real_prediciton_CDR_array = self._combine_grid_CDR(traffic_hour_real_prediciton_CDR_array, slice_range=(2, 8))
		self._10mins_internet_traffic_demand = self._calculate_internet_traffic_demand(self._10mins_CDR_internet_traffic)

	def _calculate_internet_traffic_demand(self, CDR_internet_traffic):
		print('CDR_internet_traffic shape', CDR_internet_traffic.shape)  # (1487, 6, 2)
		internet_traffic_demand = np.zeros_like(CDR_internet_traffic, dtype=None, order='K', subok=True)
		for hour_index in range(CDR_internet_traffic.shape[0]):
			for ten_minutes_index in range(CDR_internet_traffic.shape[1]):
				aggregated_traffic_demand = CDR_internet_traffic[hour_index, ten_minutes_index, 1] * self.average_demand_per_CDR
				internet_traffic_demand[hour_index, ten_minutes_index, 0] = CDR_internet_traffic[hour_index, ten_minutes_index, 0]
				internet_traffic_demand[hour_index, ten_minutes_index, 1] = aggregated_traffic_demand
		return internet_traffic_demand

	def _combine_grid_CDR(self, CDR_internet_traffic, slice_range=(2, 8)):
		print('CDR_internet_traffic', CDR_internet_traffic.shape)
		range_len = slice_range[1] - slice_range[0]
		combine_grid_CDR = np.zeros((CDR_internet_traffic.shape[1], CDR_internet_traffic.shape[2], range_len + 1))
		CDR_internet_traffic = np.transpose(CDR_internet_traffic, (1, 2, 0, 3))

		for i in range(CDR_internet_traffic.shape[0]):
			for j in range(CDR_internet_traffic.shape[1]):
				feature_list = [0 for features_list_index in range(range_len)]

				for grid_index in range(CDR_internet_traffic.shape[2]):
					for features_list_index, features_index in enumerate(range(*slice_range)):
						feature_list[features_list_index] += CDR_internet_traffic[i, j, grid_index, features_index]

				combine_grid_CDR[i, j, 0] = CDR_internet_traffic[i, j, 0, 1]  # timestamp
				for features_list_index, features in enumerate(feature_list):
					combine_grid_CDR[i, j, features_list_index + 1] = feature_list[features_list_index]
				# print(CDR)
		return combine_grid_CDR

	def _CDR_to_throghput(self, traffic_CDR_array, slice_range=(2, 8)):
		# print(traffic_CDR_array.shape)
		# grid_num = self.traffic_CDR_array.shape[0]
		range_len = slice_range[1] - slice_range[0]
		traffic_throughput_array = np.zeros((traffic_CDR_array.shape[1], traffic_CDR_array.shape[2], range_len + 1), dtype=float, order='C')
		''' 7: timestamp, real_min, real_avg, real_max, prediction_min, prediction_avg, prediction_max '''
		traffic_CDR_array = np.transpose(traffic_CDR_array, (1, 2, 0, 3))

		for i in range(traffic_CDR_array.shape[0]):
			for j in range(traffic_CDR_array.shape[1]):

				feature_list = [0 for features_list_index in range(range_len)]

				for grid_index in range(traffic_CDR_array.shape[2]):
					for features_list_index, features_index in enumerate(range(*slice_range)):
						feature_list[features_list_index] += traffic_CDR_array[i, j, grid_index, features_index]

				traffic_throughput_array[i, j, 0] = traffic_CDR_array[i, j, 0, 1]  # timestamp

				for features_list_index, features in enumerate(feature_list):
					traffic_throughput_array[i, j, features_list_index + 1] = self._calculate_traffic_load(self.cell_info['radio'], feature_list[features_list_index])

				'''
				real_min = 0
				real_avg = 0
				real_max = 0
				prediction_min = 0
				prediction_avg = 0
				prediction_max = 0

				for grid_index in range(traffic_CDR_array.shape[2]):
					real_min += traffic_CDR_array[i, j, grid_index, 2]
					real_avg += traffic_CDR_array[i, j, grid_index, 3]
					real_max += traffic_CDR_array[i, j, grid_index, 4]

					prediction_min += traffic_CDR_array[i, j, grid_index, 5]
					prediction_avg += traffic_CDR_array[i, j, grid_index, 6]
					prediction_max += traffic_CDR_array[i, j, grid_index, 7]

				traffic_throughput_array[i, j, 0] = traffic_CDR_array[i, j, 0, 1]  # timestamp
				traffic_throughput_array[i, j, 1] = self._calculate_traffic_load(self.cell_info['radio'], real_min)
				traffic_throughput_array[i, j, 2] = self._calculate_traffic_load(self.cell_info['radio'], real_avg)
				traffic_throughput_array[i, j, 3] = self._calculate_traffic_load(self.cell_info['radio'], real_max)
				traffic_throughput_array[i, j, 4] = self._calculate_traffic_load(self.cell_info['radio'], prediction_min)
				traffic_throughput_array[i, j, 5] = self._calculate_traffic_load(self.cell_info['radio'], prediction_avg)
				traffic_throughput_array[i, j, 6] = self._calculate_traffic_load(self.cell_info['radio'], prediction_max)
				'''
		return traffic_throughput_array

	def _evaluate_CDR_threshold(self, traffic_array):

		def count_exceed_threshold(grid_array, threshold):
			count_num = 0
			for each_grid in grid_array:
				for i in range(each_grid.shape[0]):
					for j in range(each_grid.shape[1]):
						if each_grid[i, j] > threshold:
							count_num += 1

			return count_num

		# grid_array = self._get_10mins_CDR_internet_traffic(self.grid_list, reload=False)
		traffic_array = traffic_array[:, :, -1]
		for threshold in range(100, 2000, 50):
			exceed_num = count_exceed_threshold(traffic_array, threshold)
			exceed_rate = (exceed_num * 10) / (traffic_array.shape[1] * traffic_array.shape[2] * 60)
			# print(exceed_rate)
			if exceed_rate < self.saturation:
				print('CDR treshold is:', threshold)
				# self.CDR_threshold = threshold
				break

		return threshold

	def _evaluate_saturation_factor(self):
		def count_factor(bandwidth):
			factor = bandwidth * 10 * 60 / (self.CDR_threshold * self.mean_load)
			# print(factor)
			return factor

		self.A_LTE = count_factor(self.B_LTE)
		self.A_UMTS = count_factor(self.B_UMTS)
		self.A_GSM = count_factor(self.B_GSM)

	def _calculate_traffic_load(self, radio_type, current_CDR):
		if radio_type == 'UMTS':
			band_width = self.B_UMTS
			factor = self.A_UMTS

		elif radio_type == 'LTE':
			band_width = self.B_LTE
			factor = self.A_LTE
		elif radio_type == 'GSM':
			band_width = self.B_GSM
			factor = self.A_GSM
		else:
			print('No such radio type')
			return 0

		if current_CDR >= self.CDR_threshold:
			# load = band_width
			load = (current_CDR * factor * self.mean_load) / (10 * 60)
		else:
			load = (current_CDR * factor * self.mean_load) / (10 * 60)

		return load

	def plot_grid_load(self, plot_range=(1388, 1487)):
		for grid in self.traffic_array:
			fig = plt.figure()

			ax1 = fig.add_subplot(211)
			ax2 = fig.add_subplot(212)
			plt.ylabel('traffic load')
			'''
			for i in range(grid.shape[0]):
				load = self._calculate_traffic_load(self.cell_info['radio'], grid[i])
				y_list.append(load)
			'''

			real_min = [self._calculate_traffic_load(self.cell_info['radio'], grid[range_index, 0, 2]) for range_index in range(*plot_range)]
			real_avg = [self._calculate_traffic_load(self.cell_info['radio'], grid[range_index, 0, 3]) for range_index in range(*plot_range)]
			real_max = [self._calculate_traffic_load(self.cell_info['radio'], grid[range_index, 0, 4]) for range_index in range(*plot_range)]

			ax1.plot(real_min, label='real_min')
			ax1.plot(real_avg, label='real_avg')
			ax1.plot(real_max, label='real_max')
			ax1.set_title('grid_id:' + str(grid[0, 0, 0]) + ' real')
			ax1.legend()

			prediction_min = [self._calculate_traffic_load(self.cell_info['radio'], grid[range_index, 0, 5]) for range_index in range(*plot_range)]
			prediction_avg = [self._calculate_traffic_load(self.cell_info['radio'], grid[range_index, 0, 6]) for range_index in range(*plot_range)]
			prediction_max = [self._calculate_traffic_load(self.cell_info['radio'], grid[range_index, 0, 7]) for range_index in range(*plot_range)]

			ax2.plot(prediction_min, label='prediction_min')
			ax2.plot(prediction_avg, label='prediciton_avg')
			ax2.plot(prediction_max, label='prediction_max')
			ax2.set_title('grid_id:' + str(grid[0, 0, 0]) + ' prediction')
			ax2.legend()

		plt.show()


class Milano_env(Env_Config):
	def __init__(self, cell_index):
		super().__init__()
		self.cell_index = cell_index
		self.n_small_cell = 5
		self.action_space = list(range(0, 11))
		self.n_actions = len(self.action_space)
		self.n_features = 3  # min avg max
		self._get_throghput_data()

	def _get_throghput_data(self):
		CDR_2_thr = CDR_to_Throughput(cell_index=self.cell_index)
		# self.hour_traffic_throughput_array = CDR_2_thr.hour_traffic_throughput_array  # (?, 1, 7) 7: gird_id, timestamp, real_min, real_avg, real_max, prediction_min, prediction_avg, prediction_max
		# self.ten_minutes_traffic_throughput_array = CDR_2_thr.ten_minutes_traffic_throughput_array  # (?, 6, 2): 6: one hour 6 data. 3: timestamp, network_traffic
		self.cell_info = CDR_2_thr.cell_info
		self._10mins_internet_traffic_demand = CDR_2_thr._10mins_internet_traffic_demand
		self.traffic_hour_real_prediciton_CDR_array = CDR_2_thr.traffic_hour_real_prediciton_CDR_array
		# print(self.traffic_hour_real_prediciton_CDR_array.shape, self._10mins_CDR_internet_traffic_demand.shape)

	def _calculate_energy_efficiency(self, _10min_traffic, num_of_small_cell):
		# print('one hour traffic shape', one_hour_traffic.shape, '10 minutes traffice shape', _10min_traffic.shape)
		num_of_cell = 1 + num_of_small_cell
		epoch_time = 60 * 10  # seconds
		total_power_consumption = 0
		totol_internet_traffic = 0
		macro_load_list = []
		
		def get_Macro_power(radio_type):
			if radio_type == 'UMTS':
				Power = self.p_total_UMTS
			elif radio_type == 'LTE':
				Power = self.P_total_LTE
			elif radio_type == 'GSM':
				Power = self.P_total_GSM_
			else:
				print('No such radio type')
				Power = 0
			return Power

		def get_Macro_average_throughput(radio_type):
			if radio_type == 'UMTS':
				avg_throughput = self.avg_throughtput_UMTS
			elif radio_type == 'LTE':
				avg_throughput = self.avg_throughtput_LTE
			elif radio_type == 'GSM':
				avg_throughput = self.avg_throughtput_GSM
			else:
				print('No such radio type')
				avg_throughput = 0
			return avg_throughput

		def get_cell_internet_traffic(
			request_internet_traffic_demand,
			epoch_time,
			num_of_small_cell,
			small_cell_avg_throughput,
			macro_cell_avg_throughput,
			pre_define_small_cell_load_rate=0.7,
			pre_define_Macro_cell_load_rate=0.5):

			small_load_time = pre_define_small_cell_load_rate * epoch_time
			small_each_responsible_interent_traffic = small_load_time * small_cell_avg_throughput
			small_total_responsible_interent_traffic = small_each_responsible_interent_traffic * num_of_small_cell
			# print(small_total_responsible_interent_traffic)
			if small_total_responsible_interent_traffic > request_internet_traffic_demand:
				Macro_cell_load_rate = np.random.normal(pre_define_Macro_cell_load_rate, 0.1)
				Macro_cell_load_rate = abs(Macro_cell_load_rate)
				# print(Macro_cell_load_rate)
				Macro_cell_load_rate = Macro_cell_load_rate if Macro_cell_load_rate < 1 else pre_define_Macro_cell_load_rate

				macro_load_time = Macro_cell_load_rate * epoch_time
				macro_resposible_interent_traffic = macro_load_time * macro_cell_avg_throughput

				# print((macro_resposible_interent_traffic / macro_cell_avg_throughput)/epoch_time)

				small_each_responsible_interent_traffic = (request_internet_traffic_demand - macro_resposible_interent_traffic) / num_of_small_cell

			else:
				macro_resposible_interent_traffic = request_internet_traffic_demand - small_total_responsible_interent_traffic

			return macro_resposible_interent_traffic, small_each_responsible_interent_traffic

		small_cell_avg_throughput = self.avg_throughtput_LTE
		macro_cell_avg_throughput = get_Macro_average_throughput(self.cell_info['radio'])

		for index, _10min in enumerate(_10min_traffic):
			request_internet_traffic_demand = _10min[1]
			if num_of_small_cell != 0:
				macro_resposible_interent_traffic, small_each_responsible_interent_traffic = get_cell_internet_traffic(
					request_internet_traffic_demand,
					epoch_time,
					num_of_small_cell,
					small_cell_avg_throughput,
					macro_cell_avg_throughput,
					0.7,
					0.5)
				each_small_cell_load_time = small_each_responsible_interent_traffic / small_cell_avg_throughput
				small_load_rate = each_small_cell_load_time / epoch_time
				Small_energy_consumption = epoch_time * self.P_Small_cst + self.P_Small_tx_power_factor * small_load_rate * epoch_time * self.P_Small_tx
				small_cell_energy_efficiency = small_each_responsible_interent_traffic / Small_energy_consumption
				# print('Small cell:load_rate:{:.4f}, energy_consumption:{:.4f} energy_effi:{:.4f}'.format(small_load_rate, Small_energy_consumption, small_cell_energy_efficiency))
				total_power_consumption += Small_energy_consumption * 4
				totol_internet_traffic += small_each_responsible_interent_traffic * 4
			else:
				macro_resposible_interent_traffic = request_internet_traffic_demand

			''' macro '''
			macro_cell_load_time = macro_resposible_interent_traffic / macro_cell_avg_throughput
			macro_load_rate = macro_cell_load_time / epoch_time
			macro_energy_comsumption = epoch_time * self.P_Macro_cst + self.P_Macro_tx_power_factor * self.P_Macro_tx * macro_load_rate * epoch_time
			macro_energy_efficiency = macro_resposible_interent_traffic / macro_energy_comsumption
			# print('Macro cell:load_rate:{:.4f}, energy_consumption:{:.4f} energy_effi:{:.4f}'.format(macro_load_rate, macro_energy_comsumption, macro_energy_efficiency))
			macro_load_list.append(macro_load_rate)
			total_power_consumption += macro_energy_comsumption
			totol_internet_traffic += macro_resposible_interent_traffic

		location_energy_effi = totol_internet_traffic / total_power_consumption
		# print('hour location_energy_effi', location_energy_effi)
		return location_energy_effi, macro_load_list

	def _reward(self, _10min_traffic, action):
		threshold = 0.7
		energy_effi, macro_load_rate_list = self._calculate_energy_efficiency(_10min_traffic, action)
		reward = energy_effi
		for macro_load_rate in macro_load_rate_list:
			if macro_load_rate > threshold:
				macro_load_rate_Eva = np.exp(macro_load_rate - threshold)
				reward -= macro_load_rate_Eva
			else:
				macro_load_rate_Eva = np.exp(threshold - macro_load_rate)
				reward += macro_load_rate_Eva

		print('energy effi:{} macro load list:{} reward:{}'.format(energy_effi, macro_load_rate_list, reward))
		return reward, energy_effi

	def step(self, action):
		done = False
		try:
			env = next(self.env_iter)
			reward = self._reward(env[1], action)
			env = env[0][0, 4:]
		except StopIteration:
			print('StopIteration')
			done = True
			env = None
			reward = None

		return env, reward, done

	def reset(self):
		self.env_iter = zip(
			self.traffic_hour_real_prediciton_CDR_array,
			self._10mins_internet_traffic_demand)
		# env = next(self.env_iter)
		# reward = self._calculate_energy_efficiency(*env, 5)
		return self.traffic_hour_real_prediciton_CDR_array[0][0, 4:]

if __name__ == "__main__":
	# nine_grid()
	# loop_all_grid()
	env = Milano_env()
	env.reset()
	env.step(5)
