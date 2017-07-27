import numpy as np
import matplotlib.pyplot as plt
import os
import json
import sys
from statistics import mean
from sklearn import preprocessing
sys.path.append('/home/mldp/ML_with_bigdata')
import data_utility as du
from multi_task_data import Prepare_Task_Data
from CNN_RNN.CNN_RNN_config import HyperParameterConfig
from CNN_RNN.CNN_RNN import CNN_RNN
import CNN_RNN.utility
root_dir = '/home/mldp/ML_with_bigdata'
logger = CNN_RNN.utility.setlog('env')


class Env_Config(object):
	def __init__(self):
		self.mean_load = 0.044
		self.B_LTE = 300
		self.B_UMTS = 14.4  # HSPA+
		self.B_GSM = 1.6

		self.avg_throughtput_LTE = 6  # 5MHz
		self.avg_throughtput_UMTS = 6  # 1.4  # HSPA
		self.avg_throughtput_GSM = 6  # 0.175  # EDGE

		self.saturation = 0.01
		self.average_demand_per_CDR = 2.5  # Mb
		self.CDR_threshold = 0

		self.P_Macro_cst = 130  # W
		self.P_Small_cst = 4.8
		self.P_Macro_tx_power_factor = 3.5
		self.P_Small_tx_power_factor = 8
		self.P_Macro_tx = 20
		self.P_Small_tx = 1  # 0.05

		self.P_total_GSM_ = 1430
		self.p_total_UMTS = 1450
		self.P_total_LTE = 1040
		self.P_total_small_cell = 30

		self.base_dir = os.path.join(root_dir, 'offloading/npy/real_prediction')
		self.small_cell_num = 10
		self.features_num = 3  # min avg max
		self.macro_threshold = 0.7
		self.small_threshold = 0.5
		self.load_factor = 1
		self.effi_factor = 10000


def generate_new_real_prediction_traffic_array():
	# target_path = os.path.join(root_dir, 'offloading/npy/real_prediction_traffic_array.npy')
	source_path = os.path.join(root_dir, 'CNN_RNN/result/CNN_RNN/all_real_prediction_traffic_array_0718.npy')
	data_array = du.load_array(source_path)
	row_center_list = list(range(40, 80, 3))
	col_center_list = list(range(30, 70, 3))
	row_range = (row_center_list[0] - 1, row_center_list[-1] + 1)
	col_range = (col_center_list[0] - 1, col_center_list[-1] + 1)
	data_array = data_array[:, :, row_range[0]: row_range[1], col_range[0]: col_range[1]]
	return data_array
	# du.save_array(data_array, target_path)


def generate_new_10mins_CDR_internet_traffic():
	# target_path = os.path.join(root_dir, 'offloading/npy/10min_CDR_internet_traffic.npy')
	TK = Prepare_Task_Data('./npy/final/')
	X_array, _ = TK.Task_max(grid_limit=[(0, 100), (0, 100)], generate_data=False)  # only need max here
	row_center_list = list(range(40, 80, 3))
	col_center_list = list(range(30, 70, 3))
	row_range = (row_center_list[0] - 1, row_center_list[-1] + 1)
	col_range = (col_center_list[0] - 1, col_center_list[-1] + 1)
	X_array = X_array[:, :, row_range[0]: row_range[1], col_range[0]: col_range[1]]
	config = Env_Config()
	for hour_index in range(X_array.shape[0]):
		for min_index in range(X_array.shape[1]):
			for row_index in range(X_array.shape[2]):
				for col_index in range(X_array.shape[3]):
					cdr_traffic_demand = np.random.normal(config.average_demand_per_CDR, 5)
					while cdr_traffic_demand < 0:
						cdr_traffic_demand = np.random.normal(config.average_demand_per_CDR, 2)
					X_array[hour_index, min_index, row_index, col_index, -1] = X_array[hour_index, min_index, row_index, col_index, -1] * cdr_traffic_demand  # cdr to throughput

	return X_array
	# du.save_array(X_array, target_path)


def generate_real_prediction_traffic_array():

	targer_dir = os.path.join(root_dir, 'offloading/npy/real_prediction')
	CNN_RNN.utility.check_path_exist(targer_dir)
	hour_target_path = os.path.join(targer_dir, 'hour_traffic_array.npy')
	_10mins_target_path = os.path.join(targer_dir, '10min_CDR_internet_traffic.npy')

	hour_traffic = generate_new_real_prediction_traffic_array()  # (1487, 1, 41, 41, 8)
	_10_min_traffic = generate_new_10mins_CDR_internet_traffic()  # (1487, 6, 41, 41, 3)

	_10_min_traffic = _10_min_traffic[1:]  # (1486, 6, 41, 41, 3)
	hour_traffic = hour_traffic[:-1]  # (1486, 1, 41, 41, 8)
	print('hour_traffic shape:{} _10_min_traffic shape:{}'.format(hour_traffic.shape, _10_min_traffic.shape))
	du.save_array(hour_traffic, hour_target_path)
	du.save_array(_10_min_traffic, _10mins_target_path)


def generate_without_prediction_traffic_array():
	targer_dir = os.path.join(root_dir, 'offloading/npy/real_without_prediction')
	CNN_RNN.utility.check_path_exist(targer_dir)
	hour_target_path = os.path.join(targer_dir, 'hour_traffic_array.npy')
	_10mins_target_path = os.path.join(targer_dir, '10min_CDR_internet_traffic.npy')

	hour_traffic = generate_new_real_prediction_traffic_array()  # (1487, 1, 41, 41, 8)
	_10_min_traffic = generate_new_10mins_CDR_internet_traffic()  # (1487, 6, 41, 41, 3)

	_10_min_traffic = _10_min_traffic[2:]  # (1485, 6, 41, 41, 3)
	hour_traffic = hour_traffic[:-2]  # (1485, 1, 41, 41, 8)
	print('hour_traffic shape:{} _10_min_traffic shape:{}'.format(hour_traffic.shape, _10_min_traffic.shape))
	du.save_array(hour_traffic, hour_target_path)
	du.save_array(_10_min_traffic, _10mins_target_path)


def generate_god_mode_traffic_array():
	targer_dir = os.path.join(root_dir, 'offloading/npy/real_god_prediction')
	CNN_RNN.utility.check_path_exist(targer_dir)
	hour_target_path = os.path.join(targer_dir, 'hour_traffic_array.npy')
	_10mins_target_path = os.path.join(targer_dir, '10min_CDR_internet_traffic.npy')

	hour_traffic = generate_new_real_prediction_traffic_array()  # (1487, 1, 41, 41, 8)
	_10_min_traffic = generate_new_10mins_CDR_internet_traffic()  # (1487, 6, 41, 41, 3)
	hour_traffic[:, :, :, :, 5] = hour_traffic[:, :, :, :, 2]
	hour_traffic[:, :, :, :, 6] = hour_traffic[:, :, :, :, 3]
	hour_traffic[:, :, :, :, 7] = hour_traffic[:, :, :, :, 4]
	_10_min_traffic = _10_min_traffic[1:]  # (1486, 6, 41, 41, 3)
	hour_traffic = hour_traffic[:-1]  # (1486, 1, 41, 41, 8)
	# print(hour_traffic[1, 0, 1, 1, 5:], hour_traffic[1, 0, 1, 1, 2:5])
	print('hour_traffic shape:{} _10_min_traffic shape:{}'.format(hour_traffic.shape, _10_min_traffic.shape))
	du.save_array(hour_traffic, hour_target_path)
	du.save_array(_10_min_traffic, _10mins_target_path)


class CDR_to_Throughput(Env_Config):

	def __init__(self, config, cell_index):
		# super().__init__()
		logger.info('cell index:{}'.format(cell_index))
		self.config = config
		self._build(cell_index)

	def _get_cell_tower_grid_pair(self):
		cell_tower_with_grid = os.path.join(
			root_dir, 'cell_tower/cell_tower_with_grid.txt')
		with open(cell_tower_with_grid, 'r') as f:
			cell_grid = json.load(f)
		return cell_grid

	def _get_hour_CDR_internt_traffic(self, grid_list):
		source_path = os.path.join(self.config.base_dir, 'hour_traffic_array.npy')
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
		logger.debug('hour_CDR_internt_traffic shape:{}'.format(traffic_array.shape))
		return traffic_array

	def _get_10mins_CDR_internet_traffic(self, grid_list, reload=True):
		target_path = './npy/10min_CDR_internet_traffic_temp.npy'
		source_path = os.path.join(self.config.base_dir, '10min_CDR_internet_traffic.npy')
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
			logger.debug('_10mins_CDR_internet_traffic shape:{}'.format(_10mins_CDR_internet_traffic.shape))  # (grid_number, 1487, 6, 3)
			du.save_array(_10mins_CDR_internet_traffic, target_path)
		else:
			_10mins_CDR_internet_traffic = du.load_array(target_path)

		return _10mins_CDR_internet_traffic

	def _build(self, cell_index):
		cell_grid = self._get_cell_tower_grid_pair()
		cell_grid.sort(key=lambda x: x['index'])
		self.grid_list = cell_grid[cell_index]['grid']
		self.cell_info = cell_grid[cell_index]
		logger.info('cell info:{}'.format(self.cell_info))
		traffic_hour_real_prediciton_CDR_array = self._get_hour_CDR_internt_traffic(self.grid_list)
		_10mins_CDR_internet_traffic = self._get_10mins_CDR_internet_traffic(self.grid_list, reload=True)

		self._10mins_CDR_internet_traffic = self._combine_grid_CDR(_10mins_CDR_internet_traffic, slice_range=(2, 3))
		self.traffic_hour_real_prediciton_CDR_array = self._combine_grid_CDR(traffic_hour_real_prediciton_CDR_array, slice_range=(2, 8))  # (timestamp, real_min, real_avg, real_max, prediction_min, prediction_avg, prediction_max)
		self._10mins_internet_traffic_demand = self._10mins_CDR_internet_traffic
		'''
		already internet traffic instead of CDR number
		'''
		# self._10mins_internet_traffic_demand = self._calculate_internet_traffic_demand(self._10mins_CDR_internet_traffic)

		# self.CDR_threshold = self._evaluate_CDR_threshold(_10mins_CDR_internet_traffic)
		# self._evaluate_saturation_factor()
		# self.hour_traffic_throughput_array = self._CDR_to_throghput(traffic_hour_real_prediciton_CDR_array, slice_range=(2, 8))
		# self.ten_minutes_traffic_throughput_array = self._CDR_to_throghput(_10mins_CDR_internet_traffic, slice_range=(2, 3))

	def _calculate_internet_traffic_demand(self, CDR_internet_traffic):
		logger.debug('CDR_internet_traffic shape:{}'.format(CDR_internet_traffic.shape))  # (1487, 6, 2)
		internet_traffic_demand = np.zeros_like(CDR_internet_traffic, dtype=None, order='K', subok=True)
		for hour_index in range(CDR_internet_traffic.shape[0]):
			for ten_minutes_index in range(CDR_internet_traffic.shape[1]):
				cdr_traffic_demand = np.random.normal(self.config.average_demand_per_CDR, 2)
				cdr_traffic_demand = abs(cdr_traffic_demand)
				aggregated_traffic_demand = CDR_internet_traffic[hour_index, ten_minutes_index, 1] * cdr_traffic_demand
				internet_traffic_demand[hour_index, ten_minutes_index, 0] = CDR_internet_traffic[hour_index, ten_minutes_index, 0]
				internet_traffic_demand[hour_index, ten_minutes_index, 1] = aggregated_traffic_demand
		return internet_traffic_demand

	def _combine_grid_CDR(self, CDR_internet_traffic, slice_range=(2, 8)):
		logger.debug('CDR_internet_traffic:{}'.format(CDR_internet_traffic.shape))
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

	# def _evaluate_CDR_threshold(self, traffic_array):

	# 	def count_exceed_threshold(grid_array, threshold):
	# 		count_num = 0
	# 		for each_grid in grid_array:
	# 			for i in range(each_grid.shape[0]):
	# 				for j in range(each_grid.shape[1]):
	# 					if each_grid[i, j] > threshold:
	# 						count_num += 1

	# 		return count_num

	# 	# grid_array = self._get_10mins_CDR_internet_traffic(self.grid_list, reload=False)
	# 	traffic_array = traffic_array[:, :, -1]
	# 	for threshold in range(100, 2000, 50):
	# 		exceed_num = count_exceed_threshold(traffic_array, threshold)
	# 		exceed_rate = (exceed_num * 10) / (traffic_array.shape[1] * traffic_array.shape[2] * 60)
	# 		# print(exceed_rate)
	# 		if exceed_rate < self.config.saturation:
	# 			print('CDR treshold is:', threshold)
	# 			# self.CDR_threshold = threshold
	# 			break

	# 	return threshold

	# def _evaluate_saturation_factor(self):
	# 	def count_factor(bandwidth):
	# 		factor = bandwidth * 10 * 60 / (self.config.CDR_threshold * self.config.mean_load)
	# 		# print(factor)
	# 		return factor

	# 	self.A_LTE = count_factor(self.config.B_LTE)
	# 	self.A_UMTS = count_factor(self.config.B_UMTS)
	# 	self.A_GSM = count_factor(self.config.B_GSM)

	# def _calculate_traffic_load(self, radio_type, current_CDR):
	# 	if radio_type == 'UMTS':
	# 		band_width = self.config.B_UMTS
	# 		factor = self.A_UMTS

	# 	elif radio_type == 'LTE':
	# 		band_width = self.config.B_LTE
	# 		factor = self.A_LTE
	# 	elif radio_type == 'GSM':
	# 		band_width = self.config.B_GSM
	# 		factor = self.A_GSM
	# 	else:
	# 		print('No such radio type')
	# 		return 0

	# 	if current_CDR >= self.CDR_threshold:
	# 		# load = band_width
	# 		load = (current_CDR * factor * self.config.mean_load) / (10 * 60)
	# 	else:
	# 		load = (current_CDR * factor * self.config.mean_load) / (10 * 60)

	# 	return load

	# def plot_grid_load(self, plot_range=(1388, 1487)):
	# 	for grid in self.traffic_array:
	# 		fig = plt.figure()

	# 		ax1 = fig.add_subplot(211)
	# 		ax2 = fig.add_subplot(212)
	# 		plt.ylabel('traffic load')
	# 		'''
	# 		for i in range(grid.shape[0]):
	# 			load = self._calculate_traffic_load(self.cell_info['radio'], grid[i])
	# 			y_list.append(load)
	# 		'''

	# 		real_min = [self._calculate_traffic_load(self.cell_info['radio'], grid[range_index, 0, 2]) for range_index in range(*plot_range)]
	# 		real_avg = [self._calculate_traffic_load(self.cell_info['radio'], grid[range_index, 0, 3]) for range_index in range(*plot_range)]
	# 		real_max = [self._calculate_traffic_load(self.cell_info['radio'], grid[range_index, 0, 4]) for range_index in range(*plot_range)]

	# 		ax1.plot(real_min, label='real_min')
	# 		ax1.plot(real_avg, label='real_avg')
	# 		ax1.plot(real_max, label='real_max')
	# 		ax1.set_title('grid_id:' + str(grid[0, 0, 0]) + ' real')
	# 		ax1.legend()

	# 		prediction_min = [self._calculate_traffic_load(self.cell_info['radio'], grid[range_index, 0, 5]) for range_index in range(*plot_range)]
	# 		prediction_avg = [self._calculate_traffic_load(self.cell_info['radio'], grid[range_index, 0, 6]) for range_index in range(*plot_range)]
	# 		prediction_max = [self._calculate_traffic_load(self.cell_info['radio'], grid[range_index, 0, 7]) for range_index in range(*plot_range)]

	# 		ax2.plot(prediction_min, label='prediction_min')
	# 		ax2.plot(prediction_avg, label='prediciton_avg')
	# 		ax2.plot(prediction_max, label='prediction_max')
	# 		ax2.set_title('grid_id:' + str(grid[0, 0, 0]) + ' prediction')
	# 		ax2.legend()

	# 	plt.show()


class Milano_env():
	def __init__(self, cell_index, config):
		# super().__init__()
		self.config = config
		self.cell_index = cell_index
		self.n_actions = config.small_cell_num + 1  # 0 to 10 total 11
		self.n_features = config.features_num
		self._get_throghput_data()

	def _get_throghput_data(self):
		CDR_2_thr = CDR_to_Throughput(self.config, cell_index=self.cell_index)
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
		small_load_list = []

		def get_Macro_power(radio_type):
			if radio_type == 'UMTS':
				Power = self.config.p_total_UMTS
			elif radio_type == 'LTE':
				Power = self.config.P_total_LTE
			elif radio_type == 'GSM':
				Power = self.config.P_total_GSM_
			else:
				print('No such radio type')
				Power = 0
			return Power

		def get_Macro_average_throughput(radio_type):
			if radio_type == 'UMTS':
				avg_throughput = self.config.avg_throughtput_UMTS
			elif radio_type == 'LTE':
				avg_throughput = self.config.avg_throughtput_LTE
			elif radio_type == 'GSM':
				avg_throughput = self.config.avg_throughtput_GSM
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
			pre_define_small_cell_share_rate = np.random.normal(pre_define_small_cell_load_rate, scale=0.1)
			pre_define_small_cell_share_rate = abs(pre_define_small_cell_share_rate)
			pre_define_small_cell_share_rate = pre_define_small_cell_share_rate if pre_define_small_cell_share_rate < 1 else pre_define_small_cell_load_rate
			pre_define_Macro_cell_share_rate = 1 - pre_define_small_cell_share_rate
			# pre_define_Macro_cell_share_rate = 1 / (num_of_small_cell + 1)
			# pre_define_small_cell_share_rate = 1 - pre_define_Macro_cell_share_rate

			# macro_load_time = pre_define_Macro_cell_load_rate * epoch_time

			macro_interent_traffic_ability = macro_cell_avg_throughput * epoch_time

			# small_load_time = pre_define_small_cell_load_rate * epoch_time
			small_internet_traffic_ability = small_cell_avg_throughput * num_of_small_cell * epoch_time

			macro_resposible_interent_traffic = request_internet_traffic_demand * pre_define_Macro_cell_share_rate
			small_resposible_interent_traffic = request_internet_traffic_demand * pre_define_small_cell_share_rate

			if small_internet_traffic_ability < small_resposible_interent_traffic and macro_interent_traffic_ability > macro_resposible_interent_traffic:
				small_each_responsible_interent_traffic = small_internet_traffic_ability / num_of_small_cell
				macro_resposible_interent_traffic = request_internet_traffic_demand - small_internet_traffic_ability

			elif small_internet_traffic_ability < small_resposible_interent_traffic and macro_interent_traffic_ability < macro_resposible_interent_traffic:
				small_each_responsible_interent_traffic = small_internet_traffic_ability / num_of_small_cell
				macro_resposible_interent_traffic = request_internet_traffic_demand - small_internet_traffic_ability

			elif small_internet_traffic_ability > small_resposible_interent_traffic and macro_interent_traffic_ability > macro_resposible_interent_traffic:
				small_each_responsible_interent_traffic = small_resposible_interent_traffic / num_of_small_cell
				macro_resposible_interent_traffic = macro_resposible_interent_traffic

			elif small_internet_traffic_ability > small_resposible_interent_traffic and macro_interent_traffic_ability < macro_resposible_interent_traffic:

				macro_exceed_internet_traffic = macro_resposible_interent_traffic - macro_interent_traffic_ability

				small_resposible_interent_traffic += macro_exceed_internet_traffic
				if small_internet_traffic_ability < small_resposible_interent_traffic:
					small_each_responsible_interent_traffic = small_internet_traffic_ability / num_of_small_cell
					macro_resposible_interent_traffic = request_internet_traffic_demand - small_internet_traffic_ability
				else:
					Macro_cell_load_rate = np.random.normal(pre_define_Macro_cell_load_rate, 0.1)
					Macro_cell_load_rate = abs(Macro_cell_load_rate)
					Macro_cell_load_rate = Macro_cell_load_rate if Macro_cell_load_rate < 1 else 1
					# print(Macro_cell_load_rate)
					macro_load_time = Macro_cell_load_rate * epoch_time
					macro_resposible_interent_traffic = macro_load_time * macro_cell_avg_throughput
					small_each_responsible_interent_traffic = (request_internet_traffic_demand - macro_resposible_interent_traffic) / num_of_small_cell

			# print('total:{}, macro:{} small:{}'.format(request_internet_traffic_demand, macro_resposible_interent_traffic, small_each_responsible_interent_traffic))
			'''

			if macro_interent_traffic_ability > request_internet_traffic_demand:
				macro_resposible_interent_traffic = request_internet_traffic_demand
				small_each_responsible_interent_traffic = 0

			elif small_internet_traffic_ability > request_internet_traffic_demand:
				Macro_cell_load_rate = np.random.normal(pre_define_Macro_cell_load_rate, 0.1)
				Macro_cell_load_rate = abs(Macro_cell_load_rate)
				# print(Macro_cell_load_rate)
				Macro_cell_load_rate = Macro_cell_load_rate if Macro_cell_load_rate < 1 else pre_define_Macro_cell_load_rate

				macro_load_time = Macro_cell_load_rate * epoch_time
				macro_resposible_interent_traffic = macro_load_time * macro_cell_avg_throughput

				small_each_responsible_interent_traffic = (request_internet_traffic_demand - macro_resposible_interent_traffic) / num_of_small_cell

			else:
				macro_resposible_interent_traffic = request_internet_traffic_demand - small_internet_traffic_ability
				small_each_responsible_interent_traffic = small_internet_traffic_ability / num_of_small_cell
			'''
			'''
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
				if small_each_responsible_interent_traffic < 0:
					# logger.info('request_internet_traffic_demand:{} macro_resposible_interent_traffic:{}'.format(request_internet_traffic_demand, macro_resposible_interent_traffic))
					# logger.warning('small_each_responsible_interent_traffic :{}'.format(small_each_responsible_interent_traffic))
					small_each_responsible_interent_traffic = 0
					macro_resposible_interent_traffic = request_internet_traffic_demand
			else:
				macro_resposible_interent_traffic = request_internet_traffic_demand - small_total_responsible_interent_traffic
			'''
			return macro_resposible_interent_traffic, small_each_responsible_interent_traffic

		small_cell_avg_throughput = self.config.avg_throughtput_LTE
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
				Small_energy_consumption = epoch_time * self.config.P_Small_cst + self.config.P_Small_tx_power_factor * small_load_rate * epoch_time * self.config.P_Small_tx
				# small_cell_energy_efficiency = small_each_responsible_interent_traffic / Small_energy_consumption
				# print('Small cell:load_rate:{:.4f}, energy_consumption:{:.4f} energy_effi:{:.4f}'.format(small_load_rate, Small_energy_consumption, small_cell_energy_efficiency))
				total_power_consumption += Small_energy_consumption * num_of_small_cell
				totol_internet_traffic += small_each_responsible_interent_traffic * num_of_small_cell
				small_load_list.append(small_load_rate)
			else:
				macro_resposible_interent_traffic = request_internet_traffic_demand
			if macro_resposible_interent_traffic < 0:
				logger.error('macro_resposible_interent_traffic :{}'.format(macro_resposible_interent_traffic))
			''' macro '''
			macro_cell_load_time = macro_resposible_interent_traffic / macro_cell_avg_throughput
			macro_load_rate = macro_cell_load_time / epoch_time
			if macro_load_rate > 1:
				load_rate_limit = 1
				macro_limit_internet_traffic = macro_cell_avg_throughput * epoch_time
			else:
				load_rate_limit = macro_load_rate
				macro_limit_internet_traffic = macro_resposible_interent_traffic

			macro_energy_comsumption = epoch_time * self.config.P_Macro_cst + self.config.P_Macro_tx_power_factor * self.config.P_Macro_tx * load_rate_limit * epoch_time
			# macro_energy_efficiency = macro_limit_internet_traffic / macro_energy_comsumption
			# print('Macro cell:load_rate:{:.4f}, energy_consumption:{:.4f} energy_effi:{:.4f}'.format(macro_load_rate, macro_energy_comsumption, macro_energy_efficiency))
			macro_load_list.append(macro_load_rate)
			total_power_consumption += macro_energy_comsumption
			totol_internet_traffic += macro_limit_internet_traffic

		location_energy_effi = totol_internet_traffic / total_power_consumption
		if location_energy_effi < 0:
			logger.error('energy_effi:{:.4f} action:{} totol_internet_traffic:{:.4f}, hour_sum:{:.4f} total_power_consumption:{:.2f}'.format(location_energy_effi, num_of_small_cell, totol_internet_traffic, np.sum(_10min_traffic[:, 1]), total_power_consumption))
		self.total_power_consumption.append(total_power_consumption)
		self.macro_cell_load.append(mean(macro_load_list))  # per hour
		self.internet_traffic_demand.append(totol_internet_traffic)  # per hour
		self.actions.append(num_of_small_cell)
		if len(small_load_list) > 0:
			self.small_cell_load.append(mean(small_load_list))
		else:
			self.small_cell_load.append(0)
		# print('hour location_energy_effi', location_energy_effi)
		return location_energy_effi, macro_load_list, small_load_list

	def _reward(self, _10min_traffic, action):
		macro_threshold = self.config.macro_threshold
		small_threshold = self.config.small_threshold
		afa = self.config.load_factor
		beta = self.config.effi_factor
		energy_effi, macro_load_rate_list, small_load_list = self._calculate_energy_efficiency(_10min_traffic, action)
		reward = energy_effi * beta
		for i, macro_load_rate in enumerate(macro_load_rate_list):

			if macro_load_rate > macro_threshold:
				macro_load_rate_Eva = np.exp((macro_load_rate - macro_threshold) * 10)
				macro_load_rate_Eva = macro_load_rate_Eva if macro_load_rate_Eva < 1000 else 1000
				reward -= afa * macro_load_rate_Eva

			if action > 0:  # small cell number > 0
				small_load = small_load_list[i]
				if macro_load_rate < macro_threshold and small_load < small_threshold:  # gap: 0.3
					small_load_eva = np.exp((small_threshold - small_load) * 10)
					reward -= afa * (small_load_eva * action) * 3

				if small_load > small_threshold and small_load < 0.7:  # gap: 0.4
					small_load_eva = np.exp(small_load * 10)
					reward += afa * (small_load_eva * action) * 3

				if small_load > 0.7 and action is not 10:
					small_load_eva = np.exp((small_load - 0.7) * 10)  # gap: 0.3
					reward -= afa * (small_load_eva * action)

		# 	else:
		# 		macro_load_rate_Eva = macro_threshold - macro_load_rate
		# 		reward += afa * macro_load_rate_Eva
			

			# if small_load > small_threshold:
			# 	small_load_eva = small_load - small_threshold
			# 	reward += afa * (small_load_eva * action)

		# print('action:{} reward:{} energy effi:{} macro load list:{} small_load_list:{} '.format(action, reward, energy_effi, macro_load_rate_list, small_load_list))
		return reward, energy_effi

	def step(self, action, training=True):
		done = False
		if training:
			try:
				env = next(self.training_env_iter)
				reward = self._reward(env[1], action)
				env = env[0][0, 4:]
			except StopIteration:
				# logger.info('StopIteration')
				done = True
				env = None
				reward = None
		else:
			try:
				env = next(self.testing_env_iter)
				reward = self._reward(env[1], action)
				env = env[0][0, 4:]
			except StopIteration:
				# logger.info('StopIteration')
				done = True
				env = None
				reward = None

		return env, reward, done

	def reset(self, training=True):

		self.total_power_consumption = []  # per hour
		self.macro_cell_load = []  # per hour
		self.internet_traffic_demand = []  # per hour
		self.small_cell_load = []
		self.actions = []

		'''
		self.traffic_hour_real_prediciton_CDR_array,  # (1487, 1, 7)
		self._10mins_internet_traffic_demand  # (1487, 6, 2)  2: timestamp, demand
		'''
		data_len = self.traffic_hour_real_prediciton_CDR_array.shape[0]
		if training:
			hour_traffic = self.traffic_hour_real_prediciton_CDR_array[: 9 * data_len // 10]
			_10min_traffic = self._10mins_internet_traffic_demand[: 9 * data_len // 10]
			self.training_env_iter = zip(hour_traffic, _10min_traffic)
		else:
			hour_traffic = self.traffic_hour_real_prediciton_CDR_array
			_10min_traffic = self._10mins_internet_traffic_demand
			self.testing_env_iter = zip(hour_traffic, _10min_traffic)

		# env = next(self.env_iter)
		# reward = self._calculate_energy_efficiency(*env, 5)
		# logger.debug('traffic_hour_real_prediciton_CDR_array shape:{}'.format(self.traffic_hour_real_prediciton_CDR_array.shape))
		# logger.debug('_10mins_internet_traffic_demand shape:{}'.format(self._10mins_internet_traffic_demand.shape))

		return hour_traffic[0][0, 4:]

	def reset_10_mins(self, training=True):
		self.total_power_consumption = []  # per hour
		self.macro_cell_load = []  # per hour
		self.internet_traffic_demand = []  # per hour
		self.small_cell_load = []
		self.actions = []

		past_10_minutes = self._10mins_internet_traffic_demand[:-1]
		current_10_minute = self._10mins_internet_traffic_demand[1:]

		data_len = past_10_minutes.shape[0]
		if training:
			past_10_minutes = past_10_minutes[: 9 * data_len // 10]
			current_10_minute = current_10_minute[: 9 * data_len // 10]
			self.training_env_iter = zip(past_10_minutes, current_10_minute)
		else:
			self.testing_env_iter = zip(past_10_minutes, current_10_minute)

		return past_10_minutes[0][:, 1]

	def step_10_mins(self, action, training=True):
		done = False
		if training:
			try:
				env = next(self.training_env_iter)
				reward = self._reward(env[1], action)
				env = env[0][:, 1]  # (6)
			except StopIteration:
				# logger.info('StopIteration')
				done = True
				env = None
				reward = None
		else:
			try:
				env = next(self.testing_env_iter)
				reward = self._reward(env[1], action)
				env = env[0][:, 1]
			except StopIteration:
				# logger.info('StopIteration')
				done = True
				env = None
				reward = None

		return env, reward, done


if __name__ == "__main__":
	# nine_grid()
	# loop_all_grid()
	# env = Milano_env(867)
	# env.reset()
	# env.step(5)
	generate_real_prediction_traffic_array()
	generate_without_prediction_traffic_array()
	generate_god_mode_traffic_array()
	# for n in range(500):
	# 	a = np.random.normal(2.5, 5)
	# 	while a < 0:
	# 		a = np.random.normal(2.5, 2)
		# print(a)