import os
import numpy as np
import sys
from env import Env_Config
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
import data_utility as du
import CNN_RNN
from multi_task_data import Prepare_Task_Data


def generate_new_real_prediction_traffic_array():
	# target_path = os.path.join(root_dir, 'offloading/npy/real_prediction_traffic_array.npy')
	source_path = os.path.join(root_dir, 'CNN_RNN/result/CNN_RNN/all_real_prediction_traffic_array_0718.npy')
	# source_path = os.path.join(root_dir, 'CNN_RNN/result/ARIMA/all_real_prediction_traffic_array.npy')
	data_array = du.load_array(source_path)
	row_center_list = list(range(40, 80, 3))
	col_center_list = list(range(30, 70, 3))
	row_range = (row_center_list[0] - 1, row_center_list[-1] + 1)
	col_range = (col_center_list[0] - 1, col_center_list[-1] + 1)
	data_array = data_array[:, :, row_range[0]: row_range[1], col_range[0]: col_range[1]]

	# for row_index in range(data_array.shape[2]):
	# 	for col_index in range(data_array.shape[3]):
	# 		grid_id = data_array[0, 0, row_index, col_index, 0]
	# 		if grid_id != 0:
	# 			print(grid_id)
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
					cdr_traffic_demand = np.random.normal(config.average_demand_per_CDR, config.scale_demand_per_CDR)
					while cdr_traffic_demand < 0:
						cdr_traffic_demand = np.random.normal(config.average_demand_per_CDR, 10)
					X_array[hour_index, min_index, row_index, col_index, -1] = X_array[hour_index, min_index, row_index, col_index, -1] * cdr_traffic_demand  # cdr to throughput

	return X_array
	# du.save_array(X_array, target_path)


def generate_real_prediction_traffic_array():
	'''
		call generate_new_real_prediction_traffic_array and generate_new_10mins_CDR_internet_traffic
	'''
	targer_dir = os.path.join(root_dir, 'offloading/npy/real_prediction')
	CNN_RNN.utility.check_path_exist(targer_dir)
	hour_target_path = os.path.join(targer_dir, 'hour_traffic_array.npy')
	_10mins_target_path = os.path.join(targer_dir, '10min_CDR_internet_traffic.npy')

	hour_traffic = generate_new_real_prediction_traffic_array()  # (1487, 1, 41, 41, 8)
	_10_min_traffic = generate_new_10mins_CDR_internet_traffic()  # (1487, 6, 41, 41, 3)

	_10_min_traffic = _10_min_traffic[1:]  # (1486, 6, 41, 41, 3)
	hour_traffic = hour_traffic[:-1]  # (1486, 1, 41, 41, 8)
	print('hour_traffic shape:{} _10_min_traffic shape:{}'.format(hour_traffic.shape, _10_min_traffic.shape))

	# for row_index in range(hour_traffic.shape[2]):
	# 	for col_index in range(hour_traffic.shape[3]):
	# 		print(hour_traffic[0, 0, row_index, col_index, 0])
	du.save_array(hour_traffic, hour_target_path)
	du.save_array(_10_min_traffic, _10mins_target_path)


# def generate_without_prediction_traffic_array():
# 	targer_dir = os.path.join(root_dir, 'offloading/npy/real_without_prediction')
# 	CNN_RNN.utility.check_path_exist(targer_dir)
# 	hour_target_path = os.path.join(targer_dir, 'hour_traffic_array.npy')
# 	_10mins_target_path = os.path.join(targer_dir, '10min_CDR_internet_traffic.npy')

# 	hour_traffic = generate_new_real_prediction_traffic_array()  # (1487, 1, 41, 41, 8)
# 	_10_min_traffic = generate_new_10mins_CDR_internet_traffic()  # (1487, 6, 41, 41, 3)

# 	_10_min_traffic = _10_min_traffic[2:]  # (1485, 6, 41, 41, 3)
# 	hour_traffic = hour_traffic[:-2]  # (1485, 1, 41, 41, 8)
# 	print('hour_traffic shape:{} _10_min_traffic shape:{}'.format(hour_traffic.shape, _10_min_traffic.shape))
# 	du.save_array(hour_traffic, hour_target_path)
# 	du.save_array(_10_min_traffic, _10mins_target_path)


# def generate_god_mode_traffic_array():
# 	targer_dir = os.path.join(root_dir, 'offloading/npy/real_god_prediction')
# 	CNN_RNN.utility.check_path_exist(targer_dir)
# 	hour_target_path = os.path.join(targer_dir, 'hour_traffic_array.npy')
# 	_10mins_target_path = os.path.join(targer_dir, '10min_CDR_internet_traffic.npy')

# 	hour_traffic = generate_new_real_prediction_traffic_array()  # (1487, 1, 41, 41, 8)
# 	_10_min_traffic = generate_new_10mins_CDR_internet_traffic()  # (1487, 6, 41, 41, 3)
# 	hour_traffic[:, :, :, :, 5] = hour_traffic[:, :, :, :, 2]
# 	hour_traffic[:, :, :, :, 6] = hour_traffic[:, :, :, :, 3]
# 	hour_traffic[:, :, :, :, 7] = hour_traffic[:, :, :, :, 4]
# 	_10_min_traffic = _10_min_traffic[1:]  # (1486, 6, 41, 41, 3)
# 	hour_traffic = hour_traffic[:-1]  # (1486, 1, 41, 41, 8)
# 	# print(hour_traffic[1, 0, 1, 1, 5:], hour_traffic[1, 0, 1, 1, 2:5])
# 	print('hour_traffic shape:{} _10_min_traffic shape:{}'.format(hour_traffic.shape, _10_min_traffic.shape))
# 	du.save_array(hour_traffic, hour_target_path)
# 	du.save_array(_10_min_traffic, _10mins_target_path)

def convert_prediction_to_non_prediction():
	source_path = os.path.join(root_dir, 'offloading/npy/real_prediction')
	target_path = os.path.join(root_dir, 'offloading/npy/real_without_prediction')

	_10_min_traffic = du.load_array(os.path.join(source_path, '10min_CDR_internet_traffic.npy'))
	hour_traffic = du.load_array(os.path.join(source_path, 'hour_traffic_array.npy'))
	print('origin 10 min shape:{} origin hour shape:{}'.format(_10_min_traffic.shape, hour_traffic.shape))

	_10_min_traffic = _10_min_traffic[1:]  # (1485, 6, 41, 41, 3)
	hour_traffic = hour_traffic[:-1]  # (1485, 1, 41, 41, 8)
	print('new 10 min shape:{} new hour shape:{}'.format(_10_min_traffic.shape, hour_traffic.shape))
	du.save_array(_10_min_traffic, os.path.join(target_path, '10min_CDR_internet_traffic'))
	du.save_array(hour_traffic, os.path.join(target_path, 'hour_traffic_array'))

if __name__ == "__main__":
	generate_real_prediction_traffic_array()
	# convert_prediction_to_non_prediction()