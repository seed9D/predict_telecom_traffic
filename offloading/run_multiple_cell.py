import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import json
import sys
from offloading import Run_Offloading, offloading_plot
from env import Env_Config
sys.path.append('/home/mldp/ML_with_bigdata')
import data_utility as du
import CNN_RNN.utility as utility
from multi_task_data import Prepare_Task_Data
root_dir = '/home/mldp/ML_with_bigdata'

logger = utility.setlog('run_multiple_cell')


def filter_cell_index():
	def get_cell_tower_grid_pair():
		cell_tower_with_grid = os.path.join(
			root_dir, 'cell_tower/cell_tower_with_grid.txt')
		with open(cell_tower_with_grid, 'r') as f:
			cell_grid = json.load(f)
		return cell_grid

	def evaluate_performance(Y_real_prediction_array, threshold):
		Y_real_prediction_array = np.transpose(Y_real_prediction_array, (2, 3, 0, 1, 4))
		grid_id_list = []
		for row_index in range(Y_real_prediction_array.shape[0]):
			for col_index in range(Y_real_prediction_array.shape[1]):
				info = Y_real_prediction_array[row_index, col_index, :, 0, :2]
				real = Y_real_prediction_array[row_index, col_index, :, 0, 2:5]
				prediction = Y_real_prediction_array[row_index, col_index, :, 0, 5:]
				# task_min_MAPE = utility.MAPE_loss(real[:, 0], prediction[:, 0])
				# task_avg_MAPE = utility.MAPE_loss(real[:, 1], prediction[:, 1])
				# task_max_MAPE = utility.MAPE_loss(real[:, 2], prediction[:, 2])
				MAPE = utility.MAPE_loss(real, prediction)
				Accu = 1 - MAPE if MAPE else 0

				if Accu > threshold and Accu:
					grid_id = info[0, 0]
					# print('grid id:{} accu:{}'.format(grid_id, Accu))
					grid_id_list.append(int(grid_id))
		return grid_id_list

	evaluate_threshold = 0.75
	all_real_prediction_traffic_array_path = os.path.join(root_dir, 'offloading/npy/real_prediction/hour_traffic_array.npy')

	CNN_RNN_MTL_array = du.load_array(all_real_prediction_traffic_array_path)
	grid_id_list = evaluate_performance(CNN_RNN_MTL_array, evaluate_threshold)
	cell_grids = get_cell_tower_grid_pair()
	cell_index_list = []
	for cell_grid in cell_grids:
		cell_index = cell_grid['index']
		grids = cell_grid['grid']
		# print(cell_grid)
		if set(grids).issubset(set(grid_id_list)) and len(grids) > 0:
			cell_index_list.append(cell_index)
	print('cell_index_list length:', len(cell_index_list))
	cell_index_list = sorted(cell_index_list)
	return cell_index_list


def dict_to_nparray(result_dict, cell_num):
	energy_effi_array = result_dict['energy_effi']
	reward_array = result_dict['reward']
	traffic_demand_array = result_dict['traffic_demand']
	macro_load_array = result_dict['macro_load']
	small_load_array = result_dict['small_load']
	action_array = result_dict['action']
	power_consumption_array = result_dict['power_consumption']
	cell_num_array = np.zeros_like(reward_array, dtype=float)
	cell_num_array.fill(cell_num)

	result_array = np.concatenate((cell_num_array, reward_array, energy_effi_array, traffic_demand_array, macro_load_array, small_load_array, power_consumption_array, action_array), axis=1)
	return result_array


def save_report(result_array, file_path):
	result_array = result_array[-149:]
	cell_num = result_array[0, 0]
	macro_load = np.mean(result_array[:, 4])
	small_load = result_array[:, 5]
	small_load[small_load == 0] = np.nan
	small_load = np.nanmean(small_load)
	energy_effi = np.mean(result_array[:, 2])
	string_ = 'cell_num:{} Macro_load:{:.2f} Small load:{:.2f} EE:{:.4f}\n'.format(cell_num, macro_load, small_load, energy_effi)

	with open(file_path, 'a',) as f:
		f.write(string_)


def run_prediction_and_RL(cell_list):

	def run_offloading(cell_num):
		config = Env_Config()
		offloading = Run_Offloading(config, cell_num)
		without_RL_with_offloading_result_dict = offloading.run_test_without_RL(10)
		without_RL_offloading_result_dict = offloading.run_test_without_RL(0)

		logger.info('macro load:{} offloading_without_RL:{} without_RL_without_offloading:{}'.format(
			np.mean(without_RL_offloading_result_dict['macro_load'][-149:]),
			np.mean(without_RL_with_offloading_result_dict['energy_effi'][-149:]),
			np.mean(without_RL_offloading_result_dict['energy_effi'][-149:])))
		offloading.RL_train()
		with_RL_result_dict = offloading.run_test_with_RL(reload=False)
		with_RL_result_array = dict_to_nparray(with_RL_result_dict, cell_num)
		# without_RL_with_offloading_result_array = dict_to_nparray(without_RL_with_offloading_result_dict)
		# without_RL_offloading_result_array = dict_to_nparray(without_RL_offloading_result_dict)
		offloading_plot(with_RL_result_dict, without_RL_with_offloading_result_dict, without_RL_offloading_result_dict)
		return with_RL_result_array

	plt.ion()

	target_path = os.path.join(root_dir, 'offloading/result/with_preidction_with_RL')
	utility.check_path_exist(target_path)
	store_path = os.path.join(target_path, 'loop_report.txt')
	if os.path.exists(store_path):
		os.remove(store_path)

	cell_list_len = len(cell_list)
	all_cell_result_array_list = []
	for cell_num in cell_list:

		with_RL_result_array = run_offloading(cell_num)
		all_cell_result_array_list.append(with_RL_result_array)
		logger.info('cell_num:{} average effi mean:{}'.format(cell_num, np.mean(with_RL_result_array[-149:, 2])))
		save_report(with_RL_result_array, store_path)
		all_cell_result_array = np.stack(all_cell_result_array_list, axis=0)
		du.save_array(all_cell_result_array, os.path.join(target_path, 'all_cell_result_array.npy'))

	all_cell_result_array = np.stack(all_cell_result_array_list, axis=0)
	du.save_array(all_cell_result_array, os.path.join(target_path, 'all_cell_result_array.npy'))
	plt.ioff()


def run_offloading_without_RL(cell_list):
	def run(cell_num):
		config = Env_Config()
		offloading = Run_Offloading(config, cell_num)
		without_RL_with_offloading_result_dict = offloading.run_test_without_RL(10)
		without_RL_with_offloading_result_array = dict_to_nparray(without_RL_with_offloading_result_dict, cell_num)

		return without_RL_with_offloading_result_array

	target_path = os.path.join(root_dir, 'offloading/result/offloading_without_RL')
	utility.check_path_exist(target_path)
	store_path = os.path.join(target_path, 'loop_report.txt')
	if os.path.exists(store_path):
		os.remove(store_path)
	cell_list_len = len(cell_list)
	all_cell_result_array_list = []
	for cell_num in cell_list:
		logger.info('cell_num:{}'.format(cell_num))
		result_array = run(cell_num)
		all_cell_result_array_list.append(result_array)
		save_report(result_array, store_path)
		all_cell_result_array = np.stack(all_cell_result_array_list, axis=0)
		du.save_array(all_cell_result_array, os.path.join(target_path, 'all_cell_result_array.npy'))
	all_cell_result_array = np.stack(all_cell_result_array_list, axis=0)
	du.save_array(all_cell_result_array, os.path.join(target_path, 'all_cell_result_array.npy'))
	plt.ioff()


def run_without_offloading(cell_list):
	def run(cell_num):
		config = Env_Config()
		offloading = Run_Offloading(config, cell_num)
		result_dict = offloading.run_test_without_RL(0)
		result_array = dict_to_nparray(result_dict, cell_num)

		return result_array

	target_path = os.path.join(root_dir, 'offloading/result/without_offloading_without_RL')
	utility.check_path_exist(target_path)
	store_path = os.path.join(target_path, 'loop_report.txt')
	if os.path.exists(store_path):
		os.remove(store_path)
	cell_list_len = len(cell_list)
	all_cell_result_array_list = []

	for cell_num in cell_list:
		logger.info('cell_num:{}'.format(cell_num))
		result_array = run(cell_num)
		all_cell_result_array_list.append(result_array)
		save_report(result_array, store_path)
		all_cell_result_array = np.stack(all_cell_result_array_list, axis=0)
		du.save_array(all_cell_result_array, os.path.join(target_path, 'all_cell_result_array.npy'))
	all_cell_result_array = np.stack(all_cell_result_array_list, axis=0)
	du.save_array(all_cell_result_array, os.path.join(target_path, 'all_cell_result_array.npy'))
	plt.ioff()


def run_god_prediction_and_RL(cell_list):
	def run_offloading(cell_num):
		config = Env_Config()
		config.base_dir = os.path.join(root_dir, 'offloading/npy/real_god_prediction')
		offloading = Run_Offloading(config, cell_num)
		offloading_without_RL_result_dict = offloading.run_test_without_RL(10)
		without_offloading_result_dict = offloading.run_test_without_RL(0)

		logger.info('macro load:{} offloading_without_RL:{} without_offloading:{}'.format(
			np.mean(without_offloading_result_dict['macro_load'][-149:]),
			np.mean(offloading_without_RL_result_dict['energy_effi'][-149:]),
			np.mean(without_offloading_result_dict['energy_effi'][-149:])))
		offloading.RL_train()
		result_dict = offloading.run_test_with_RL(reload=False)
		result_array = dict_to_nparray(result_dict, cell_num)
		# without_RL_with_offloading_result_array = dict_to_nparray(without_RL_with_offloading_result_dict)
		# without_RL_offloading_result_array = dict_to_nparray(without_RL_offloading_result_dict)
		offloading_plot(result_dict, offloading_without_RL_result_dict, without_offloading_result_dict)
		return result_array

	plt.ion()
	target_path = os.path.join(root_dir, 'offloading/result/RL_with_god_prediction')
	utility.check_path_exist(target_path)

	cell_list_len = len(cell_list)
	all_cell_result_array_list = []
	for cell_num in cell_list:

		result_array = run_offloading(cell_num)
		all_cell_result_array_list.append(result_array)
		logger.info('cell_num:{} average effi mean:{}'.format(cell_num, np.mean(result_array[-149:, 2])))
		all_cell_result_array = np.stack(all_cell_result_array_list, axis=0)
		du.save_array(all_cell_result_array, os.path.join(target_path, 'all_cell_result_array.npy'))

	all_cell_result_array = np.stack(all_cell_result_array_list, axis=0)
	du.save_array(all_cell_result_array, os.path.join(target_path, 'all_cell_result_array.npy'))
	plt.ioff()


def run_without_preidction_and_with_RL(cell_list):
	def run_offloading(cell_num):
		config = Env_Config()
		config.base_dir = os.path.join(root_dir, 'offloading/npy/real_without_prediction')
		offloading = Run_Offloading(config, cell_num)
		offloading_without_RL_result_dict = offloading.run_test_without_RL(10)
		without_offloading_result_dict = offloading.run_test_without_RL(0)

		logger.info('macro load:{} offloading_without_RL:{} without_offloading:{}'.format(
			np.mean(without_offloading_result_dict['macro_load'][-149:]),
			np.mean(offloading_without_RL_result_dict['energy_effi'][-149:]),
			np.mean(without_offloading_result_dict['energy_effi'][-149:])))
		offloading.RL_train()
		result_dict = offloading.run_test_with_RL(reload=False)
		result_array = dict_to_nparray(result_dict, cell_num)
		# without_RL_with_offloading_result_array = dict_to_nparray(without_RL_with_offloading_result_dict)
		# without_RL_offloading_result_array = dict_to_nparray(without_RL_offloading_result_dict)
		offloading_plot(result_dict, offloading_without_RL_result_dict, without_offloading_result_dict)
		return result_array

	plt.ion()
	target_path = os.path.join(root_dir, 'offloading/result/RL_without_prediction')
	utility.check_path_exist(target_path)

	cell_list_len = len(cell_list)
	all_cell_result_array_list = []
	for cell_num in cell_list:

		result_array = run_offloading(cell_num)
		all_cell_result_array_list.append(result_array)
		logger.info('cell_num:{} average effi mean:{}'.format(cell_num, np.mean(result_array[-149:, 2])))

		all_cell_result_array = np.stack(all_cell_result_array_list, axis=0)
		du.save_array(all_cell_result_array, os.path.join(target_path, 'all_cell_result_array.npy'))

	all_cell_result_array = np.stack(all_cell_result_array_list, axis=0)
	du.save_array(all_cell_result_array, os.path.join(target_path, 'all_cell_result_array.npy'))
	plt.ioff()


def run_without_prediction_and_RL_10mins(cell_list):

	def run_offloading(cell_num):
		config = Env_Config()
		config.features_num = 6
		offloading = Run_Offloading(config, cell_num)
		without_RL_with_offloading_result_dict = offloading.run_test_without_RL(10)
		without_RL_offloading_result_dict = offloading.run_test_without_RL(0)

		logger.info('macro load:{} offloading_without_RL:{} without_RL_without_offloading:{}'.format(
			np.mean(without_RL_offloading_result_dict['macro_load'][-149:]),
			np.mean(without_RL_with_offloading_result_dict['energy_effi'][-149:]),
			np.mean(without_RL_offloading_result_dict['energy_effi'][-149:])))
		offloading.RL_train(mins=True)
		with_RL_result_dict = offloading.run_test_with_RL(reload=False, mins=True)
		with_RL_result_array = dict_to_nparray(with_RL_result_dict, cell_num)
		# without_RL_with_offloading_result_array = dict_to_nparray(without_RL_with_offloading_result_dict)
		# without_RL_offloading_result_array = dict_to_nparray(without_RL_offloading_result_dict)
		offloading_plot(with_RL_result_dict, without_RL_with_offloading_result_dict, without_RL_offloading_result_dict)
		return with_RL_result_array

	plt.ion()
	target_path = os.path.join(root_dir, 'offloading/result/RL_10mins_without_prediction')
	utility.check_path_exist(target_path)
	store_path = os.path.join(target_path, 'loop_report.txt')
	if os.path.exists(store_path):
		os.remove(store_path)

	cell_list_len = len(cell_list)
	all_cell_result_array_list = []
	for cell_num in cell_list:

		with_RL_result_array = run_offloading(cell_num)
		all_cell_result_array_list.append(with_RL_result_array)
		logger.info('cell_num:{} average effi mean:{}'.format(cell_num, np.mean(with_RL_result_array[-149:, 2])))
		save_report(with_RL_result_array, store_path)
		all_cell_result_array = np.stack(all_cell_result_array_list, axis=0)
		du.save_array(all_cell_result_array, os.path.join(target_path, 'all_cell_result_array.npy'))

	all_cell_result_array = np.stack(all_cell_result_array_list, axis=0)
	du.save_array(all_cell_result_array, os.path.join(target_path, 'all_cell_result_array.npy'))
	plt.ioff()


def run_all_method():
	cell_list = filter_cell_index()
	print(cell_list)
	# run_prediction_and_RL(cell_list)
	# run_without_preidction_and_with_RL(cell_list)

	# run_offloading_without_RL(cell_list)
	# run_without_offloading(cell_list)

	run_without_prediction_and_RL_10mins(cell_list)
	# run_god_prediction_and_RL(cell_list)


if __name__ == '__main__':

	run_all_method()
