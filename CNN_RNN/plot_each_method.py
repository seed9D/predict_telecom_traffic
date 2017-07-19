import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from operator import itemgetter, getitem
from itertools import groupby
from functools import reduce
import json
import os
import report_func
import utility
import sys
sys.path.append('/home/mldp/ML_with_bigdata')
import data_utility as du

method_LM_result_path = '/home/qiuhui/processed_data/'
method_CNN_RNN_result_path = '/home/mldp/ML_with_bigdata/CNN_RNN/result/CNN_RNN/Y_real_prediction.npy'
method_CNN_3D_result_path = '/home/mldp/ML_with_bigdata/CNN_RNN/result/CNN_3D/Y_real_prediction.npy'
method_RNN_result_path = '/home/mldp/ML_with_bigdata/CNN_RNN/result/RNN/Y_real_prediction.npy'

plot_grid_id_list = [4258, 4456, 4457]
time_shift = 2


def get_CNN_RNN():
	CNN_RNN_prediction = du.load_array(method_CNN_RNN_result_path)
	info = CNN_RNN_prediction[time_shift:, :, :, :, :2]
	real = CNN_RNN_prediction[time_shift:, :, :, :, 2:5]
	CNN_RNN_prediction = CNN_RNN_prediction[time_shift:, :, :, :, 5:]
	return info, real, CNN_RNN_prediction


def get_CNN_3D():
	CNN_3D_prediction = du.load_array(method_CNN_3D_result_path)
	info = CNN_3D_prediction[time_shift:, :, :, :, :2]
	real = CNN_3D_prediction[time_shift:, :, :, :, 2:5]
	CNN_3D_prediction = CNN_3D_prediction[time_shift:, :, :, :, 5:]
	return info, real, CNN_3D_prediction


def get_RNN():
	RNN_prediction = du.load_array(method_RNN_result_path)
	info = RNN_prediction[time_shift:, :, :, :, :2]
	real = RNN_prediction[time_shift:, :, :, :, 2:5]
	RNN_prediction = RNN_prediction[time_shift:, :, :, :, 5:]
	return info, real, RNN_prediction


def get_LM(grid_id, task_name):
	min_file = os.path.join(method_LM_result_path, str(grid_id) + '_' + task_name + '.npy')
	min_array = du.load_array(min_file)
	return min_array


def plot_all_method_together(plot_dict, task_name):
	def get_xlabel(timestamps):
		xlabel_list = []
		for timestamp in timestamps:
			datetime = utility.set_time_zone(timestamp)
			xlabel_list.append(utility.date_time_covert_to_str(datetime))
		return xlabel_list

	xlabel_list = get_xlabel(plot_dict['info'][:, 1])
	x_len = len(xlabel_list)
	grid_id = plot_dict['info'][0, 0]
	fig, ax = plt.subplots(1, 1)
	ax.set_xlabel('time sequence')
	ax.set_ylabel('number of CDR')

	ax.plot(range(x_len), plot_dict['real'], label='Real', color='k')
	ax.plot(range(x_len), plot_dict['CNN_RNN'], label='CNN_RNN', marker='.')
	ax.plot(range(x_len), plot_dict['CNN_3D'], label='CNN_3D', linestyle='--')
	ax.plot(range(x_len), plot_dict['RNN'], label='RNN', marker='v')
	# ax.plot(range(x_len), plot_dict['LM'], label='Levenberg–Marquardt', marker='x')
	ax.set_xticks(list(range(0, x_len, 2)))
	ax.set_xticklabels(xlabel_list[0:x_len:2], rotation=45)
	ax.set_title(task_name + ' Grid id: ' + str(int(grid_id)))
	ax.grid()
	ax.legend()

	return fig


def plot_all_task_together(plot_dict):
	def get_xlabel(timestamps):
		xlabel_list = []
		for timestamp in timestamps:
			datetime = utility.set_time_zone(timestamp)
			xlabel_list.append(utility.date_time_covert_to_str(datetime))
		return xlabel_list

	def plot_by_task(axis, xlabel_list, plot_dict, task_name):
		x_len = len(xlabel_list)
		axis.plot(range(x_len), plot_dict['real'], label='Real', color='k')
		axis.plot(range(x_len), plot_dict['CNN_RNN'], label='CNN_RNN', linestyle='--')
		axis.plot(range(x_len), plot_dict['CNN_3D'], label='CNN_3D', linestyle='--')
		axis.plot(range(x_len), plot_dict['RNN'], label='RNN', linestyle='--')
		axis.set_xticks(list(range(0, x_len, 6)))
		axis.set_xticklabels(xlabel_list[0:x_len:6], rotation=45)

		axis.grid()
		axis.legend()
		axis.set_title(task_name)

	xlabel_list = get_xlabel(plot_dict['info'][:, 1])
	grid_id = plot_dict['info'][0, 0]
	fig = plt.figure()
	ax_min = fig.add_subplot(311)
	ax_avg = fig.add_subplot(312)
	ax_max = fig.add_subplot(313)


	min_dict = {
		'real': plot_dict['real'][:, 0],
		'CNN_RNN': plot_dict['CNN_RNN'][:, 0],
		'RNN': plot_dict['RNN'][:, 0],
		'CNN_3D': plot_dict['CNN_3D'][:, 0]
	}
	plot_by_task(ax_min, xlabel_list, min_dict, 'Task_min')

	avg_dict = {
		'real': plot_dict['real'][:, 1],
		'CNN_RNN': plot_dict['CNN_RNN'][:, 1],
		'RNN': plot_dict['RNN'][:, 1],
		'CNN_3D': plot_dict['CNN_3D'][:, 1]
	}
	plot_by_task(ax_avg, xlabel_list, avg_dict, 'Task_avg')

	max_dict = {
		'real': plot_dict['real'][:, 2],
		'CNN_RNN': plot_dict['CNN_RNN'][:, 2],
		'RNN': plot_dict['RNN'][:, 2],
		'CNN_3D': plot_dict['CNN_3D'][:, 2]
	}
	plot_by_task(ax_max, xlabel_list, max_dict, 'Task_max')
	plt.xlabel('time sequence')
	plt.ylabel('number of CDR')
	plt.title('Grid id:' + str(int(grid_id)))


def plot_min_task_all_together(plot_dict):
	row = 0
	col = 0
	interval = (9, 40)

	plot_dict_min = {
		'info': plot_dict['info'][interval[0]:interval[1], 0, row, col],
		'real': plot_dict['real'][interval[0]:interval[1], 0, row, col, 0],
		'CNN_RNN': plot_dict['CNN_RNN'][interval[0]:interval[1], 0, row, col, 0],
		'CNN_3D': plot_dict['CNN_3D'][interval[0]:interval[1], 0, row, col, 0],
		'RNN': plot_dict['RNN'][interval[0]:interval[1], 0, row, col, 0]
	}

	grid_id = int(plot_dict_min['info'][0, 0])
	LM_array = get_LM(grid_id, 'min')[interval[0]:interval[1]]
	plot_dict_min['LM'] = LM_array
	# print(plot_dict_min['CNN_3D'].shape)
	# print(LM_array.shape)
	plot_all_method_together(plot_dict_min, 'Task_min')


def plot_avg_task_all_together(plot_dict):
	row = 2
	col = 0
	interval = (100, 131)
	plot_dict_avg = {
		'info': plot_dict['info'][interval[0]:interval[1], 0, row, col],
		'real': plot_dict['real'][interval[0]:interval[1], 0, row, col, 1],
		'CNN_RNN': plot_dict['CNN_RNN'][interval[0]:interval[1], 0, row, col, 1],
		'CNN_3D': plot_dict['CNN_3D'][interval[0]:interval[1], 0, row, col, 1],
		'RNN': plot_dict['RNN'][interval[0]:interval[1], 0, row, col, 1]
	}
	grid_id = int(plot_dict_avg['info'][0, 0])

	plot_all_method_together(plot_dict_avg, 'Task_avg')


def get_each_method():
	info, real, CNN_RNN_prediction = get_CNN_RNN()
	_, _, CNN_3D_prediction = get_CNN_3D()
	_, _, RNN_prediction = get_RNN()

	plot_dict = {
		'info': info,
		'real': real,
		'CNN_RNN': CNN_RNN_prediction,
		'CNN_3D': CNN_3D_prediction,
		'RNN': RNN_prediction
	}
	# row = 0
	# col = 0
	# plot_dict_1 ={
	# 	'info': info[:, 0, row, col],
	# 	'real': real[:, 0, row, col],
	# 	'CNN_RNN': CNN_RNN_prediction[:, 0, row, col],
	# 	'CNN_3D': CNN_3D_prediction[:, 0, row, col],
	# 	'RNN': RNN_prediction[:, 0, row, col]
	# }
	# plot_all_task_together(plot_dict_1)
	# plot_min_task_all_together(plot_dict)
	# 
	plot_avg_task_all_together(plot_dict)
	plt.show()


def evaluate_accuracy_composition():
	def task_value(obj, key_paths):
		key_paths.sort(key=itemgetter(1), reverse=False)  # sorted by task_name
		uni_task_key = []
		task_groups = []
		for k, g in groupby(key_paths, lambda x: x[1]):  # group by task
			uni_task_key.append(k)
			task_groups.append(list(g))
		task_value_dict = {}
		for task_index, each_task_name in enumerate(uni_task_key):
			each_task_group = task_groups[task_index]
			task_iter = iter(each_task_group)
			values = [reduce(getitem, ele, obj) for ele in task_iter]
			task_value_dict[each_task_name] = values

		return task_value_dict

	def plot_count_bar(data_frame, title):
		bins = np.arange(0, 1.1, 0.1, dtype=np.float)
		# print(task_max_df)
		cats_CNN_RNN_MTL = pd.cut(data_frame['CNN RNN(MTL)'], bins)
		cats_CNN_RNN_STL = pd.cut(data_frame['CNN RNN(STL)'], bins)
		cats_CNN_3D = pd.cut(data_frame['CNN 3D'], bins)
		cats_RNN = pd.cut(data_frame['RNN'], bins)
		cats_ARIMA = pd.cut(data_frame['ARIMA'], bins)
		# print(cats)
		CNN_RNN_MTL_grouped = data_frame['CNN RNN(MTL)'].groupby(cats_CNN_RNN_MTL)
		CNN_RNN_STL_grouped = data_frame['CNN RNN(STL)'].groupby(cats_CNN_RNN_STL)
		CNN_3D_grouped = data_frame['CNN 3D'].groupby(cats_CNN_3D)
		RNN_grouped = data_frame['RNN'].groupby(cats_RNN)
		ARIMA_grouped = data_frame['ARIMA'].groupby(cats_ARIMA)

		CNN_RNN_MTL_bin_counts = CNN_RNN_MTL_grouped.count()
		CNN_RNN_STL_bin_counts = CNN_RNN_STL_grouped.count()
		CNN_3D_bin_counts = CNN_3D_grouped.count()
		RNN_bin_counts = RNN_grouped.count()
		ARIMA_bin_counts = ARIMA_grouped.count()
		# print(CNN_3D_bin_counts)

		bin_counts = pd.concat([ARIMA_bin_counts, RNN_bin_counts, CNN_3D_bin_counts, CNN_RNN_MTL_bin_counts, CNN_RNN_STL_bin_counts], axis=1)
		bin_counts.columns = ['ARIMA', 'RNN', 'CNN 3D', 'CNN RNN(MTL)', 'CNN RNN(STL)']
		bin_counts.index = ['0~10', '10~20', '20~30', '30~40', '40~50', '50~60', '60~70', '70~80', '80~90', '90~100']
		bin_counts.index.name = 'Accuracy %'
		ax = bin_counts.plot(kind='bar', alpha=0.7, rot=0, width=0.8)
		for p in ax.patches:
			ax.annotate(str(int(p.get_height())), xy=(p.get_x(), p.get_height()))

		plt.legend(loc='upper left', fontsize = 'large')
		plt.title(title)
		plt.ylabel('number of grids')
		# print(bin_counts)

	def plot_KDE(data_frame, title):
		data_frame = data_frame[['ARIMA', 'RNN', 'CNN 3D', 'CNN RNN(MTL)', 'CNN RNN(STL)']]
		ax = data_frame.plot(kind='kde', title=title + ' KDE plot')
		ax.set_xlabel('Accuracy')
		ax.set_xlim(0.1, 1)

	def convert_to_data_frame_by_task(CNN_RNN, CNN_3D, RNN, ARIMA, CNN_RNN_STL):
		CNN_RNN_key_paths = list(utility.find_in_obj(CNN_RNN, 'Accuracy'))
		CNN_3D_key_paths = list(utility.find_in_obj(CNN_3D, 'Accuracy'))
		RNN_key_paths = list(utility.find_in_obj(RNN, 'Accuracy'))
		ARIMA_key_paths = list(utility.find_in_obj(ARIMA, 'Accuracy'))
		CNN_RNN_STL_key_paths = list(utility.find_in_obj(CNN_RNN_STL, 'Accuracy'))

		CNN_RNN_accu_dict = task_value(CNN_RNN, CNN_RNN_key_paths)
		CNN_3D_accu_dict = task_value(CNN_3D, CNN_3D_key_paths)
		RNN_accu_dict = task_value(RNN, RNN_key_paths)
		ARIMA_accu_dict = task_value(ARIMA, ARIMA_key_paths)
		CNN_RNN_STL_accu_dict = task_value(CNN_RNN_STL, CNN_RNN_STL_key_paths)
		# print(CNN_RNN_accu_dict['task_max'][-1], CNN_3D_accu_dict['task_max'][-1])

		task_max_df = pd.DataFrame({
			'CNN RNN(MTL)': CNN_RNN_accu_dict['task_max'],
			'CNN RNN(STL)': CNN_RNN_STL_accu_dict['task_max'],
			'CNN 3D': CNN_3D_accu_dict['task_max'],
			'RNN': RNN_accu_dict['task_max'],
			'ARIMA': ARIMA_accu_dict['task_max']})
		task_max_df = task_max_df.dropna(axis=0, how='any')
		# print(task_max_df.shape)

		task_avg_df = pd.DataFrame({
			'CNN RNN(MTL)': CNN_RNN_accu_dict['task_avg'],
			'CNN RNN(STL)': CNN_RNN_STL_accu_dict['task_avg'],
			'CNN 3D': CNN_3D_accu_dict['task_avg'],
			'RNN': RNN_accu_dict['task_avg'],
			'ARIMA': ARIMA_accu_dict['task_avg']})
		task_avg_df = task_avg_df.dropna(axis=0, how='any')

		task_min_df = pd.DataFrame({
			'CNN RNN(MTL)': CNN_RNN_accu_dict['task_min'],
			'CNN RNN(STL)': CNN_RNN_STL_accu_dict['task_min'],
			'CNN 3D': CNN_3D_accu_dict['task_min'],
			'RNN': RNN_accu_dict['task_min'],
			'ARIMA': ARIMA_accu_dict['task_min']})
		task_min_df = task_min_df.dropna(axis=0, how='any')

		return task_min_df, task_avg_df, task_max_df

	CNN_RNN_all_grid_result = './result/CNN_RNN/all_grid_result_report_0718.txt'
	CNN_3D_all_grid_result = './result/CNN_3D/all_grid_result_report.txt'
	RNN_all_grid_result = './result/RNN/all_grid_result_report.txt'
	ARIMA_all_grid_result = './result/ARIMA/all_grid_result_report.txt'
	CNN_RNN_STL_all_grid_result = './result/CNN_RNN_STL/all_grid_result_report.txt'

	with open(CNN_RNN_all_grid_result, 'r') as fp:
		CNN_RNN = json.load(fp, encoding=None)

	with open(CNN_3D_all_grid_result, 'r') as fp:
		CNN_3D = json.load(fp, encoding=None)

	with open(RNN_all_grid_result, 'r') as fp:
		RNN = json.load(fp, encoding=None)

	with open(ARIMA_all_grid_result, 'r') as fp:
		ARIMA = json.load(fp, encoding=None)

	with open(CNN_RNN_STL_all_grid_result, 'r') as fp:
		CNN_RNN_STL = json.load(fp, encoding=None)

	task_min, task_avg, task_max = convert_to_data_frame_by_task(CNN_RNN, CNN_3D, RNN, ARIMA, CNN_RNN_STL)
	
	print(task_min.describe())
	print(task_avg.describe())
	print(task_max.describe())

	# task_min.plot.hist()
	# task_avg.hist()
	# task_max.plot.hist()
	plot_KDE(task_min, 'Task Min')
	plot_KDE(task_avg, 'Task avg')
	plot_KDE(task_max, 'Task Max')
	# task_avg.plot(kind='kde')
	plot_count_bar(task_min, 'Task Min')
	plot_count_bar(task_avg, 'Task Avg')
	plot_count_bar(task_max, 'Task Max')

	plt.show()


def evaluate_different_method():
	def evaluate_performance(Y_real_prediction_array, file_path):
		def print_total_report(task_report):
			for task_name, ele in task_report.items():
				print('{}: Accuracy:{:.4f} MAE:{:.4f} RMSE:{:.4f}'.format(task_name, ele['Accuracy'], ele['AE'], ele['RMSE']))

		row_center_list = list(range(40, 80, 3))
		col_center_list = list(range(30, 70, 3))
		row_range = (row_center_list[0], row_center_list[-1])
		col_range = (col_center_list[0], col_center_list[-1])
		# print((row_range[1] - row_range[0]) * (col_range[1] -  col_range[0]))
		array_len = Y_real_prediction_array.shape[0]
		divide_threshold = (9 * array_len) // 10

		Y_real_prediction_array = Y_real_prediction_array[:, :, row_range[0]: row_range[1], col_range[0]: col_range[1]]
		training_data = Y_real_prediction_array[:divide_threshold]
		testing_data = Y_real_prediction_array[divide_threshold:]

		training_info = training_data[:, :, :, :, :2]
		training_real = training_data[:, :, :, :, 2:5]
		training_prediction = training_data[:, :, :, :, 5:]

		testing_info = testing_data[:, :, :, :, :2]
		testing_real = testing_data[:, :, :, :, 2:5]
		testing_prediction = testing_data[:, :, :, :, 5:]
		report_dict = report_func.report_loss_accu(testing_info, testing_real, testing_prediction, file_path)
		print_total_report(report_dict['total'])
		# print(report_dict['total'])
	CNN_RNN_all_grid_path = './result/CNN_RNN/all_real_prediction_traffic_array.npy'
	CNN_3D_all_grid_path = './result/CNN_3D/all_real_prediction_traffic_array_0718.npy'
	RNN_all_grid_path = './result/RNN/all_real_prediction_traffic_array_0718.npy'
	ARIMA_all_grid_path = './result/ARIMA/all_real_prediction_traffic_array.npy'
	CNN_RNN_STL_all_grid_path = './result/CNN_RNN_STL/all_real_prediction_traffic_array_0715.npy'

	# CNN_RNN_array = du.load_array(CNN_RNN_all_grid_path)
	# CNN_3D_array = du.load_array(CNN_3D_all_grid_path)
	# RNN_array = du.load_array(RNN_all_grid_path)
	ARIMA_array = du.load_array(ARIMA_all_grid_path)
	# CNN_RNN_STL_array = du.load_array(CNN_RNN_STL_all_grid_path)

	# evaluate_performance(CNN_RNN_array, './result/CNN_RNN/all_grid_result_report.txt')
	# evaluate_performance(CNN_3D_array, './result/CNN_3D/all_grid_result_report.txt')
	# evaluate_performance(RNN_array, './result/RNN/all_grid_result_report.txt')
	evaluate_performance(ARIMA_array, './result/ARIMA/all_grid_result_report.txt')
	# evaluate_performance(CNN_RNN_STL_array, './result/CNN_RNN_STL/all_grid_result_report.txt')


def evaluate_MTL_and_STL():
	def task_value(obj, key_paths):
		key_paths.sort(key=itemgetter(1), reverse=False)  # sorted by task_name
		uni_task_key = []
		task_groups = []
		for k, g in groupby(key_paths, lambda x: x[1]):  # group by task
			uni_task_key.append(k)
			task_groups.append(list(g))
		task_value_dict = {}
		for task_index, each_task_name in enumerate(uni_task_key):
			each_task_group = task_groups[task_index]
			task_iter = iter(each_task_group)
			# values = [reduce(getitem, ele, obj) for ele in task_iter]
			grid_id_accu_list = []
			for ele in task_iter:
				accu = reduce(getitem, ele, obj)
				grid_id_accu_list.append((ele[0], accu))  # (grid_id, accu)
				grid_id_accu_list = sorted(grid_id_accu_list, key=itemgetter(0))  # sort by grid id
			task_value_dict[each_task_name] = grid_id_accu_list

		return task_value_dict

	def convert_to_data_frame_by_task(method_dict):

		def get_data_frame(method_dict, task_name):
			MTL_task_df = pd.DataFrame({
				'Grid_id': [ele[0] for ele in method_dict['CNN_RNN_MTL'][task_name]],
				'MTL': [ele[1] for ele in method_dict['CNN_RNN_MTL'][task_name]]})
			MTL_task_df = MTL_task_df.set_index('Grid_id')

			STL_task_df = pd.DataFrame({
				'Grid_id': [ele[0] for ele in method_dict['CNN_RNN_STL'][task_name]],
				'STL': [ele[1] for ele in method_dict['CNN_RNN_STL'][task_name]]})
			STL_task_df = STL_task_df.set_index('Grid_id')
			task_df = pd.concat([MTL_task_df, STL_task_df], axis=1, join='outer')
			task_df = task_df.dropna(axis=0, how='any')
			return task_df

		for key, obj in method_dict.items():
			obj_key_path = list(utility.find_in_obj(obj, 'Accuracy'))  # key_path :[grid_id, task_type, 'Accuracy']
			acc_dict = task_value(obj, obj_key_path)
			method_dict[key] = acc_dict

		max_df = get_data_frame(method_dict, 'task_max')
		avg_df = get_data_frame(method_dict, 'task_avg')
		min_df = get_data_frame(method_dict, 'task_min')
		return min_df, avg_df, max_df
		# print(method_dict)

	def plot_improvement_heat_map(task_df):
		improve_df = (task_df.loc[:, 'MTL'] - task_df.loc[:, 'STL']) * 100
		improve_df.drop('total', inplace=True)
		data_array = np.zeros([100, 100], dtype=float)
		for i, value in enumerate(improve_df.values):
			grid_id = int(improve_df.index[i])
			row, col = utility.compute_row_col(grid_id)
			data_array[row, col] = value

		plt.imshow(data_array.T, vmin=-10, vmax=10, cmap=plt.get_cmap('bwr'))
		plt.grid(True)
		plt.colorbar()
		plt.show()

	def compare_two_task(task_df):
		task_df['larger'] = task_df.apply(lambda x: 'MTL' if x['MTL'] > x['STL'] else 'STL', axis=1)
		print(task_df.describe())
		df_larger = task_df.loc[task_df['larger'] == 'MTL']
		# df_larger['impove'] = task_df.apply(lambda x: (x['MTL'] - x['STL']), axis=1)
		df_improve = (df_larger.loc[:, 'MTL'] - df_larger.loc[:, 'STL']) * 100
		print(df_improve.describe())
		# print(task_df['larger'].value_counts())
		

	CNN_RNN_MTL_all_grid_result = './result/CNN_RNN/all_grid_result_report_0718.txt'
	CNN_RNN_STL_all_grid_result = './result/CNN_RNN_STL/all_grid_result_report.txt'

	with open(CNN_RNN_MTL_all_grid_result, 'r') as fp:
		CNN_RNN_MTL = json.load(fp, encoding=None)

	with open(CNN_RNN_STL_all_grid_result, 'r') as fp:
		CNN_RNN_STL = json.load(fp, encoding=None)

	method_dict = {
		'CNN_RNN_MTL': CNN_RNN_MTL,
		'CNN_RNN_STL': CNN_RNN_STL
	}
	min_task, avg_task, max_task = convert_to_data_frame_by_task(method_dict)
	# compare_two_task(min_task)
	# compare_two_task(avg_task)
	compare_two_task(max_task)
	plot_improvement_heat_map(max_task)


if __name__ == '__main__':
	evaluate_MTL_and_STL()
	# evaluate_different_method()
	# evaluate_accuracy_composition()

'''

def get_quihui_result(search_grid_id):
	min_file = os.path.join(method_1_result_path, str(search_grid_id) + '_min.npy')
	avg_file = os.path.join(method_1_result_path, str(search_grid_id) + '_avg.npy')
	max_file = os.path.join(method_1_result_path, str(search_grid_id) + '_max.npy')

	min_array = du.load_array(min_file)[-100:]
	avg_array = du.load_array(avg_file)[-100:]
	max_array = du.load_array(max_file)[-100:]
	
	min_array = np.reshape(min_array, (100, 1, 1), order='C')
	avg_array = np.reshape(avg_array, (100, 1, 1), order='C')
	max_array = np.reshape(max_array, (100, 1, 1), order='C')

	qiuhui_array = np.concatenate((min_array, avg_array, max_array), axis=-1)
	# print(qiuhui_array.shape)
	return qiuhui_array


def get_CNN_RNN_result(search_grid_id):
	def search(search_array, search_grid_id):
		for row in range(CNN_RNN_result.shape[2]):
			for col in range(CNN_RNN_result.shape[3]):
				grid_id_ = CNN_RNN_result[0, 0, row, col, 0]
				if int(search_grid_id) == int(grid_id_):
					return CNN_RNN_result[:, :, row, col, :]

	CNN_RNN_result = du.load_array(method_CNN_RNN_result_path)

	return search(CNN_RNN_result, search_grid_id)


def plot_result(info_real, CNN_RNN, qiuhui):
	def plot_task(task_name, task_info, task_real, task_cnn_rnn, task_quihui):
		fig = plt.figure()
		plt.xlabel('time sequence')
		plt.ylabel('activity strength')

		plt.plot(task_real, label='real', marker='.')
		plt.plot(task_cnn_rnn, label='CNN_RNN', marker='.')
		plt.plot(task_quihui, label='??', marker='.')
		plt.title(task_name + ': grid_id ' + str(task_info[0]))
		plt.grid()
		plt.legend()

	plot_task('min', info_real[0, 0, (0, 1)], info_real[:, 0, 2], CNN_RNN[:, 0, 0], qiuhui[:, 0, 0])
	plot_task('avg', info_real[0, 0, (0, 1)], info_real[:, 0, 3], CNN_RNN[:, 0, 1], qiuhui[:, 0, 1])
	plot_task('max', info_real[0, 0, (0, 1)], info_real[:, 0, 4], CNN_RNN[:, 0, 2], qiuhui[:, 0, 2])

	plt.show()

for each_grid_id in plot_grid_id_list:
	CNN_RNN = get_CNN_RNN_result(each_grid_id)
	info_real = CNN_RNN[:, :, :5]
	CNN_RNN = CNN_RNN[:, :, 5:]
	qiuhui = get_quihui_result(each_grid_id)
	plot_result(info_real, CNN_RNN, qiuhui)
'''