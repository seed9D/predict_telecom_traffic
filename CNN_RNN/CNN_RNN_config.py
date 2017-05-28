import os
import json
import numpy as np
from CNN_RNN import CNN_RNN
from collections import OrderedDict
import operator
import itertools
from functools import reduce
import shutil

class HyperParameterConfig:
	def __init__(self):
		self.iter_epoch = 2000
		self.batch_size = 100
		self.learning_rate = 0.0005
		self.keep_rate = 0.7
		self.weight_decay = 0.01

		'''build_bi_RNN_network'''
		self.RNN_num_layers = 3
		self.RNN_num_step = 6
		self.RNN_hidden_node_size = 256
		self.RNN_cell = 'LSTMcell'
		self.RNN_cell_init_args = {'forget_bias': 1.0, 'use_peepholes': True, 'state_is_tuple': True}
		self.RNN_init_state_noise_stddev = 0.5
		self.RNN_initializer = 'xavier'

		'''build_CNN_network'''
		self.CNN_layer_activation_fn = 'relu'
		self.CNN_layer_1_5x5_kernel_shape = [5, 5, 1, 32]
		self.CNN_layer_1_5x5_kernel_strides = [1, 1, 1, 1]
		self.CNN_layer_1_5x5_conv_Winit = 'xavier'

		self.CNN_layer_1_5x5_pooling = 'max_pool'
		self.CNN_layer_1_5x5_pooling_ksize = [1, 2, 2, 1]
		self.CNN_layer_1_5x5_pooling_strides = [1, 2, 2, 1]

		self.CNN_layer_1_3x3_kernel_shape = [3, 3, 1, 32]
		self.CNN_layer_1_3x3_kernel_strides = [1, 1, 1, 1]
		self.CNN_layer_1_3x3_conv_Winit = 'xavier'

		self.CNN_layer_1_3x3_pooling = 'max_pool'
		self.CNN_layer_1_3x3_pooling_ksize = [1, 2, 2, 1]
		self.CNN_layer_1_3x3_pooling_strides = [1, 2, 2, 1]

		self.CNN_layer_1_pooling = 'avg_pool'
		self.CNN_layer_1_pooling_ksize = [1, 2, 2, 1]
		self.CNN_layer_1_pooling_strides = [1, 2, 2, 1]

		self.CNN_layer_2_kernel_shape = [5, 5, 64, 64]
		self.CNN_layer_2_strides = [1, 1, 1, 1]
		self.CNN_layer_2_conv_Winit = 'xavier'

		self.CNN_layer_2_pooling_ksize = [1, 2, 2, 1]
		self.CNN_layer_2_pooling_strides = [1, 1, 1, 1]
		self.CNN_layer_2_pooling = 'avg_pool'

		'''share layer Dense'''
		self.fully_connected_W_init = 'xavier'
		self.fully_connected_units = 1024

		'''prediction layer'''
		self.prediction_layer_1_W_init = 'xavier'
		self.prediction_layer_1_uints = 1024

		self.prediction_layer_2_W_init = 'xavier'

	def get_variable(self):
		# total = vars(self)
		total = self.__dict__
		key_var = {}
		for key, value in total.items():
			if key.startswith('__') or callable(value):
				continue
			key_var[key] = value
		# print(key_var)
		return key_var

	def get_json_str(self):
		key_var = self.get_variable()
		json_string = json.dumps(key_var, sort_keys=True, indent=4)
		return json_string

	def save_json(self, file_path='./result/temp.json'):
		key_var = self.get_variable()
		if not os.path.exists(os.path.dirname(file_path)):
			os.makedirs(os.path.dirname(file_path))

		with open(file_path, 'w') as outfile:
			json.dump(key_var, outfile, sort_keys=True, indent=4)


class GridSearch():
	def __init__(self, model_path, X_array, Y_array):
		self.X_array = X_array
		self.Y_array = Y_array
		model_basename = os.path.basename(model_path)
		self.model_dirname = os.path.dirname(model_path)
		self.model_name = os.path.splitext(model_basename)

	def _find_in_obj(self, obj, condition, path=None):
		if path is None:
			path = []

		if isinstance(obj, list):
			for index, value in enumerate(obj):
				new_path = list(path)
				new_path.append(index)
				for result in self._find_in_obj(value, condition, path=new_path):
					yield result
		if isinstance(obj, dict):
			for key, value in obj.items():
				new_path = list(path)
				new_path.append(key)
				for result in self._find_in_obj(value, condition, path=new_path):
					yield result

				if condition == key:
					new_path = list(path)
					new_path.append(key)
					yield new_path

	def search_learning_rate(self):
		def find_highest_accu(obj):
			key_paths = self._find_in_obj(obj, 'testing_accurcy')  # generator

			key_paths = list(key_paths)
			key_paths.sort(key=operator.itemgetter(1))  # sort by element 2 in tuple
			uni_task_key = []
			task_groups = []
			max_pairs = []
			for k, g in itertools.groupby(key_paths, lambda x: x[1]):  # group by task
				uni_task_key.append(k)
				task_groups.append(list(g))
				# value = reduce(operator.getitem, key_pair, obj)
			for each_task_group in task_groups:
				task_iter = iter(each_task_group)
				key_value = [(ele, reduce(operator.getitem, ele, obj)) for ele in task_iter]
				max_pair = max(key_value, key=lambda x: x[1])

				'''
				for ele in task_iter:
					value = reduce(operator.getitem, ele, obj)
					print('rate:{} task:{} {}:{}'.format(
						ele[0],
						ele[1],
						ele[2],
						value))
				'''
				max_pairs.append(max_pair)
			return max_pairs

		hyper_config = HyperParameterConfig()
		rate_summerize = OrderedDict()
		# rate_list = list(range(0.0001, 0.05, 0.0005))
		rate_array = np.arange(0.00001, 0.001, 0.00005)
		rate_array = rate_array.tolist()

		basic_result_path = '/home/mldp/ML_with_bigdata/CNN_RNN/result/search_learning_rate/'
		shutil.rmtree(basic_result_path)
		if not os.path.exists(basic_result_path):
			os.makedirs(basic_result_path)

		for rate in rate_array:
			print('rate:{}'.format(rate))
			hyper_config.learning_rate = round(float(rate), 6)
			save_model_path = self.model_dirname + '/' + self.model_name[0] + '_rate_' + str(hyper_config.learning_rate) + '.ckpt'
			result_path = basic_result_path + 'rate_' + str(hyper_config.learning_rate) + '/'

			if not os.path.exists(result_path):
				os.makedirs(result_path)

			model_path = {
				'reload_path': '/home/mldp/ML_with_bigdata/CNN_RNN/output_model/CNN_RNN_test.ckpt',
				'save_path': save_model_path,
				'result_path': result_path
			}
			summerize = self._run_CNN_RNN(model_path, hyper_config)
			rate_summerize[str(rate)] = summerize

		print('search finish!!')

		max_pairs = find_highest_accu(rate_summerize)
		for max_pair in max_pairs:
			print('rate:{} in task: {} {}:{}'.format(
				max_pair[0][0],
				max_pair[0][1],
				max_pair[0][2],
				max_pair[1]))

		with open(basic_result_path + 'search_rate_report.txt', 'w') as outfile:
			json_string = json.dumps(rate_summerize, sort_keys=True, indent=4)
			# print(json_string)
			outfile.write(json_string + '\n')
			for max_pair in max_pairs:
				outfile.write('rate:{} in task: {} {}:{} \n'.format(
					max_pair[0][0],
					max_pair[0][1],
					max_pair[0][2],
					max_pair[1]))

		'''
		with open(basic_result_path + 'search_rate_report.txt', 'w') as outfile:
			for rate_v, rate_v_element in rate_summerize.items():
				print(rate_v, ': ')
				outfile.write()
				for task, result in rate_v_element.items():
					print(task, ':')
					print('\t', result)
		'''
	

	def _run_CNN_RNN(self, model_path, config):
		input_data_shape = [self.X_array.shape[1], self.X_array.shape[2], self.X_array.shape[3], self.X_array.shape[4]]
		output_data_shape = [self.Y_array.shape[1], self.Y_array.shape[2], self.Y_array.shape[3], 1]
		cnn_rnn = CNN_RNN(input_data_shape, output_data_shape, config)
		cnn_rnn.create_MTL_task(self.X_array, self.Y_array[:, :, :, :, 0, np.newaxis], 'min_traffic')
		cnn_rnn.create_MTL_task(self.X_array, self.Y_array[:, :, :, :, 1, np.newaxis], 'avg_traffic')
		cnn_rnn.create_MTL_task(self.X_array, self.Y_array[:, :, :, :, 2, np.newaxis], 'max_traffic')
		return cnn_rnn.start_MTL_train(model_path, reload=False)


if __name__ == '__main__':

	config = HyperParameterConfig()
	# con_key_val = config.get_variable()
	config.save_json()
