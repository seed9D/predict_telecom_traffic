import tensorflow as tf
import tensorlayer as tl
import os
import json


class HyperParameterConfig:
	batch_size = 100
	learning_rate = 0.0005
	keep_rate = 0.7
	weight_decay = 0.01

	'''build_bi_RNN_network'''
	RNN_num_layers = 3
	RNN_num_step = 6
	RNN_hidden_node_size = 256
	RNN_cell = 'LSTMcell'
	RNN_cell_init_args = {'forget_bias': 1.0, 'use_peepholes': True, 'state_is_tuple': True}
	RNN_init_state_noise_stddev = 0.5
	RNN_initializer = 'xavier'

	'''build_CNN_network'''
	CNN_layer_activation_fn = 'relu'
	CNN_layer_1_5x5_kernel_shape = [5, 5, 1, 32]
	CNN_layer_1_5x5_kernel_strides = [1, 1, 1, 1]
	CNN_layer_1_5x5_conv_Winit = 'xavier'

	CNN_layer_1_5x5_pooling = 'max_pool'
	CNN_layer_1_5x5_pooling_ksize = [1, 2, 2, 1]
	CNN_layer_1_5x5_pooling_strides = [1, 2, 2, 1]

	CNN_layer_1_3x3_kernel_shape = [3, 3, 1, 32]
	CNN_layer_1_3x3_kernel_strides = [1, 1, 1, 1]
	CNN_layer_1_3x3_conv_Winit = 'xavier'

	CNN_layer_1_3x3_pooling = 'max_pool'
	CNN_layer_1_3x3_pooling_ksize = [1, 2, 2, 1]
	CNN_layer_1_3x3_pooling_strides = [1, 2, 2, 1]

	CNN_layer_1_pooling = 'avg_pool'
	CNN_layer_1_pooling_ksize = [1, 2, 2, 1]
	CNN_layer_1_pooling_strides = [1, 2, 2, 1]

	CNN_layer_2_kernel_shape = [5, 5, 64, 64]
	CNN_layer_2_strides = [1, 1, 1, 1]
	CNN_layer_2_conv_Winit = 'xavier'

	CNN_layer_2_pooling_ksize = [1, 2, 2, 1]
	CNN_layer_2_pooling_strides = [1, 1, 1, 1]
	CNN_layer_2_pooling = 'avg_pool'

	'''share layer Dense'''
	fully_connected_W_init = 'xavier'
	fully_connected_units = 1024

	'''prediction layer'''
	prediction_layer_1_W_init = 'xavier'
	prediction_layer_1_uints = 1024

	prediction_layer_2_W_init = 'xavier'

	def get_variable(self):
		total = vars(HyperParameterConfig)
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

if __name__ == '__main__':

	config = HyperParameterConfig()
	# con_key_val = config.get_variable()
	config.save_json()
