import numpy as np
import os
import tensorflow as tf
import functools as fn
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from glob import glob
from math import sqrt


class CNN_autoencoder:
	def __init__(self, *data_shape, **network_para):
		self.network_para = network_para
		self.learning_rate = 0.0005
		self.training_iters = 1000
		self.batch_size = 100
		self.display_step = 25
		self.dropout = 0.7
		self.shuffle_capacity = 800
		self.shuffle_min_after_dequeue = 300
		self.weight_decay = 10
		self.Kl_beta = 1e-3
		self.kl_sparsity_parameter = 0.4

		# self.n_input = 100*100

		self.input_temporal = data_shape[0]
		self.input_vertical = data_shape[1]
		self.input_horizontal = data_shape[2]
		self.input_channel = data_shape[3]

		self.output_temporal = 1
		self.output_vertical = data_shape[1]
		self.output_horizontal = data_shape[2]
		self.output_channel = data_shape[3]

		# placeholder
		self.Xs = tf.placeholder(tf.float32, shape=[
			None, self.input_temporal, self.input_vertical, self.input_horizontal, self.input_channel], name='pre_input_x')
		self.Ys = tf.placeholder(tf.float32, shape=[
			None, self.output_temporal, self.output_vertical, self.output_horizontal, self.output_channel], name='pre_input_y')
		self.keep_prob = tf.placeholder(tf.float32, name='pre_keep_prob')
		self.norm = tf.placeholder(tf.bool, name='pre_norm')
		# operation
		self._set_pre_train_para(**network_para)
		self._set_predictior_para(**network_para)

	def _set_predictior_para(self, **network_para):
		# tf.reset_default_graph()
		# pre_train_saver = tf.train.import_meta_graph('/home/mldp/ML_with_bigdata/output_model/AE_pre_32_32_test.ckpt_part.meta')
		# pre_train_graph = tf.get_default_graph()
		'''
		self.pre_traion_output = pre_train_graph.get_tensor_by_name('pre_train/pre_train_output:0')
		self.pre_train_abs = pre_train_graph.get_tensor_by_name('pre_train/pre_traion_abs:0')
		# placeholder
		self.Xs = pre_train_graph.get_tensor_by_name('pre_train/pre_input_x:0')
		self.Ys = pre_train_graph.get_tensor_by_name('pre_train/pre_input_y:0')
		self.keep_prob = pre_train_graph.get_tensor_by_name('pre_train/pre_keep_prob:0')
		self.norm = pre_train_graph.get_tensor_by_name('pre_train/pre_norm:0')
		'''
		with tf.variable_scope("predictior"):
			self.fc1 = network_para.get('fc1')
			# fc1_input = self.input_temporal * self.input_vertical * self.input_horizontal * self.input_channel
			fc1_input = self.input_temporal * self.input_vertical * self.input_horizontal * self.input_channel * network_para.get('conv3')
			fc2_output = self.output_temporal * self.output_vertical * self.output_horizontal * self.output_channel
			with tf.device('/cpu:0'):
				self.predictor_weights = {
					'fc1': self.weight_variable([fc1_input, self.fc1], 'fc1_w'),
					'fc2': self.weight_variable([self.fc1, fc2_output], 'fc2_w')
				}
				self.predictor_bias = {
					'fc1': self.bias_variable([self.fc1], 'fc1_b'),
					'fc2': self.bias_variable([fc2_output], 'fc2_b')
				}
			'''
			with tf.variable_scope("pre_train") as scope:
				self._set_pre_train_para(**self.network_para)
				scope.reuse_variables()
				self.pre_encoder_OP, self.pre_endecoder_OP = self._pre_train_net(
					self.Xs,
					self.pre_weights,
					self.pre_bias,
					self.keep_prob,
					self.norm)
			'''
			self.train_predictor = self._predictor_net(self.pre_train_output, self.predictor_weights, self.predictor_bias, self.keep_prob, self.norm)
			MSE_loss = self.MSE_loss(self.Ys, self.train_predictor)
			RMSE_loss = self._RMSE_loss(self.Ys, self.train_predictor)
			MAE_loss = self._absolute_error(self.Ys, self.train_predictor)
			L2_norm = self._L2_norm(self.predictor_weights)
			L2_norm += self._L2_norm(self.pre_weights)
			train_cost = RMSE_loss + L2_norm * self.weight_decay

			self.predictor_loss = {
				'MSE': MSE_loss,
				'RMSE': RMSE_loss,
				'MAE': MAE_loss,
				'L2': L2_norm,
				'cost': train_cost
			}
			# opt_vars = [v for v in tf.trainable_variables() if v.name.startswith("train")]
			# self.predictor_optimization = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.predictor_cost_OP, var_list=opt_vars)
			self.predictor_optimization = tf.train.AdamOptimizer(
				learning_rate=self.learning_rate).minimize(train_cost)
		self.saver = tf.train.Saver()

	def _predictor_net(self, x, weights, bias, dropout, norm=0):
		input_shape = x.get_shape()
		print('x:', input_shape)
		train_net_input = tf.reshape(x, [-1, weights['fc1'].get_shape().as_list()[0]])
		print('train_net_input:', train_net_input.get_shape())
		fc1 = self.batch_normalize(train_net_input, 'fc1', norm)
		# fc1 = tf.nn.relu(fc1)
		fc1 = tf.add(tf.matmul(fc1, weights['fc1']), bias['fc1'])
		fc1 = tf.nn.relu(fc1)
		fc1 = tf.nn.dropout(fc1, dropout)
		print('fc1:', fc1.get_shape())

		fc2 = self.batch_normalize(fc1, 'fc2', norm)
		fc2 = tf.add(tf.matmul(fc2, weights['fc2']), bias['fc2'])
		fc2 = tf.nn.dropout(fc2, dropout)
		fc2 = tf.nn.relu(fc2)
		print('fc2: {}'.format(fc2.get_shape()))
		train_output = tf.reshape(fc2, [-1, self.output_temporal, self.output_vertical, self.output_horizontal, self.output_channel])
		print('train_output {}'.format(train_output.get_shape()))
		return train_output

	def _set_pre_train_para(self, **network_para):

		def train_layer(input_array, weight, bias, var_name, dropout, norm=0, loss_func='RMSE', weight_decay=self.weight_decay):
			def build_network(input_array, weight, bias, var_name, dropout, norm=0):
				print('input shape :{}'.format(input_array.get_shape()))
				strides_size = {'temporal': 1, 'vertical': 1, 'horizontal': 1}
				deconv_w = self.weight_variable([3, 5, 5, weight.get_shape().as_list()[3], weight.get_shape().as_list()[4]], 'temp')
				deconv_b = self.bias_variable([input_array.get_shape().as_list()[-1]], 'temp')
				# encoder
				conv = self.batch_normalize(input_array, var_name, norm)
				conv = self.conv3d(conv, weight, bias, strides_size)
				conv = tf.nn.relu(conv)
				conv = tf.nn.dropout(conv, dropout)

				# decoder
				print('conv shape :{}'.format(conv.get_shape()))
				output_shape_of_deconv = tf.pack([
					tf.shape(input_array)[0],
					self.input_temporal,
					self.input_vertical,
					self.input_horizontal,
					input_array.get_shape().as_list()[-1]])
				deconv = self.deconv3d(conv, deconv_w, deconv_b, output_shape_of_deconv, strides_size)
				# deconv = tf.nn.relu(deconv)
				deconv = tf.nn.dropout(deconv, dropout)

				return conv, deconv
			conv, deconv = build_network(input_array, weight, bias, var_name, dropout, norm)

			# layer operation
			kl_divergence, temp_term = self._sum_Kl_div(conv)
			pre_MSE = self.MSE_loss(deconv, input_array)
			pre_MAE = self._absolute_error(input_array, deconv)
			pre_RMSE = self._RMSE_loss(input_array, deconv)
			pre_L2 = self._L2_norm({var_name: weight})

			if loss_func == 'MAE':
				pre_cost_OP = pre_MAE + pre_L2 * weight_decay  # + kl_divergence * self.Kl_beta
			elif loss_func == 'MSE':
				pre_cost_OP = pre_MSE + pre_L2 * weight_decay  # + kl_divergence * self.Kl_beta
			else:
				pre_cost_OP = pre_RMSE + pre_L2 * weight_decay  # + kl_divergence * self.Kl_beta
			opt_vars = [v for v in tf.trainable_variables() if v.name.startswith("pre_train/" + var_name)]
			for v in tf.trainable_variables():
				print(v)
			print()
			'''
			pre_optimizer_OP = tf.train.AdamOptimizer(
				learning_rate=self.learning_rate).minimize(pre_cost_OP, var_list=opt_vars)
			'''
			optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
			gvs = optimizer.compute_gradients(pre_cost_OP, var_list=opt_vars)

			for grad, var in gvs:
				if grad is not None:
					print(grad, var)
			capped_gvs = [(tf.clip_by_value(grad, -15., 15.), var) for grad, var in gvs if grad is not None]
			pre_optimizer_OP = optimizer.apply_gradients(capped_gvs)
			loss = {
				'cost': pre_cost_OP,
				'L2': pre_L2,
				'RMSE': pre_RMSE,
				'MAE': pre_MAE,
				'MSE': pre_MSE,
				'kl_diver': kl_divergence,
				'temp': conv
			}
			return pre_optimizer_OP, conv, deconv, loss

		with tf.variable_scope("pre_train"):
			# network parameter
			self.pooling_times = 0
			self.conv1 = network_para.get('conv1')
			self.conv2 = network_para.get('conv2')
			self.conv3 = network_para.get('conv3')
			with tf.device('/cpu:0'):

				# variable, control filter size
				self.pre_weights = {
					'conv1': self.weight_variable([3, 5, 5, self.input_channel, self.conv1], 'conv1_w'),
					'conv2': self.weight_variable([3, 5, 5, self.conv1, self.conv2], 'conv2_w'),
					'conv3': self.weight_variable([3, 5, 5, self.conv2, self.conv3], 'conv3_w'),
					# 'deconv1': self.weight_variable([3, 5, 5, self.conv2, self.conv3], 'deconv1_w'),
					# 'deconv2': self.weight_variable([3, 5, 5, self.conv1, self.conv2], 'deconv2_w'),
					# 'deconv3': self.weight_variable([3, 5, 5, self.input_channel, self.conv1], 'deconv3_w')
				}
				self.pre_bias = {
					'conv1': self.bias_variable([self.conv1], 'conv1_b'),
					'conv2': self.bias_variable([self.conv2], 'conv2_b'),
					'conv3': self.bias_variable([self.conv3], 'conv3_b'),
					# 'deconv1': self.bias_variable([self.conv2], 'deconv1_b'),
					# 'deconv2': self.bias_variable([self.conv1], 'deconv2_b'),
					# 'deconv3': self.bias_variable([self.input_channel], 'deconv3_b')
				}

				# self.mean = tf.Variable(tf.zeros([self.input_channel]),name='mean',trainable=False,)
				# self.std = tf.Variable(tf.zeros([self.input_channel]),name='std',trainable=False,)
			pre_optimizer_OP_layer_1, en_conv_1, self.de_conv_1, self.pre_loss_1 = train_layer(
				self.Xs,
				self.pre_weights['conv1'],
				self.pre_bias['conv1'],
				'conv1',
				self.keep_prob,
				self.norm,
				'MAE',
				weight_decay=100)
			pre_optimizer_OP_layer_2, en_conv_2, de_conv_2, pre_loss_2 = train_layer(
				en_conv_1,
				self.pre_weights['conv2'],
				self.pre_bias['conv2'],
				'conv2',
				self.keep_prob,
				self.norm,
				'RMSE',
				weight_decay=100)
			pre_optimizer_OP_layer_3, en_conv_3, de_conv_3, pre_loss_3 = train_layer(
				en_conv_2,
				self.pre_weights['conv3'],
				self.pre_bias['conv3'],
				'conv3',
				self.keep_prob,
				self.norm,
				'RMSE')
			self.pre_optimizer_list = [pre_optimizer_OP_layer_1, pre_optimizer_OP_layer_2, pre_optimizer_OP_layer_3]
			self.pre_loss = [self.pre_loss_1, pre_loss_2, pre_loss_3]
			self.pre_train_output = en_conv_3
			'''
			# operation
			self.pre_encoder_OP, self.pre_endecoder_OP, kl_divergence_OP = self._pre_train_net(
				self.Xs,
				self.pre_weights,
				self.pre_bias,
				self.keep_prob,
				self.norm)
			self.pre_encoder_OP = tf.identity(self.pre_encoder_OP, name='pre_train_encoder')
			self.pre_endecoder_OP = tf.identity(self.pre_endecoder_OP, name='pre_train_output')

			self.pre_MSE = self.MSE_loss(self.pre_endecoder_OP, self.Ys)

			absolute_distance = self._absolute_error(self.Ys, self.pre_endecoder_OP)
			self.pre_absolute_distance = tf.identity(absolute_distance, name='pre_traion_abs')

			self.pre_RMSE = self._RMSE_loss(self.Ys, self.pre_endecoder_OP)
			self.pre_L2 = self._L2_norm(self.pre_weights)

			self.pre_cost_OP = self.pre_MSE + self.weight_decay * self.pre_L2 + self.Kl_beta * kl_divergence_OP
			self.pre_optimizer_OP = tf.train.AdamOptimizer(
				learning_rate=self.learning_rate).minimize(self.pre_cost_OP)
			self.pre_weight_bias = tf.train.Saver({
				'conv1_w': self.pre_weights['conv2'],
				'conv3_w': self.pre_weights['conv3'],
				'deconv1_w': self.pre_weights['deconv1'],
				'deconv2_w': self.pre_weights['deconv2'],
				'deconv3_w': self.pre_weights['deconv3'],
				'conv1_b': self.pre_bias['conv1'],
				'conv2_b': self.pre_bias['conv2'],
				'conv3_b': self.pre_bias['conv3'],
				'deconv1_b': self.pre_bias['deconv1'],
				'deconv2_b': self.pre_bias['deconv2'],
				'deconv3_b': self.pre_bias['deconv3']})
			# self.init_OP = tf.global_variables_initializer()
			self.pre_traion_output = self.pre_endecoder_OP
			self.pre_train_saver = tf.train.Saver([v for v in tf.all_variables() if 'pre_train' in v.name])
			'''
			self.saver = tf.train.Saver()

	def _pre_train_net(self, x, weights, bias, dropout, norm=0):
			kl_list = []
			k_size = {'temporal': 1, 'vertical': 2, 'horizontal': 2}
			strides_size = {'temporal': 1, 'vertical': 1, 'horizontal': 1}
			# layer 1
			print('x shape :{}'.format(x.get_shape()))
			conv1 = self.conv3d(x, weights['conv1'], bias['conv1'], strides_size)
			conv1 = tf.nn.relu(conv1)
			# conv1 = maxpool3d(conv1, k_size, strides_size)
			conv1 = tf.nn.dropout(conv1, dropout)
			kl_list.append(conv1)

			# layer 2
			print('conv1 shape :{}'.format(conv1.get_shape()))
			conv2 = self.batch_normalize(conv1, 'conv2', norm)
			conv2 = self.conv3d(conv2, weights['conv2'], bias['conv2'], strides_size)
			conv2 = tf.nn.relu(conv2)
			# conv2 = maxpool3d(conv2, k_size, strides_size)
			conv2 = tf.nn.dropout(conv2, dropout)
			kl_list.append(conv2)
			# encode_output = conv2
			# layer 3
			print('conv2 shape :{}'.format(conv2.get_shape()))
			conv3 = self.batch_normalize(conv2, 'conv3', norm)
			conv3 = self.conv3d(conv3, weights['conv3'], bias['conv3'], strides_size)
			conv3 = tf.nn.relu(conv3)
			# conv3 = maxpool3d(conv3, k_size, strides_size)
			conv3 = tf.nn.dropout(conv3, dropout)
			kl_list.append(conv3)
			encode_output = conv3
			print('encode layer shape:%s' % encode_output.get_shape())

			# layer 4
			deconv1 = self.batch_normalize(encode_output, 'deconv1', norm)
			# deconv1 = maxunpool3d(deconv1, k_size)
			output_shape_of_dconv1 = tf.pack([
				tf.shape(x)[0],
				self.input_temporal // (2 ** self.pooling_times),
				self.input_vertical // (2 ** self.pooling_times),
				self.input_horizontal // (2 ** self.pooling_times),
				self.conv2])
			deconv1 = self.deconv3d(deconv1, weights['deconv1'], bias['deconv1'], output_shape_of_dconv1, strides_size)
			deconv1 = tf.nn.relu(deconv1)
			deconv1 = tf.nn.dropout(deconv1, dropout)

			# output_shape_of_dconv1_unpool = tf.pack([tf.shape(x)[0],
			# self.input_temporal/(2**(self.pooling_times-1)),
			# self.input_vertical/(2**(self.pooling_times-1)),
			# self.input_horizontal/(2**(self.pooling_times-1)),
			# self.conv2])
			# deconv1 = self.maxpool3d(deconv1,output_shape_of_dconv1_unpool)
			# layer 5
			output_shape_of_dconv2 = tf.pack([
				tf.shape(x)[0],
				self.input_temporal // (2 ** self.pooling_times),
				self.input_vertical // (2 ** self.pooling_times),
				self.input_horizontal // (2 ** self.pooling_times),
				self.conv1
			])
			deconv2 = self.batch_normalize(deconv1, 'deconv2', norm)
			# deconv2 = maxunpool3d(deconv2, k_size)
			deconv2 = self.deconv3d(
				deconv2,
				weights['deconv2'],
				bias['deconv2'],
				output_shape_of_dconv2,
				strides_size)
			deconv2 = tf.nn.relu(deconv2)
			deconv2 = tf.nn.dropout(deconv2, dropout)
			# layer 6
			deconv3 = self.batch_normalize(deconv2, 'deconv3', norm)
			# deconv3 = maxunpool3d(deconv3, k_size)
			output_shape_of_dconv3 = tf.pack([
				tf.shape(x)[0],
				self.input_temporal // (2 ** self.pooling_times),
				self.input_vertical // (2 ** self.pooling_times),
				self.input_horizontal // (2 ** self.pooling_times),
				self.input_channel
			])
			deconv3 = self.deconv3d(
				deconv3,
				weights['deconv3'],
				bias['deconv3'],
				output_shape_of_dconv3,
				strides_size
			)
			# deconv3 = tf.nn.dropout(deconv3, dropout)
			# deconv3 = tf.nn.tanh(deconv3)
			endecoder_output = deconv3
			print('endecoder_output output shape :%s' % endecoder_output.get_shape())
			kl_list = tf.pack(kl_list)
			sum_of_kl = self._sum_Kl_div(kl_list)
			print('sum_of_kl {}'.format(sum_of_kl))
			return encode_output, endecoder_output, sum_of_kl

	def set_training_data(self, input_x, input_y):

		print('input_x shape:{}'.format(input_x.shape))
		print('input_y shape:{}'.format(input_y.shape))
		self.output_temporal = input_y.shape[1]
		self.output_vertical = input_y.shape[2]
		self.output_horizontal = input_y.shape[3]
		self.output_channel = input_y.shape[4]
		# input_x, self.mean, self.std = self.feature_normalize_input_data(input_x)
		self.mean = 0
		self.std = 1
		X_data = input_x[0:-1]
		Y_data = input_y[1:]

		# Y_data = Y_data[:,np.newaxis]
		# print(X_data[1,0,0,0,-1],Y_data[0,0,0,0,-1])
		training_X = X_data[0:int(9 * X_data.shape[0] / 10)]
		training_Y = Y_data[0:int(9 * Y_data.shape[0] / 10)]
		self.testing_X = X_data[int(9 * X_data.shape[0] / 10):]
		self.testing_Y = Y_data[int(9 * Y_data.shape[0] / 10):]
		self.training_file = 'training.tfrecoeds'
		self.testing_file = 'testing.tfrecoeds'
		print('training X shape:{}, training Y shape:{}'.format(
			training_X.shape, training_Y.shape))
		self._write_to_Tfrecord(training_X, training_Y, self.training_file)
		self._write_to_Tfrecord(self.testing_X, self.testing_Y, self.testing_file)
		self.training_data_number = training_X.shape[0]

	def MSE_loss(self, real, predict):
			with tf.variable_scope('MSE_loss'):

				# print('endecoder_OP shape:{}, Ys shape{}'.format(self.endecoder_OP[:,5].get_shape(),self.Ys.get_shape()))
				# loss = tf.reduce_mean(tf.pow(predict - real, 2))
				# loss = tf.div(loss, 2)
				'''
				L2 = 0
				for key, value in weights.items():
					L2 += tf.nn.l2_loss(value)
				loss += tf.reduce_mean(L2 * self.weight_decay)
				'''
			return tf.reduce_mean(tf.pow(predict - real, 2))

	def _L2_norm(self, weights):

		L2 = tf.zeros(1, tf.float32)
		for value in weights.values():
			L2 = tf.add(L2, tf.nn.l2_loss(value))
		# L2 = tf.nn.l2_loss(weights)
		return tf.reduce_mean(L2)

	def _absolute_error(self, real, predict):
		return tf.reduce_mean(tf.abs(predict - real))

	def _RMSE_loss(self, real, predict):
		return tf.sqrt(tf.reduce_mean(tf.pow(predict - real, 2)))

	def _sum_Kl_div(self, layer_output):
		def kl_log(x):
			x = tf.clip_by_value(x, 1e-30, 100)  # avoid negative value
			x = tf.log(x + 1e-30)
			x = tf.clip_by_value(x, 1e-30, 1000)  # avlid inf value
			return x

		def kl_div(rho_head, rho):
			invers_rho_head = tf.sub(tf.constant(1.), rho_head)
			invers_rho = tf.sub(tf.constant(1.), rho)
			first_term = tf.mul(rho, kl_log(tf.div(rho, rho_head)))
			second_term = tf.mul(invers_rho, kl_log(tf.div(invers_rho, invers_rho_head)))
			kl_diver = tf.add(first_term, second_term)  # each hidden unit j, it's an array
			kl_div_mean = tf.reduce_mean(kl_diver)  # a value
			return kl_div_mean, rho_head
		# following is for sparse auto encoder operation
		# each layer's average output
		# calculate each hidden unit's average output from total training data in each layer
		'''
		average_output_elayer = []
		for conv_layer in KL_list:
			average_output_elayer.append(tf.reduce_mean(conv_layer, axis=0))
		'''
		layer_output_shape = layer_output.get_shape().as_list()
		dim_prod = fn.reduce(lambda x, y: x * y, layer_output_shape[1:])
		layer_output = tf.reshape(layer_output, [-1, dim_prod])
		average_output = tf.reduce_mean(layer_output, axis=0)
		# each layer's sum of each hiiden unit KL divergence
		'''
		kl_encode_layer = []
		for average_output in average_output_elayer:
			kl_encode_layer.append(kl_div(average_output, self.kl_sparsity_parameter))
		sum_of_kl = sum(kl for kl in kl_encode_layer)
		'''
		mean_of_kl, first_term = kl_div(average_output, self.kl_sparsity_parameter)
		return mean_of_kl, first_term

	def weight_variable(self, shape, name):
		# initial = tf.truncated_normal(shape, stddev=0.1)
		initial = np.random.randn(*shape) * sqrt(2.0 / np.prod(shape))
		return tf.Variable(initial, dtype=tf.float32, name=name)

	def bias_variable(self, shape, name):
		# initial = tf.random_normal(shape)
		initial = np.random.randn(*shape) * sqrt(2.0 / np.prod(shape))
		return tf.Variable(initial, dtype=tf.float32, name=name)

	def conv3d(self, x, W, b, strides_size):
		x = tf.nn.conv3d(
			x, W, strides=[
				1,
				strides_size['temporal'],
				strides_size['vertical'],
				strides_size['horizontal'],
				1], padding='SAME')
		x = tf.nn.bias_add(x, b)
		return x

	def deconv3d(self, x, W, b, output_shape, strides_size):
		'''
			filter shape:[depth, height, width, output_channels, in_channels]
		'''
		print('input shape:{} filter shape:{} output_shape:{}'.format(
			x.get_shape(), W.get_shape(), output_shape.get_shape()))
		x = tf.nn.conv3d_transpose(x, W, output_shape, strides=[
			1,
			strides_size['temporal'],
			strides_size['vertical'],
			strides_size['horizontal'],
			1], padding='SAME')
		x = tf.nn.bias_add(x, b)
		return x

	def maxpool3d(self, x, k_size, strides_size):

		x = tf.nn.max_pool3d(x, ksize=[1, k_size['temporal'], k_size['vertical'], k_size['horizontal'], 1], strides=[
			1,
			strides_size['temporal'],
			strides_size['vertical'],
			strides_size['horizontal'], 1], padding='SAME')
		self.pooling_times += 1
		return x

	def maxunpool3d(self, x, k_size):
		input_shape = tf.shape(x, out_type=tf.int32)
		output_shape = (
			input_shape[0],
			input_shape[1] * k_size['temporal'],
			input_shape[2] * k_size['vertical'],
			input_shape[3] * k_size['horizontal'],
			input_shape[4])
		'''
		dim = len(input_shape[1:-2])
		out = (tf.reshape(x, ))
		for i in range(dim, 0, -1):
			out = tf.concat([out, tf.zeros_like(out, dtype=tf.int32)], i)
		x = tf.reshape(out, output_shape)
		'''
		print('x shape {}'.format(x.get_shape()))
		out = tf.concat(4, [x, tf.zeros_like(x)])
		out = tf.concat(3, [out, tf.zeros_like(out)])
		out = tf.concat(2, [out, tf.zeros_like(out)])
		x = tf.reshape(out, output_shape)
		self.pooling_times -= 1
		return x

	def batch_normalize(self, x, scope, norm=0):
		axes = list(range(len(x.get_shape()) - 1))
		decay = 0.999
		with tf.variable_scope(scope):
			pop_mean = tf.get_variable('pop_mean', [x.get_shape(
			)[-1]], trainable=False, dtype=tf.float32, initializer=tf.constant_initializer(0))
			pop_var = tf.get_variable('pop_var', [x.get_shape(
			)[-1]], trainable=False, dtype=tf.float32, initializer=tf.constant_initializer(1))
			# scale = tf.get_variable(tf.ones([x.get_shape()[-1]]),name='scale')
			scale = tf.get_variable('scale', [x.get_shape(
			)[-1]], dtype=tf.float32, initializer=tf.constant_initializer(1))
			# beta = tf.get_variable(tf.zeros([x.get_shape()[-1]]),name='beta')
			beta = tf.get_variable('beta', [x.get_shape(
			)[-1]], dtype=tf.float32, initializer=tf.constant_initializer(0))
		if norm is 1:
			batch_mean, batch_var = tf.nn.moments(x, axes=axes)
			train_mean = tf.assign(
				pop_mean, pop_mean * decay + batch_mean * (1 - decay))
			train_var = tf.assign(
				pop_var, pop_var * decay + batch_var * (1 - decay))
			with tf.control_dependencies([train_mean, train_var]):
				return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, scale, 1e-3)
		else:
			return tf.nn.batch_normalization(x, pop_mean, pop_var, beta, scale, 1e-3)

	def feature_normalize_input_data(self, input_x):
		input_shape = input_x.shape

		axis = tuple(range(len(input_shape) - 1))
		mean = np.mean(input_x, axis=axis)
		std = np.std(input_x, axis=axis)
		print('mean:{},std:{}'.format(mean, std))
		X_normalize = (input_x - mean) / std
		return X_normalize, mean, std

	def un_normalize_data(self, sess, input_x):
		# mean,std = sess.run([self.mean,self.std])
		# print('mean:{},std:{}'.format(self.mean,self.std))
		input_x = input_x * self.std + self.mean
		return input_x

	def reload_tfrecord(self, tfrecored_file, tftesting_file):
		if not os.path.isfile(tfrecored_file):
			print('{} not exists'.format(tfrecored_file))
		else:
			self.training_file = tfrecored_file
			self.testing_file = tftesting_file

	def predict_data(self, input_x, model_path, stage):
		input_x = (input_x - self.mean) / self.std
		print('input_x shape:', input_x.shape)
		testing_x = input_x[0:-1]
		testing_y = input_x[1:]
		with tf.Session() as sess:
			loss = 0
			predict_y = None
			if stage == 'pre_train':
				self._reload_model(sess, model_path['pretrain_reload'])
				# input_x_length = input_x.shape[0]
				loss, predict_y = self._testing_data(sess, testing_x, testing_x, stage)  # testing x itself
			elif stage == 'train':
				self._reload_model(sess, model_path['reload'])
				# input_x_length = input_x.shape[0]
				loss, predict_y = self._testing_data(sess, testing_x, testing_y, stage)  # testing x itself
			print('predict finished!')
			print('loss:{} predict_y shape {}'.format(loss, predict_y.shape))

			# testing_y = self.un_normalize_data(sess,testing_y)
			return input_x[0:-1], predict_y
	'''
	def set_pre_model_name(self, reload_model_path, save_model_path):

		if not glob(reload_model_path + '.*'):
			print('{} not exists'.format(reload_model_path))
			# exit(1)
		else:
			self.model_path = reload_model_path

		output_dir = os.path.dirname(save_model_path)
		if not os.path.isdir(output_dir):
			os.makedirs(output_dir)
		self.save_model = save_model_path
	'''
	def _reload_model(self, sess, model_path):
		print('reloading model {}.....'.format(model_path))

		self.saver.restore(sess, model_path)

	def _save_model(self, sess, model_path):
		print('saving model.....')
		if not os.path.isdir('./output_model'):
			os.makedirs('./output_model')
		try:
			save_path = self.saver.save(sess, model_path)
			# self.pre_train_saver.save(sess, model_path + '_part')
		except Exception:
			save_path = self.saver.save(sess, './output_model/temp.ckpt')
		finally:
			print('save_path{}'.format(save_path))

	def _write_to_Tfrecord(self, X_array, Y_array, filename):
		writer = tf.python_io.TFRecordWriter(filename)
		for index, each_record in enumerate(X_array):
			tensor_record = each_record.astype(np.float32).tobytes()
			tensor_result = Y_array[index].astype(np.float32).tobytes()
			# print('in _write_to_Tfrecord',X_array.shape,Y_array.shape)
			example = tf.train.Example(features=tf.train.Features(feature={
				'index': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
				'record': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tensor_record])),
				'result': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tensor_result]))
			}))

			writer.write(example.SerializeToString())
		writer.close()

	def _read_data_from_Tfrecord(self, filename):
		filename_queue = tf.train.string_input_producer([filename])
		reader = tf.TFRecordReader()
		_, serialized_example = reader.read(filename_queue)
		features = tf.parse_single_example(
			serialized_example,
			features={
				'index': tf.FixedLenFeature([], tf.int64),
				'record': tf.FixedLenFeature([], tf.string),
				'result': tf.FixedLenFeature([], tf.string)
			})
		index = features['index']
		record = tf.decode_raw(features['record'], tf.float32)
		result = tf.decode_raw(features['result'], tf.float32)
		record = tf.reshape(record, [
			self.input_temporal,
			self.input_vertical,
			self.input_horizontal,
			self.input_channel])
		result = tf.reshape(result, [
			self.output_temporal,
			self.output_vertical,
			self.output_horizontal,
			self.output_channel])

		return index, record, result

	def _read_all_data_from_Tfreoced(self, filename):
		record_iterator = tf.python_io.tf_record_iterator(path=filename)
		record_list = []
		result_list = []
		for string_record in record_iterator:
			example = tf.train.Example()
			example.ParseFromString(string_record)
			index = example.features.feature['index'].int64_list.value[0]
			record = example.features.feature['record'].bytes_list.value[0]
			result = example.features.feature['result'].bytes_list.value[0]
			record = np.fromstring(record, dtype=np.float32)
			record = record.reshape((
				self.input_temporal,
				self.input_vertical,
				self.input_horizontal,
				self.input_channel))

			result = np.fromstring(result, dtype=np.float32)
			result = result.reshape((
				self.output_temporal,
				self.output_vertical,
				self.output_horizontal,
				self.output_channel))
			record_list.append(record)
			result_list.append(result)

		record = np.stack(record_list)
		result = np.stack(result_list)
		return index, record, result

	def _testing_data(self, sess, input_x, input_y, stage):
		batch_num = int(input_x.shape[0] / self.batch_size)
		testing_data_number = input_y.shape[0]
		if testing_data_number % self.batch_size is not 0:
			batch_len = batch_num + 1
		else:
			batch_len = batch_num

		with tf.device('/gpu:0'):
			predict_list = []
			cum_loss = 0
			for batch_index in range(batch_len):
				predict = 0
				loss = 0
				if stage == 'pre_train':
					loss, predict = sess.run([self.pre_loss_1['cost'], self.de_conv_1], feed_dict={
						self.Xs: input_x[batch_index * self.batch_size:(batch_index + 1) * self.batch_size],
						# self.Ys: input_y[batch_index * self.batch_size:(batch_index + 1) * self.batch_size],
						self.keep_prob: 1,
						self.norm: 0
					})
				elif stage == 'train':
					loss, predict = sess.run([self.predictor_loss['cost'], self.train_predictor], feed_dict={
						self.Xs: input_x[batch_index * self.batch_size:(batch_index + 1) * self.batch_size],
						self.Ys: input_y[batch_index * self.batch_size:(batch_index + 1) * self.batch_size],
						self.keep_prob: 1,
						self.norm: 0
					})
				predict = self.un_normalize_data(sess, predict)
				un_normalize_input_y = self.un_normalize_data(
					sess, input_y[batch_index * self.batch_size:(batch_index + 1) * self.batch_size])

				for i in range(3):
					for j in range(predict.shape[1]):
						print('predict:{},real:{}'.format(
							predict[i, j, 10, 20, 0], un_normalize_input_y[i, j, 10, 20, 0]))

				for predict_element in predict:
					predict_list.append(predict_element)
				cum_loss += loss

			return cum_loss / batch_num, np.stack(predict_list)

	def _save_history(self, input_data):

		with open('history.txt', 'w') as f:
			f.write('{}\t{}\t{}\t{}\t{}\n'.format(
				'epoch', 'training_lostt', 'testing_loss', 'absolute_distance', 'RMSE'))
			for i, epoch in enumerate(input_data['epoch']):
				f.write('{}\t{}\t{}\t{}\t{}\n'.format(
					input_data['epoch'][i],
					input_data['training_loss_his'][i],
					input_data['testting_loss_hist'][i],
					input_data['absolute_distance'][i],
					input_data['RMSE'][i]))

	def start_pre_training(self, model_path, restore=False):
		'''
		with tf.variable_scope("pre_train"):
			self._set_pre_train_para(**self.network_para)
		'''
		fig = None

		def plot_history(axs, history_data):
			# plt.xlabel('epoch')
			# plt.ylabel('training and testing loss')
			ax, ax1, ax2, ax3 = axs
			ax.set_xlabel('epoch')
			ax.set_ylabel('training and testing loss')
			ax.spines['top'].set_color('none')
			ax.spines['bottom'].set_color('none')
			ax.spines['left'].set_color('none')
			ax.spines['right'].set_color('none')
			ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
			ax1.set_title('training cost and testing cost')
			ax2.set_title('AE loss')
			ax3.set_title('RMSE loss')
			try:
				ax1.legend_.remove()
				ax2.legend_.remove()
				ax3.legend_.remove()
				ax1.lines.pop(0).remove()
				ax2.lines.pop(0).remove()
				ax3.lines.pop(0).remove()
			except Exception:
				pass
			ax1.plot(
				history_data['epoch'],
				history_data['training_loss_his'],
				'g-',
				lw=0.5,
				label='training loss')
			'''
			ax1.plot(
				history_data['epoch'],
				history_data['testting_loss_hist'],
				'r-',
				lw=0.5,
				label='testing loss')
			'''
			# ax1.legend()
			ax2.plot(
				history_data['epoch'],
				history_data['absolute_distance'],
				'b-',
				lw=0.5,
				label='absolute_distance loss')
			# ax2.legend()
			ax3.plot(
				history_data['epoch'],
				history_data['RMSE'],
				'r-',
				lw=0.5,
				label='RMSE loss')
			# ax3.legend()
			plt.pause(1)

		data = self._read_data_from_Tfrecord(self.training_file)
		batch_tuple_OP = tf.train.shuffle_batch(
			data,
			batch_size=self.batch_size,
			capacity=self.shuffle_capacity,
			min_after_dequeue=self.shuffle_min_after_dequeue)

		with tf.Session() as sess:
			if restore:
				print('reloading model....')
				self._reload_model(sess, model_path['pretrain_reload'])
			else:
				sess.run(tf.global_variables_initializer())
			coord = tf.train.Coordinator()
			treads = tf.train.start_queue_runners(sess=sess, coord=coord)
			tf.summary.FileWriter('logs/', sess.graph)
			for i, optimizer in enumerate(self.pre_optimizer_list):
				if i not in [0, 1, 2]:
					continue
				history_data = {
					'training_loss_his': [],
					'testting_loss_hist': [],
					'absolute_distance': [],
					'RMSE': [],
					'epoch': []
				}
				fig = plt.figure()
				plt.ion()
				ax = fig.add_subplot(1, 1, 1, frameon=False)
				ax1 = fig.add_subplot(3, 1, 1)
				ax2 = fig.add_subplot(3, 1, 2)
				ax3 = fig.add_subplot(3, 1, 3)
				axs = [ax, ax1, ax2, ax3]
				epoch = 1
				cumulate_loss = 0
				cumulate_RMSE = 0
				cumulate_abd = 0
				loss_set = self.pre_loss[i]
				while epoch < self.training_iters:
					index, batch_x, batch_y = sess.run(batch_tuple_OP)
					# print('index:{}'.format(index))
					loss = 0.

					with tf.device('/gpu:0'):
						_, loss, absolute_distance, L2, kl, RMSE, temp, MSE = sess.run([
							optimizer,
							loss_set['cost'],
							loss_set['MAE'],
							loss_set['L2'],
							loss_set['kl_diver'],
							loss_set['RMSE'],
							loss_set['temp'],
							loss_set['MSE']],
							feed_dict={
							self.Xs: batch_x,
							self.Ys: batch_y,
							self.keep_prob: self.dropout,
							self.norm: 1})
					print('i:%d Epoch:%d  cost:%g absolute_distance:%g L2:%g kl:%g RMSE:%g MSE:%g' % (i, epoch, loss, absolute_distance, L2, kl, RMSE, MSE))
					# print(temp)
					cumulate_loss += loss
					cumulate_RMSE += RMSE
					cumulate_abd += absolute_distance

					if epoch % self.display_step == 0:
						average_training_loss = cumulate_loss / self.display_step
						average_RMSE = cumulate_RMSE / self.display_step
						average_cumulate_abd = cumulate_abd / self.display_step
						index, testing_X, testing_Y = self._read_all_data_from_Tfreoced(
							self.testing_file)
						testing_loss, _ = self._testing_data(
							sess, testing_X, testing_X, 'pre_train')   # batch_x it_self
						print('testing_loss:{} average_training_loss:{} average absolute distance {} average RMSE {}'.format(
							testing_loss, average_training_loss, average_cumulate_abd, average_RMSE))
						history_data['epoch'].append(epoch)
						history_data['training_loss_his'].append(average_training_loss)
						history_data['testting_loss_hist'].append(0)
						history_data['absolute_distance'].append(average_cumulate_abd)
						history_data['RMSE'].append(average_RMSE)
						cumulate_loss = 0
						cumulate_RMSE = 0
						cumulate_abd = 0
						plot_history(axs, history_data)
						
					if epoch % 200 == 0 and epoch != 0:
						self._save_model(sess, model_path['pretrain_save'])
						# self._save_model(sess, model_path['save_weight_bias'])
						self._save_history(history_data)
					epoch += 1
			coord.request_stop()
			coord.join(treads)
			print('pre training finished!')
			# plt.ioff()
			self._save_model(sess, model_path['pretrain_save'])

	def start_train(self, model_path, restore=False):

		fig = None

		def plot_history(axs, history_data):
			ax, ax1, ax2, ax3 = axs
			ax.set_xlabel('epoch')
			ax.set_ylabel('training and testing loss')
			ax.spines['top'].set_color('none')
			ax.spines['bottom'].set_color('none')
			ax.spines['left'].set_color('none')
			ax.spines['right'].set_color('none')
			ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
			ax1.set_title('training cost and testing cost')
			ax2.set_title('AE loss')
			ax3.set_title('RMSE loss')
			try:
				ax1.lines.pop(0).remove()
				ax2.lines.pop(0).remove()
				ax3.lines.pop(0).remove()
			except Exception:
				pass
			ax1.plot(
				history_data['epoch'],
				history_data['training_loss_his'],
				'g-',
				lw=0.5,
				label='training_loss')
			ax1.plot(
				history_data['epoch'],
				history_data['testing_loss_his'],
				'r-',
				lw=0.5,
				label='testing loss')
			# ax1.legend()
			ax2.plot(
				history_data['epoch'],
				history_data['absolute_distance'],
				'b-',
				lw=0.5,
				label='absolute_distance loss')
			# ax2.legend()
			ax3.plot(
				history_data['epoch'],
				history_data['RMSE'],
				'r-',
				lw=0.5,
				label='RMSE loss')
			# ax3.legend()
			plt.pause(1)
		data = self._read_data_from_Tfrecord(self.training_file)
		# stop_pre_train_gradient_OP = tf.stop_gradient(self.pre_traion_output)
		batch_tuple_OP = tf.train.shuffle_batch(
			data,
			batch_size=self.batch_size,
			capacity=self.shuffle_capacity,
			min_after_dequeue=self.shuffle_min_after_dequeue)
		cum_value = {
			'cost': 0,
			'AE_loss': 0,
			'SE_loss': 0,
			'RMSE_loss': 0,
		}
		history_data = {
			'training_loss_his': [],
			'testing_loss_his': [],
			'absolute_distance': [],
			'RMSE': [],
			'epoch': []
		}
		fig = plt.figure()
		plt.ion()
		ax = fig.add_subplot(1, 1, 1, frameon=False)
		ax1 = fig.add_subplot(3, 1, 1)
		ax2 = fig.add_subplot(3, 1, 2)
		ax3 = fig.add_subplot(3, 1, 3)
		axs = [ax, ax1, ax2, ax3]
		with tf.Session() as sess:
			epoch = 1
			# restore pre-train model
			if restore:
				print('reloading model....')
				self._reload_model(sess, model_path['reload'])
			else:
				sess.run(tf.global_variables_initializer())
				print('reloading pre_train model....')
				self._reload_model(sess, model_path['pretrain_reload'])
			coord = tf.train.Coordinator()
			treads = tf.train.start_queue_runners(sess=sess, coord=coord)
			with tf.device('/gpu:0'):
				while epoch < self.training_iters:
					index, batch_x, batch_y = sess.run(batch_tuple_OP)
					loss = 0
					_, loss, MSE, MAE, RMSE, L2 = sess.run([
						self.predictor_optimization,
						self.predictor_loss['cost'],
						self.predictor_loss['MSE'],
						self.predictor_loss['MAE'],
						self.predictor_loss['RMSE'],
						self.predictor_loss['L2']],
						feed_dict={
						self.Xs: batch_x,
						self.Ys: batch_y,
						self.keep_prob: self.dropout,
						self.norm: 1})

					pre_train_loss = sess.run(self.pre_loss_1['cost'], feed_dict={
						self.Xs: batch_x,
						self.Ys: batch_y,
						self.keep_prob: self.dropout,
						self.norm: 1})
					print('epoch: {} loss: {:.4f} pre_train_loss:{:.4f} MSE:{:.4f} MAE:{:.4f} RMSE:{:.4f} L2:{:.4f}'.format(
						epoch,
						loss,
						pre_train_loss,
						MSE,
						MAE,
						RMSE,
						L2))
					cum_value['cost'] += loss
					cum_value['SE_loss'] += MSE
					cum_value['AE_loss'] += MAE
					cum_value['RMSE_loss'] += RMSE
					if epoch % self.display_step == 0:
						average_AE_loss = cum_value['AE_loss'] / self.display_step
						average_SE_loss = cum_value['SE_loss'] / self.display_step
						average_RMSE = cum_value['RMSE_loss'] / self.display_step
						average_cost = cum_value['cost'] / self.display_step
						index, testing_X, testing_Y = self._read_all_data_from_Tfreoced(self.testing_file)
						testing_loss, _ = self._testing_data(sess, testing_X, testing_Y, 'train')
						print('testing_loss:{} average_training_loss:{}'.format(testing_loss, average_SE_loss))

						history_data['epoch'].append(epoch)
						history_data['testing_loss_his'].append(testing_loss)
						history_data['training_loss_his'].append(average_cost)
						history_data['RMSE'].append(average_RMSE)
						history_data['absolute_distance'].append(average_AE_loss)
						plot_history(axs, history_data)
						cum_value['cost'] = 0
						cum_value['SE_loss'] = 0
						cum_value['AE_loss'] = 0
						cum_value['RMSE_loss'] = 0

					if epoch % 200 == 0 and epoch != 0:
						self._save_model(sess, model_path['save'])
					epoch += 1
			coord.request_stop()
			coord.join(treads)
			print('training finished!')
