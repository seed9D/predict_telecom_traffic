import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import sys
import pytz
import os
from datetime import datetime
sys.path.append('/home/mldp/ML_with_bigdata')
# import .data_utility as du
import data_utility as du
import numpy as np

input_file_x = '/home/mldp/ML_with_bigdata/npy/11_0.npy'
input_file_y = '/home/mldp/ML_with_bigdata/npy/final/testing_raw_data/one_hour_max_value/one_hour_max_0.npy'


def set_time_zone(timestamp):
	UTC_timezone = pytz.timezone('UTC')
	Mi_timezone = pytz.timezone('Europe/Rome')
	date_time = datetime.utcfromtimestamp(float(timestamp))
	date_time = date_time.replace(tzinfo=UTC_timezone)
	date_time = date_time.astimezone(Mi_timezone)
	return date_time


def date_time_covert_to_str(date_time):
	return date_time.strftime('%Y-%m-%d %H:%M')


def prepare_data(input_file):
	data_array = du.load_array(input_file)
	print('saving array shape:{}'.format(data_array.shape))
	# du.save_array(data_array, './npy/autoregrssion_raw_data')

	i_len = data_array.shape[0]
	j_len = data_array.shape[1]
	row = 10
	col = 20
	data_frame = {
		'date': [],
		'internet': []
	}
	for i in range(i_len):
		for j in range(j_len):
			date_string = set_time_zone(data_array[i, j, row, col, 1])
			date_string = date_time_covert_to_str(date_string)
			data_frame['internet'].append(data_array[i, j, row, col, -1])
			data_frame['date'].append(date_string)
	data_frame = pd.DataFrame(data_frame)
	return data_frame


class RNN_Traffic():
	def __init__(self):

		self.truncated_backprop_length = 6
		self.batch_size = 24
		self.num_inputs = 1
		self.num_outputs = 1
		self.state_size = 150  # number of units in RNN cell
		self.learning_rate = 0.0001
		self.weight_decay = 0.01
		self.num_layers = 4
		self.dropout_rate = 0.6
		self.hidden_layer_1 = self.truncated_backprop_length * 10

		self.X = tf.placeholder(tf.float32, [self.batch_size, self.truncated_backprop_length], name='input_placeholder')
		self.Y = tf.placeholder(tf.float32, [self.batch_size, self.num_outputs], name='result_placeholder')
		self.keep_prob = tf.placeholder(tf.float32, name='pre_keep_prob')
		self.init_state = tf.placeholder(tf.float32, [self.num_layers, 2, self.batch_size, self.state_size])  # 2: hidden state and cell state
		state_per_layer_list = tf.unpack(self.init_state, axis=0)
		self.rnn_tuple_state = tuple(
			[tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1]) for idx in range(self.num_layers)])

		self.weights = {
			'in': tf.Variable(tf.random_normal([self.truncated_backprop_length, self.hidden_layer_1]), name='weights_in'),
			'out': tf.Variable(tf.random_normal([self.state_size, self.num_outputs]), name='weights_out')
		}

		self.bias = {
			'in': tf.Variable(tf.random_normal(shape=[self.hidden_layer_1, ], stddev=0.1), name='Bias_in'),
			'out': tf.Variable(tf.random_normal(shape=[self.num_outputs, ], stddev=0.1), name='Bias_out')
		}
		input_rnn = tf.add(tf.matmul(self.X, self.weights['in']), self.bias['in'])
		input_rnn = tf.nn.relu(input_rnn)
		# input_rnn = tf.nn.dropout(input_rnn, self.keep_prob)
		self.states_series, self.current_state = self._build_RNN(input_rnn, self.rnn_tuple_state, self.keep_prob)
		last_output = tf.unstack(tf.transpose(self.states_series, [1, 0, 2]))  # a (batch_size, state_size) list with len num_step
		'''
			last_output is a list with len 6
				each element's shape (24, 64) <---->(batch_size, state_size)
		'''
		# states_series_flat = tf.reshape(states_series, [batch_size, -1, state_size])

		self.predict_maxvalue = tf.matmul(last_output[-1], self.weights['out']) + self.bias['out']
		opt_vars = tf.trainable_variables()
		self.L2_loss = self._L2_loss(opt_vars)
		self.RMSE_loss = tf.sqrt(tf.reduce_mean(tf.pow(self.predict_maxvalue - self.Y, 2)))
		self.cost_op = self.RMSE_loss + self.L2_loss * self.weight_decay  # RMSE
		optimizer = tf.train.AdamOptimizer(self.learning_rate)
		gvs = optimizer.compute_gradients(self.cost_op, var_list=opt_vars)
		capped_gvs = [(tf.clip_by_norm(grad, 5), var) for grad, var in gvs if grad is not None]
		self.train_op = optimizer.apply_gradients(capped_gvs)
		# self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost_op)
		# self.init = tf.global_variables_initializer()
		self.saver = tf.train.Saver()

	def _L2_loss(self, vars_list):
		L2_loss = tf.add_n([tf.nn.l2_loss(v) for v in vars_list if 'Bias' not in v.name])
		for v in vars_list:
			if 'Bias' not in v.name:
				print(v)
		return L2_loss

	def _build_RNN(self, X, rnn_tuple_state, keep_rate):
		# x_batch_size = x.get_shape()[0]
		# X = tf.matmul(X, weights['in']) + biases['in']
		# X = tf.reshape(X, [-1, truncated_backprop_length, state_size])
		# rnn_inputs = tf.unstack(X, axis=1)
		with tf.variable_scope('RNN'):
			X = tf.reshape(X, [self.batch_size, self.truncated_backprop_length, -1])
			RNNcell = tf.nn.rnn_cell.BasicLSTMCell(self.state_size, state_is_tuple=True)  # first para is the number of units in the LSTM cell
			RNNcell = tf.nn.rnn_cell.DropoutWrapper(RNNcell, output_keep_prob=keep_rate)
			RNNcell = tf.nn.rnn_cell.MultiRNNCell([RNNcell] * self.num_layers, state_is_tuple=True)
			init_state = RNNcell.zero_state(self.batch_size, dtype=tf.float32)
			states_series, current_state = tf.nn.dynamic_rnn(RNNcell, X, initial_state=init_state, time_major=False)
		# print(current_state)
		'''
			current_state:
				3 LSTMSatteTuple: (2, 24, 64)
					2 :belongs to hidden states and cell states
					24 : batch size
					64 : states size
			states_series:
				shape [batch_size, truncated_backprop_length, state_size]
		'''

		return states_series, current_state

	def _shuffle(self, input_x, input_y):
		# input_x = np.reshape(input_x, [-1, batch_size, num_inputs])
		# input_y = np.reshape(input_y, [-1, batch_size, num_outputs])
		# print(input_x.shape, input_y.shape)
		z = list(zip(input_x, input_y))
		np.random.shuffle(z)
		input_x, input_y = zip(*z)

		data_length = len(input_x)

		batch_partition_length = data_length // self.batch_size
		# print(data_length)
		data_x = np.zeros([batch_partition_length, self.batch_size, self.truncated_backprop_length], dtype=np.float32)
		data_y = np.zeros([batch_partition_length, self.batch_size, self.num_outputs], dtype=np.float32)

		for i in range(batch_partition_length - 1):
			data_x[i] = input_x[i * self.batch_size: (i + 1) * self.batch_size]
			data_y[i] = input_y[i * self.batch_size: (i + 1) * self.batch_size]
		return data_x, data_y

	def _plot_loss_rate(self, training_losses, test_cost):
		plt.subplot(1, 1, 1)
		plt.cla()
		plt.plot(training_losses, 'g-', label='training losses')
		plt.plot(test_cost, 'r-', label='testing losses')
		plt.legend()
		plt.draw()
		plt.pause(0.001)

	def testing_data(self, sess, test_x, test_y):
		predict, cost = sess.run([self.predict_maxvalue, self.cost_op], feed_dict={
			self.X: test_x,
			self.Y: test_y,
			self.keep_prob: 1})
		# print(predict.shape)
		for time_ in range(len(test_y)):
			print('predict:{:.4f} real:{:.4f}'.format(predict[time_, 0], test_y[time_, 0]))
		return cost

	def _save_model(self, sess):
		model_path = './output_model/RNN.ckpt'
		print('saving model.....')
		if not os.path.exists(model_path):
			os.makedirs(model_path)
		try:
			save_path = self.saver.save(sess, model_path)
			# self.pre_train_saver.save(sess, model_path + '_part')
		except Exception:
			save_path = self.saver.save(sess, './output_model/temp.ckpt')
		finally:
			print('save_path{}'.format(save_path))

	def _reload_model(self, sess):
		model_path = './output_model/RNN.ckpt'
		print('reloading model {}.....'.format(model_path))
		self.saver.restore(sess, model_path)

	def predict_rnn(self):
		def plot_result(real, predict):
			print('real shape:{} predict shape:{}'.format(real.shape, predict.shape))
			plt.figure()
			plt.subplot(1, 1, 1)
			plt.plot(real, 'g-', label="real")
			plt.plot(predict, 'r:', label='predict')
			plt.legend()
			plt.show()

		train_x, train_y = self.train_x, self.train_y
		test_x, test_y = self.test_x, self.test_y
		_current_state = np.zeros([self.num_layers, 2, self.batch_size, self.state_size])
		with tf.Session() as sess:
			self._reload_model(sess)
			'''plot testing'''
			predic_v, cost_c = sess.run(
				[self.predict_maxvalue, self.RMSE_loss],
				feed_dict={
					self.X: test_x,
					self.Y: test_y,
					self.keep_prob: 1,
					self.init_state: _current_state})
			print('testing set RMSE:{}'.format(cost_c))
			plot_result(test_y, predic_v)

			train_batch_num = train_x.shape[0] // self.batch_size
			''' plot training '''
			predict_list = []
			train_cost = 0
			for batch_index in range(train_batch_num):
				batch_x = train_x[batch_index * self.batch_size: (batch_index + 1) * self.batch_size]
				batch_y = train_y[batch_index * self.batch_size: (batch_index + 1) * self.batch_size]
				predic_v, cost_c = sess.run(
					[self.predict_maxvalue, self.RMSE_loss],
					feed_dict={
						self.X: batch_x,
						self.Y: batch_y,
						self.keep_prob: 1,
						self.init_state: _current_state})
				predict_list.extend(predic_v)
				train_cost += cost_c
			print('training set RMSe:{}'.format(train_cost / train_batch_num))
			plot_result(train_y[:200], np.array(predict_list[:200]))

	def train_rnn(self):
		train_x, train_y = self.train_x, self.train_y
		test_x, test_y = self.test_x, self.test_y

		# input_x = np.reshape(input_x, [-1, batch_size, num_inputs])
		# input_y = np.reshape(input_y, [-1, batch_size, num_outputs])
		with tf.Session() as sess:
			# sess.run(tf.global_variables_initializer())
			self._reload_model(sess)
			plt.ion()
			plt.figure()
			training_losses = []
			testing_lossses = []
			for epoch in range(10000):
				outputs = self._shuffle(train_x, train_y)
				_current_state = np.zeros([self.num_layers, 2, self.batch_size, self.state_size])
				training_loss = 0
				batch_num = 0
				for index, output in enumerate(zip(*outputs)):

					batch_x, batch_y = output
					# print(batch_x.shape, batch_y.shape)
					_, cost, _current_state, L2_loss = sess.run([self.train_op, self.cost_op, self.current_state, self.L2_loss], feed_dict={
						self.X: batch_x,
						self.Y: batch_y,
						self.keep_prob: self.dropout_rate,
						self.init_state: _current_state})
					training_loss += cost
					batch_num = index
				training_loss = training_loss / batch_num
				if epoch % 50 == 0 and epoch != 0:
					test_cost = self.testing_data(sess, test_x, test_y)
					training_losses.append(training_loss)
					testing_lossses.append(test_cost)
					self._plot_loss_rate(training_losses, testing_lossses)
					print('epoch:{} training cost:{:.4f} testng cost:{:.4f} L2_loss:{:.4f}'.format(epoch, training_loss, test_cost, L2_loss))

				if epoch % 200 == 0 and epoch is not 0:
					self._save_model(sess)
			print('training finished !')
			self._save_model(sess)
		plt.ioff()
		plt.show()

	def set_training_data(self, input_X, input_Y):
		input_X = input_X.reshape([-1, self.truncated_backprop_length])
		input_Y = input_Y.reshape([-1, self.num_outputs])
		input_X = input_X[:-1]
		input_Y = input_Y[1:]
		print(input_X.shape)
		print(input_Y.shape)
		self.train_x, self.test_x = input_X[:input_X.shape[0] - 24], input_X[input_X.shape[0] - 24:]
		self.train_y, self.test_y = input_Y[:input_Y.shape[0] - 24], input_Y[input_Y.shape[0] - 24:]


if __name__ == '__main__':

	data_x = prepare_data(input_file_x)
	input_X = data_x['internet'].values
	data_y = prepare_data(input_file_y)
	input_Y = data_y['internet'].values

	rnn = RNN_Traffic()
	rnn.set_training_data(input_X, input_Y)
	rnn.train_rnn()
	# rnn.predict_rnn()
	'''
	data_internet_len = len(data_internet)
	train, test = data_internet[:int(data_internet_len - 144)], data_internet[int(data_internet_len - 144):]
	'''
