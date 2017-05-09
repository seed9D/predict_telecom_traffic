import tensorflow as tf
import tensorlayer as tl
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
import os

class CNN_RNN:
	def __init__(self, *data_shape):
		self.batch_size = 100
		self.shuffle_min_after_dequeue = 600
		self.shuffle_capacity = self.shuffle_min_after_dequeue + 3 * self.batch_size
		learning_rate = 0.001
		beta_1 = 0.9
		beta_2 = 0.999
		self.input_temporal = data_shape[0]
		self.input_vertical = data_shape[1]
		self.input_horizontal = data_shape[2]
		self.input_channel = data_shape[3]

		self.output_temporal = 1
		self.output_vertical = data_shape[1]
		self.output_horizontal = data_shape[2]
		self.output_channel = data_shape[3]
		predictor_output = self.output_temporal * self.output_vertical * self.output_horizontal * self.output_channel

		self.RNN_num_layers = 3
		self.RNN_num_step = 6
		self.RNN_hidden_node_size = 150

		# placeholder
		with tf.device('/cpu:0'):
			self.Xs = tf.placeholder(tf.float32, shape=[
				None, self.input_temporal, self.input_vertical, self.input_horizontal, self.input_channel], name='Input_x')
			self.Ys = tf.placeholder(tf.float32, shape=[
				None, self.output_temporal, self.output_vertical, self.output_horizontal, self.output_channel], name='Input_y')
			self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
			self.norm = tf.placeholder(tf.bool, name='norm')
			self.RNN_init_state = tf.placeholder(tf.float32, [self.RNN_num_layers, 2, None, self.RNN_hidden_node_size])  # 2: hidden state and cell state

		# variables

			self.weights = {
				'output_layer': self._weight_variable([self.RNN_hidden_node_size, predictor_output], 'output_layer_w')
			}
			self.bias = {
				'output_layer': self._bias_variable([predictor_output], 'output_layer_b')
			}

		# operation

		self.tl_CNN_output = self._build_CNN_network(self.Xs, is_training=1)
		# self.tl_RNN_output = self._build_RNN_network(self.tl_CNN_output)
		# self.tl_output_layer = tl.layers.DenseLayer(self.tl_RNN_output, n_units=self.output_vertical * self.output_horizontal, act=tl.activation.identity, name='output_layer')
		network = tl.layers.FlattenLayer(self.tl_CNN_output, name='flatten_layer')
		'''
		self.RNN_states_series, self.RNN_current_state = self._build_RNN_network_tf(network, self.keep_prob)
		RNN_last_output = tf.unpack(tf.transpose(self.RNN_states_series, [1, 0, 2]))  # a (batch_size, state_size) list with len num_step
		output_layer = tf.add(tf.matmul(RNN_last_output[-1], self.weights['output_layer']), self.bias['output_layer'])
		'''
		self.tl_RNN_output = self._build_RNN_network(network, is_training=1)
		network = tl.layers.DenseLayer(self.tl_RNN_output, n_units=25 * 10, act=tf.nn.relu, name='dense_layer')
		network = tl.layers.DropoutLayer(network, keep=0.7, name='drop_1')
		self.tl_output_layer = tl.layers.DenseLayer(network, n_units=predictor_output, act=tl.activation.identity, name='output_layer')
		output_layer = self.tl_output_layer.outputs

		self.output_layer = tf.reshape(output_layer, [-1, self.output_temporal, self.output_vertical, self.output_horizontal, self.output_channel])
		self.cost_op = tf.reduce_mean(tf.pow(self.output_layer - self.Ys, 2))

		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta_1, beta2=beta_2)
		opt_vars = tf.trainable_variables()
		gvs = optimizer.compute_gradients(self.cost_op, var_list=opt_vars)
		capped_gvs = [(tf.clip_by_norm(grad, 5), var) for grad, var in gvs if grad is not None]
		self.optimizer_op = optimizer.apply_gradients(capped_gvs)

		self.saver = tf.train.Saver()

	def _build_CNN_network(self, input_X, is_training=1):
		with tf.device('/cpu:0'):
			with tf.variable_scope('CNN'):
				CNN_input = tf.reshape(input_X, [-1, self.input_vertical, self.input_horizontal, self.input_channel])
				print('CNN_input shape:{}'.format(CNN_input))
				network = tl.layers.InputLayer(CNN_input, name='input_layer')
				network = tl.layers.BatchNormLayer(network, name='batchnorm_layer_1')
				network = tl.layers.Conv2dLayer(
					network,
					act=tf.nn.relu,
					shape=[5, 5, 1, 32],
					strides=[1, 1, 1, 1],
					padding='SAME',
					name='cnn_layer_1')

				network = tl.layers.PoolLayer(
					network,
					ksize=[1, 2, 2, 1],
					strides=[1, 2, 2, 1],
					padding='SAME',
					pool=tf.nn.max_pool,
					name='pool_layer_1')
				network = tl.layers.DropoutLayer(network, keep=0.7, name='drop_1')
				network = tl.layers.Conv2dLayer(
					network,
					act=tf.nn.relu,
					shape=[5, 5, 32, 32],
					strides=[1, 1, 1, 1],
					padding='SAME',
					name='cnn_layer_2')

				network = tl.layers.PoolLayer(
					network,
					ksize=[1, 2, 2, 1],
					strides=[1, 2, 2, 1],
					padding='SAME',
					pool=tf.nn.max_pool,
					name='pool_layer_2')
				# network = tl.layers.DropoutLayer(network, keep=0.7, name='drop_2')
				# network = tl.layers.FlattenLayer(network, name='flatten_layer')

				# network = tl.layers.DenseLayer(network, n_units=512, act=tf.nn.relu, name='fc_1')
				# network = tl.layers.DropoutLayer(network, keep=0.7, name='drop_3')
				print('network output shape:{}'.format(network.outputs.get_shape()))
			return network

	def _build_RNN_network_tf(self, input_X, keep_rate):
		def _get_states():
			state_per_layer_list = tf.unpack(self.RNN_init_state, axis=0)
			rnn_tuple_state = tuple(
				[tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1]) for idx in range(self.RNN_num_layers)])
			return rnn_tuple_state
		with tf.variable_scope('RNN'):
			input_X = input_X.outputs
			print('rnn network input shape :{}'.format(input_X.get_shape()))
			input_X = tf.reshape(input_X, [-1, self.RNN_num_step, input_X.get_shape().as_list()[-1]])
			print('rnn reshape input shape :{}'.format(input_X.get_shape()))
			RNN_cell = tf.nn.rnn_cell.BasicLSTMCell(self.RNN_hidden_node_size, state_is_tuple=True)
			RNN_cell = tf.nn.rnn_cell.DropoutWrapper(RNN_cell, output_keep_prob=keep_rate)
			RNN_cell = tf.nn.rnn_cell.MultiRNNCell([RNN_cell] * self.RNN_num_layers, state_is_tuple=True)

			status_tuple = _get_states()
			states_series, current_state = tf.nn.dynamic_rnn(RNN_cell, input_X, initial_state=status_tuple, time_major=False)

		return states_series, current_state

	def _build_RNN_network(self, input_X, is_training=1):
		print('rnn network input shape :{}'.format(input_X.outputs.get_shape()))
		# network = tl.layers.FlattenLayer(input_X, name='flatten_layer')  # [batch_size, mask_row, mask_col, n_mask] —> [batch_size, mask_row * mask_col * n_mask]
		with tf.variable_scope('RNN'):
			input_X = tl.layers.BatchNormLayer(input_X, name='batchnorm_layer_1')
			network = tl.layers.ReshapeLayer(input_X, shape=[-1, self.RNN_num_step, int(input_X.outputs._shape[-1])], name='reshape_layer_1')
			network = tl.layers.RNNLayer(
				network,
				cell_fn=tf.nn.rnn_cell.BasicLSTMCell,
				cell_init_args={'forget_bias': 0.0},
				n_hidden=self.RNN_hidden_node_size,
				initializer=tf.random_uniform_initializer(-0.1, 0.1),
				n_steps=self.RNN_num_step,
				initial_state=None,
				return_last=False,
				return_seq_2d=False,  # trigger with return_last is False. if True, return shape: (?, 200); if False, return shape: (?, 6, 200)
				name='basic_lstm_layer_1')
			if is_training:
				network = tl.layers.DropoutLayer(network, keep=0.7, name='drop_1')
			network = tl.layers.RNNLayer(
				network,
				cell_fn=tf.nn.rnn_cell.BasicLSTMCell,
				cell_init_args={'forget_bias': 0.0},
				n_hidden=self.RNN_hidden_node_size,
				initializer=tf.random_uniform_initializer(-0.1, 0.1),
				n_steps=self.RNN_num_step,
				initial_state=None,
				return_last=True,
				return_seq_2d=False,  # trigger with return_last is False. if True, return shape: (?, 200); if False, return shape: (?, 6, 200)
				name='basic_lstm_layer_2')
			if is_training:
				network = tl.layers.DropoutLayer(network, keep=0.7, name='drop_2')

		'''
		network = tl.layers.DynamicRNNLayer(
			network,
			cell_fn=tf.nn.rnn_cell.BasicLSTMCell,
			n_hidden=64,
			initializer=tf.random_uniform_initializer(-0.1, 0.1),
			n_steps=num_step,
			return_last=False,
			return_seq_2d=True,
			name='basic_lstm_layer_2')
		'''
		return network

	def _weight_variable(self, shape, name):
		# initial = tf.truncated_normal(shape, stddev=0.1)
		initial = np.random.randn(*shape) * sqrt(2.0 / np.prod(shape))
		return tf.Variable(initial, dtype=tf.float32, name=name)

	def _bias_variable(self, shape, name):
		# initial = tf.random_normal(shape)
		initial = np.random.randn(*shape) * sqrt(2.0 / np.prod(shape))
		return tf.Variable(initial, dtype=tf.float32, name=name)

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

	def _save_model(self, sess):
		model_path = './output_model/CNN_RNN.ckpt'
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

	def _reload_model(self, sess, model_path):
		print('reloading model {}.....'.format(model_path))
		self.saver.restore(sess, model_path)

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
		X_data = input_x
		Y_data = input_y

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

	def _testing_data(self, sess, test_x, test_y):
		predict_list = []
		cum_loss = 0
		batch_num = test_x.shape[0] // self.batch_size
		# _current_state = np.zeros([self.RNN_num_layers, 2, self.batch_size, self.RNN_hidden_node_size])
		print('batch_num:', batch_num)
		for batch_index in range(batch_num):
			dp_dict = tl.utils.dict_to_one(self.tl_output_layer.all_drop)
			batch_x = test_x[batch_index * self.batch_size: (batch_index + 1) * self.batch_size]
			batch_y = test_y[batch_index * self.batch_size: (batch_index + 1) * self.batch_size]
			feed_dict = {
				self.Xs: batch_x,
				self.Ys: batch_y,
				self.keep_prob: 1,
				self.norm: 0}
			feed_dict.update(dp_dict)
			with tf.device('/gpu:0'):
				loss, predict = sess.run([self.cost_op, self.output_layer], feed_dict=feed_dict)

			for i in range(4):
				for j in range(predict.shape[1]):
					print('batch index: {} predict:{:.4f} real:{:.4f}'.format(batch_index, predict[i, j, 10, 10, 0], batch_y[i, j, 10, 10, 0]))
			print()
			for predict_element in predict:
				predict_list.append(predict_element)
			cum_loss += loss
		return cum_loss / batch_num, np.stack(predict_list)

	def start_predict(self, testing_x, testing_y, model_path):
		print('input_x shape {}'.format(testing_x.shape))
		print('input_y shape {}'.format(testing_y.shape))

		with tf.Session() as sess:
			loss = 0
			self._reload_model(sess, model_path['reload_path'])
			loss, predict = self._testing_data(sess, testing_x, testing_y)
		print('preddict finished!')
		print('prediction loss:{} predict array shape:{}'.format(loss, predict.shape))
		return loss, predict

	def start_train(self):
		training_loss = 0
		display_step = 100
		train_his = {
			'epoch': [],
			'training_cost': [],
			'testing_cost': []
		}
		plt.ion()
		plt.figure()

		def _plot_loss_rate(history_data):
			plt.subplot(1, 1, 1)
			plt.cla()
			plt.plot(history_data['epoch'], history_data['training_cost'], 'g-', label='training losses')
			plt.plot(history_data['epoch'], history_data['testing_cost'], 'b-', label='testing losses')

			# plt.plot(test_cost, 'r-', label='testing losses')
			plt.legend()
			plt.draw()
			plt.pause(0.001)

		data = self._read_data_from_Tfrecord(self.training_file)
		batch_tuple_OP = tf.train.shuffle_batch(
			data,
			batch_size=self.batch_size,
			capacity=self.shuffle_capacity,
			min_after_dequeue=self.shuffle_min_after_dequeue)

		with tf.Session() as sess:
			coord = tf.train.Coordinator()
			treads = tf.train.start_queue_runners(sess=sess, coord=coord)
			tf.summary.FileWriter('logs/', sess.graph)
			sess.run(tf.global_variables_initializer())
			# _current_state = np.zeros([self.RNN_num_layers, 2, self.batch_size, self.RNN_hidden_node_size])
			with tf.device('/gpu:0'):
				for epoch in range(20000):
					index, batch_x, batch_y = sess.run(batch_tuple_OP)
					feed_dict = {
						self.Xs: batch_x,
						self.Ys: batch_y,
						self.keep_prob: 0.7,
						self.norm: 1}
					feed_dict.update(self.tl_output_layer.all_drop)
					_, loss = sess.run([self.optimizer_op, self.cost_op], feed_dict=feed_dict)
					training_loss += loss
					if epoch % display_step == 0 and epoch is not 0:
						'''testing'''
						index, testing_X, testing_Y = self._read_all_data_from_Tfreoced(self.testing_file)
						testing_loss, _ = self._testing_data(sess, testing_X, testing_Y)
						training_loss = training_loss / display_step
						train_his['training_cost'].append(training_loss)
						train_his['epoch'].append(epoch)
						train_his['testing_cost'].append(testing_loss)
						print('epoch:{}, training cost:{:.4f}, testing cost:{:.4f}'.format(epoch, training_loss, testing_loss))
						_plot_loss_rate(train_his)
						training_loss = 0
					if epoch % 500 == 0 and epoch is not 0:
						self._save_model(sess)
				coord.request_stop()
				coord.join(treads)
				print('training finished!')
			plt.ioff()
			plt.show()

if __name__ == '__main__':
	data_shape = [6, 25, 25, 1]
	cnn_rnn = CNN_RNN(*data_shape)
