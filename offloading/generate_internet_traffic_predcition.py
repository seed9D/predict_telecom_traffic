from env import Internet_Traffic, compute_row_col, comput_grid_id

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



def loop_all_grid():

	def check_grid_completion(grid_list, traffic):
		Y_info = traffic.Y_info
		for row_ in range(Y_info.shape[2]):
			for col in range(Y_info.shape[3]):
				grid_id = Y_info[0, 0, row_, col, 0]
				print(grid_id)
				grid_list.append(grid_id)
		return grid_list

	# output_col_range = list(range(13, 85, 3))

	traffic = Internet_Traffic()

	row_center_list = list(range(40, 80, 3))
	col_center_list = list(range(30, 70, 3))
	'''
	gird_id, timestamp, real_min, real_avg, real_max, prediction_min, prediction_avg, prediction_max
	'''

	traffic_array = np.zeros([1487, 1, grid_row_num, grid_column_num, 8], dtype=float)

	for row_center in row_center_list:
		for col_center in col_center_list:
			print('row_center:{} col_center:{}'.format(row_center, col_center))
			grid_id = comput_grid_id(row_center, col_center)
			traffic.set_grid_id(grid_id)
			# if row_center == row_center_list[0] and col_center ==
			# col_center_list[0]:
			traffic.set_NN_mode()
			traffic.run_train()
			traffic.run_predict()
			traffic_data = traffic.fetch_real_prediction()

			for i in range(traffic_data.shape[0]):
				for j in range(traffic_data.shape[1]):
					for row in range(traffic_data.shape[2]):
						for col in range(traffic_data.shape[3]):
							grid_id = traffic_data[i, j, row, col, 0]
							row_index, col_index = compute_row_col(grid_id)
							# print(row_index, col_index)
							traffic_array[i, j, row_index, col_index] = traffic_data[
								i, j, row, col]
							# print(traffic_array[i, j, row_index, col_index])

		du.save_array(traffic_array, './result/real_prediction_traffic_array')
	du.save_array(traffic_array, './result/real_prediction_traffic_array')


def nine_grid():

	traffic = Internet_Traffic()
	traffic.set_grid_id(3257)
	traffic.set_NN_mode()
	traffic.run_train()
	traffic.run_predict()
	# traffic.fetch_real_prediction()
	# compute_row_col()

if __name__ == "__main__":
	# nine_grid()
	# loop_all_grid()