import data_utility as du
import numpy as np
import CNN_autoencoder as cn
from sklearn import preprocessing


def prepare_training_data():
	'''
	input_dir_list = [
			"/home/mldp/big_data/openbigdata/milano/SMS/11/data_preproccessing_10/",
			"/home/mldp/big_data/openbigdata/milano/SMS/12/data_preproccessing_10/"]

	for input_dir in input_dir_list:
			filelist = list_all_input_file(input_dir)
			filelist.sort()
			load_data_format(input_dir, filelist)
	'''
	def task_1():
		'''
		X: past one hour 
		Y: next hour's max value
		'''
		filelist = du.list_all_input_file('./npy/')
		filelist.sort()
		for i, filename in enumerate(filelist):
			if filename != 'training_raw_data.npy':
				data_array = du.load_array('./npy/' + filename)
				data_array = data_array[:, :, 40:65, 40:65, -1, np.newaxis]  # only network activity
				print('saving  array shape:{}'.format(data_array.shape))
				du.save_array(
					data_array, './npy/final/training_raw_data' + '_' + str(i))

				max_array = du.get_MAX_internet_array(data_array)
				du.save_array(max_array, './npy/final/one_hour_max_value/one_hour_max' + '_' + str(i))

	def task_2():
		'''
		rolling 10 minutes among timeflows
		X: past one hour
		Y: next 10 minutes value
		'''
		filelist_X = du.list_all_input_file('./npy/npy_roll/X/')
		filelist_Y = du.list_all_input_file('./npy/npy_roll/Y/')
		filelist_X.sort()
		filelist_Y.sort()
		for i, filename in enumerate(filelist_X):
			data_array = du.load_array('./npy/npy_roll/X/' + filename)
			data_array = data_array[:, :, 40:65, 40:65, -1, np.newaxis]  # only network activity
			print('saving  array shape:{}'.format(data_array.shape))
			du.save_array(data_array, './npy/npy_roll/final/X/' + 'X_' + str(i))

		for i, filename in enumerate(filelist_Y):
			data_array = du.load_array('./npy/npy_roll/Y/' + filename)
			data_array = data_array[:, :, 40:65, 40:65, -1, np.newaxis]  # only network activity
			print('saving  array shape:{}'.format(data_array.shape))
			du.save_array(data_array, './npy/npy_roll/final/Y/' + 'Y_' + str(i))

	task_2()


def get_X_and_Y_array():

	def task_1():
		'''
		X: past one hour
		Y: next hour's max value
		'''
		training_data_list = du.list_all_input_file('./npy/final/')
		training_data_list.sort()
		X_array_list = []
		for filename in training_data_list:
			X_array_list.append(du.load_array('./npy/final/' + filename))

		X_array = np.concatenate(X_array_list, axis=0)
		# X_array = X_array[:, :, 0:21, 0:21, :]
		del X_array_list

		Y_data_list = du.list_all_input_file('./npy/final/one_hour_max_value/')
		Y_data_list.sort()
		Y_array_list = []
		for filename in Y_data_list:
			Y_array_list.append(du.load_array('./npy/final/one_hour_max_value/' + filename))
		Y_array = np.concatenate(Y_array_list, axis=0)
		del Y_array_list
		X_array = feature_scaling(X_array)
		Y_array = feature_scaling(Y_array)
		X_array = X_array[0: -1]  # important!!
		Y_array = Y_array[1:]  # important!! Y should shift 10 minutes
		return X_array, Y_array

	def task_2():
		'''
		rolling 10 minutes among timeflows
		return :
			X_array: past one hour [?, 6, row, col, 1]
			Y_array: next 10 minutes value [?, 1, row, col, 1]

		'''
		X_file_list = du.list_all_input_file('./npy/npy_roll/final/X/')
		Y_file_list = du.list_all_input_file('./npy/npy_roll/final/Y/')
		X_file_list.sort()
		Y_file_list.sort()
		X_array_list = []
		Y_array_list = []
		# X array
		for filename in X_file_list:
			X_array_list.append(du.load_array('./npy/npy_roll/final/X/' + filename))
		X_array = np.concatenate(X_array_list, axis=0)
		del X_array_list
		# Y array
		for filename in Y_file_list:
			Y_array_list.append(du.load_array('./npy/npy_roll/final/Y/' + filename))
		Y_array = np.concatenate(Y_array_list, axis=0)
		del Y_array_list
		X_array = feature_scaling(X_array)
		Y_array = feature_scaling(Y_array)
		return X_array, Y_array

	return task_2()


def feature_scaling(input_datas):
	input_shape = input_datas.shape
	input_datas = input_datas.reshape(input_shape[0], -1)
	min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 255))
	output = min_max_scaler.fit_transform(input_datas)
	output = output.reshape(input_shape)
	return output


if __name__ == '__main__':
	# prepare_training_data()
	X_array, Y_array = get_X_and_Y_array()
	# parameter
	network_parameter = {
		'conv1': 32,
		'conv2': 32,
		'conv3': 32,
		'conv4': 32,
		'conv5': 32,
		'conv6': 32,
		'conv7': 32,
		'fc1': 512}  # hidden layer
	data_shape = [X_array.shape[1], X_array.shape[2], X_array.shape[3], X_array.shape[4]]
	train_CNN = cn.CNN_autoencoder(*data_shape, **network_parameter)
	# train_CNN.reload_tfrecord('./training.tfrecoeds','./testing.tfrecoeds')
	# train_CNN.set_model_name(
		# '/home/mldp/ML_with_bigdata/output_model/CNN_autoencoder_64_64_AE_self.ckpt',
		# '/home/mldp/ML_with_bigdata/output_model/CNN_autoencoder_64_64_AE_self.ckpt')
	'''
		pre train model : roll_10_pre_64_64_64_64_32_32_16_1024.mkpt
	'''	
	model_path = {
		'pretrain_save': '/home/mldp/ML_with_bigdata/output_model/roll_10_pre_64_64_64_64_32_32_16_1024.ckpt',
		'pretrain_reload': '/home/mldp/ML_with_bigdata/output_model/roll_10_pre_64_64_64_64_32_32_16_1024.ckpt',
		'reload': '/home/mldp/ML_with_bigdata/output_model/roll_10_train_32_512_v2.ckpt',
		'save': '/home/mldp/ML_with_bigdata/output_model/roll_10_train_32_512_v2.ckpt'
	}
	train_CNN.set_training_data(X_array, Y_array)
	del X_array, Y_array
	# train_CNN.start_pre_training(model_path, restore=False)
	train_CNN.start_train(model_path, restore=False)
