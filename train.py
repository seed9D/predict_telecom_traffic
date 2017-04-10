import data_utility as du
import numpy as np
import CNN_autoencoder as cn


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
	filelist = du.list_all_input_file('./npy/')
	filelist.sort()
	for i, filename in enumerate(filelist):
		if filename != 'training_raw_data.npy':
			data_array = du.load_array('./npy/' + filename)
			data_array = data_array[:, :, 0:40, 0:40, -1, np.newaxis]  # only network activity
			print('saving  array shape:{}'.format(data_array.shape))
			du.save_array(
				data_array, './npy/final/training_raw_data' + '_' + str(i))


if __name__ == '__main__':
	# prepare_training_data()

	training_data_list = du.list_all_input_file('./npy/final/')
	training_data_list.sort()
	X_array_list = []
	for filename in training_data_list:
		X_array_list.append(du.load_array('./npy/final/' + filename))

	X_array = np.concatenate(X_array_list, axis=0)
	# X_array = X_array[:, :, 0:21, 0:21, :]
	del X_array_list

	# parameter
	network_parameter = {'conv1': 32, 'conv2': 32, 'conv3': 0, 'fc1': 1024, 'fc2': 512}
	data_shape = [X_array.shape[1], X_array.shape[2], X_array.shape[3], X_array.shape[4]]
	train_CNN = cn.CNN_autoencoder(*data_shape, **network_parameter)
	# train_CNN.reload_tfrecord('./training.tfrecoeds','./testing.tfrecoeds')
	# train_CNN.set_model_name(
		# '/home/mldp/ML_with_bigdata/output_model/CNN_autoencoder_64_64_AE_self.ckpt',
		# '/home/mldp/ML_with_bigdata/output_model/CNN_autoencoder_64_64_AE_self.ckpt')
	model_path = {
		'pretrain_save': '/home/mldp/ML_with_bigdata/output_model/AE_pre_32_32_test.ckpt',
		'pretrain_reload': '/home/mldp/ML_with_bigdata/output_model/AE_pre_32_32_test.ckpt',
		'reload': '/home/mldp/ML_with_bigdata/output_model/train_test.ckpt',
		'save': '/home/mldp/ML_with_bigdata/output_model/train_test.ckpt'
	}
	train_CNN.set_training_data(X_array)
	del X_array
	# train_CNN.start_pre_training(model_path, restore=False)
	train_CNN.start_train(model_path, restore=False)
