import numpy as np
import os


def list_all_input_file(input_dir):
	onlyfile = [f for f in os.listdir(input_dir) if (os.path.isfile(
		os.path.join(input_dir, f)) and os.path.splitext(f)[1] == ".npy")]
	return onlyfile


def save_array(x_array, out_file):
	print('saving file to {}...'.format(out_file))
	np.save(out_file, x_array, allow_pickle=True)


def load_array(input_file):
	print('loading file from {}...'.format(input_file))
	X = np.load(input_file)
	return X


def load_data_format(input_dir, filelist):
	def load_array(input_file):
		print('loading file from {}...'.format(input_file))
		X = np.load(input_file)
		return X.astype(np.float32)

	def split_array(data_array):
		# print('data_array shape :', data_array.shape)
		split_block_size = 6  # one hour
		data_array_depth = data_array.shape[0]
		remainder = data_array_depth % split_block_size
		split_block_num = int(data_array_depth / split_block_size)

		# new_data_array_size = [split_block_num,data_array.shape[1:]]
		# print('new_data_array_size:',new_data_array_size)

		split_data_list = np.split(
			data_array[:data_array_depth - remainder], split_block_num)
		new_data_array = np.stack(split_data_list, axis=0)
		# print('new_data_array shape:', new_data_array.shape)

		return new_data_array

	def shift_data(data_array):
		'''
			generate more data
		'''
		array_list = []
		for shift_index in range(3):
			shift_array = data_array[shift_index:]
			array_list.append(split_array(shift_array))

		return array_list

	def array_concatenate(x, y):  # for reduce
		return np.concatenate((x, y), axis=0)

	month, _ = os.path.split(input_dir)
	month = month.split('/')[-2]

	data_array_list = []
	for file_name in filelist:
		data_array_list.append(load_array(input_dir + file_name))
	data_array = np.concatenate(data_array_list, axis=0)
	del data_array_list
	shift_data_array_list = shift_data(data_array)
	for i, array in enumerate(shift_data_array_list):
		print('data format shape:', array.shape)
		save_array(array, './npy/' + month + '_' + str(i))  # saving all shift array
		