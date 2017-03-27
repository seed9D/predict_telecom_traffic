import pickle
import numpy as np
import os
import tensorflow as tf
import functools as fn
from glob import glob
from sys import exit
'''
todo:
	normalization
	batch_normalize

'''

class CNN_autoencoder:
	def __init__(self, *data_shape, **network_para):
		def MSE_loss(self):
			with tf.variable_scope('loss'):
				#print('endecoder_OP shape:{}, Ys shape{}'.format(self.endecoder_OP[:,5].get_shape(),self.Ys.get_shape()))
				loss = tf.reduce_mean(tf.pow(self.endecoder_OP - self.Ys, 2))
				loss = tf.div(loss, 2)
			return loss

		def net_layer(self,x, weights, bias, dropout, norm=0):

			k_size = {'temporal': 2, 'vertical': 2, 'horizontal': 2}
			strides_size = {'temporal': 2, 'vertical': 2, 'horizontal': 2}
			# layer 1
			conv1 = conv3d(x, weights['conv1'], bias['conv1'])
			conv1 = tf.nn.relu(conv1)
			# conv1 = self.maxpool3d(conv1,**k_size,**strides_size)
			conv1 = tf.nn.dropout(conv1, dropout)

			# layer 2
			conv2 = conv3d(conv1, weights['conv2'], bias['conv2'])
			conv2 = tf.nn.relu(conv2)
			# conv2 = self.maxpool3d(conv2,**k_size,**strides_size)
			conv2 = tf.nn.dropout(conv2, dropout)

			encode_output = conv2
			print('encode layer shape:%s' % encode_output.get_shape())
			# layer 3
			output_shape_of_dconv1 = tf.pack([tf.shape(x)[0],
											  self.input_temporal,
											  self.input_vertical,
											  self.input_horizontal,
											  self.conv1])
			deconv1 = deconv3d(conv2, weights['deconv1'], bias[
									'deconv1'], output_shape_of_dconv1)
			deconv1 = tf.nn.relu(deconv1)
			# output_shape_of_dconv1_unpool = tf.pack([tf.shape(x)[0],
			#	self.input_temporal/(2**(self.pooling_times-1)),
			#	self.input_vertical/(2**(self.pooling_times-1)),
			#	self.input_horizontal/(2**(self.pooling_times-1)),
			#	self.conv2])
			# deconv1 = self.maxpool3d(deconv1,output_shape_of_dconv1_unpool)

			# layer 4
			output_shape_of_dconv2 = tf.pack([tf.shape(x)[0],
											  self.input_temporal,
											  self.input_vertical,
											  self.input_horizontal,
											  self.input_channel])
			deconv2 = deconv3d(deconv1, weights['deconv2'], bias[
									'deconv2'], output_shape_of_dconv2)
			deconv2 = tf.nn.relu(deconv2)

			endecoder_output = deconv2
			print('endecoder_output output shape :%s' %
				  endecoder_output.get_shape())

			return encode_output, endecoder_output

		def weight_variable( shape, name):
			initial = tf.truncated_normal(shape, stddev=0.1)
			return tf.Variable(initial, dtype=tf.float32, name=name)

		def bias_variable( shape, name):
			initial = tf.random_normal(shape)
			return tf.Variable(initial, dtype=tf.float32, name=name)

		def conv3d( x, W, b, strides=1):
			x = tf.nn.conv3d(
				x, W, strides=[1, strides, strides, strides, 1], padding='SAME')
			x = tf.nn.bias_add(x, b)
			return x

		def deconv3d(x, W, b, output_shape, strides=1):
			'''
					filter shape:[depth, height, width, output_channels, in_channels]
			'''
			print('input shape:{} filter shape:{} output_shape:{}'.format(
				x.get_shape(), W.get_shape(), output_shape.get_shape()))
			x = tf.nn.conv3d_transpose(x, W, output_shape, strides=[
									   1, strides, strides, strides, 1], padding='SAME')
			x = tf.nn.bias_add(x, b)
			return x

		def maxpool3d( x, k_size, strides_size):
			x = tf.nn.max_pool3d(x, k=[1, k['temporal'], k['vertical'], k['horizontal'], 1], strides=[
								 1, strides_size['temporal'], strides_size['vertical'], strides_size['horizontal'], 1], padding='SAME')
			pooling_times += 1
			return x

		def maxunpool3d( x, shape):
			pooling_times -= 1
			return x
		self.learning_rate = 0.0007
		self.training_iters = 20000
		self.batch_size = 35
		self.display_step = 50
		self.dropout = 0.7
		self.shuffle_capacity = 700
		self.shuffle_min_after_dequeue = 500
		# self.n_input = 100*100

		# network parameter
		self.conv1 = network_para.get('conv1')
		self.conv2 = network_para.get('conv2')
		self.pooling_times = 0
		# self.deconv1 = network_para.get('deconv1')
		# self.deconv2 = network_para.get('deconv2')
		
		
		self.input_temporal = data_shape[0]
		self.input_vertical = data_shape[1]
		self.input_horizontal = data_shape[2]
		self.input_channel = data_shape[3]
		# placeholder
		with tf.device('/cpu:0'):
			self.Xs = tf.placeholder(tf.float32, shape=[
									 None, self.input_temporal, self.input_vertical, self.input_horizontal, self.input_channel])
			self.Ys = tf.placeholder(tf.float32, shape=[
									 None,  self.input_temporal, self.input_vertical, self.input_horizontal, self.input_channel])
			self.keep_prob = tf.placeholder(tf.float32)
			self.norm = tf.placeholder(tf.bool, name='norm')
			

			# variable control filter size
			self.weights = {
				'conv1': weight_variable([3, 3, 3, self.input_channel, self.conv1], 'conv1_w'),
				'conv2': weight_variable([3, 3, 3, self.conv1, self.conv2], 'conv2_w'),
				'deconv1': weight_variable([3, 3, 3, self.conv1, self.conv2], 'deconv1_w'),
				'deconv2': weight_variable([3, 3, 3, self.input_channel, self.conv1], 'deconv1_w')
			}
			self.bias = {
				'conv1': bias_variable([self.conv1], 'conv1_b'),
				'conv2': bias_variable([self.conv2], 'conv2_b'),
				'deconv1': bias_variable([self.conv1], 'deconv1_b'),
				'deconv2': bias_variable([self.input_channel], 'deconv2_b')
			}
			self.mean = tf.Variable(0,name='mean')
			self.std = tf.Variable(0,name='std')
		# operation
	
		self.encoder_OP, self.endecoder_OP = net_layer(self,
			self.Xs, self.weights, self.bias, self.keep_prob, self.norm)
		self.cost_OP = MSE_loss(self)
		
		self.optimizer_OP = tf.train.AdamOptimizer(
			learning_rate=self.learning_rate).minimize(self.cost_OP)

		self.init_OP = tf.global_variables_initializer()
		self.saver = tf.train.Saver()

	
	def set_training_data(self,input_x):
		
		print('input_x shape:{}'.format(input_x.shape))

		X_data = input_x[0:-1]
		Y_data = input_x[1:]
		X_data,mean,std = self.feature_normalize_input_data(X_data)
		
		#Y_data = Y_data[:,np.newaxis]
		#print(X_data[1,0,0,0,-1],Y_data[0,0,0,0,-1])
		training_X = X_data[0:int(9*X_data.shape[0]/10)]
		training_Y = Y_data[0:int(9*X_data.shape[0]/10)]
		self.testing_X =  X_data[int(9*X_data.shape[0]/10):]
		self.testing_Y =  Y_data[int(9*X_data.shape[0]/10):]
		self.training_file = 'training.tfrecoeds'
		self.testing_file = 'testing.tfrecoeds'
		print('training X shape:{}, training Y shape:{}'.format(training_X.shape,training_Y.shape))
		self._write_to_Tfrecord(training_X ,training_Y,self.training_file)
		self._write_to_Tfrecord(self.testing_X ,self.testing_Y,self.testing_file)
		self.training_data_number = training_X.shape[0]

	def feature_normalize_input_data(self,input_x):
			input_shape = input_x.shape

			axis = tuple(range(len(input_shape)-1))
			mean  = np.mean(input_x,axis=axis)
			std = np.std(input_x,axis=axis)
			print('mean:{},std:{}'.format(mean,std))
			X_normalize =(input_x-mean)/std
			return X_normalize,mean,std
	def reload_tfrecord(self,tfrecored_file):
		if not os.path.isfile(tfrecored_file):
			print('{} not exists'.format(tfrecored_file))
		else:
			self.training_file = tfrecored_file


	def predict_data(self,input_x):
		testing_x = input_x[0:-1]
		testing_y = input_x[1:]
		with tf.Session() as sess:
			self._reload_model(sess)

			mean,std = sess.run([self.mean,self.std])
			testing_x = (testing_x-mean)/std
			
			input_x_length = input_x.shape[0]
			_, predict_list = _testing_data(sess,testing_x,testing_y)
			print('predict finished!')
			print('predict_list shape{}'.format(predict_list))

	
	def set_model_name(self,reload_model_path,save_model_path):
		
		if not glob(reload_model_path+'.*'):
			print('{} not exists'.format(reload_model_path))
			exit(1)
		else:
			self.model_path = reload_model_path

		output_dir = os.path.dirname(save_model_path)
		if not os.path.isdir(output_dir):
			os,makedirs(output_dir)
		self.save_model = save_model_path


	def _reload_model(self,sess):
		print('reloading model {}.....'.format(self.model_path))

		try:
			self.saver.restore(sess,self.model_path)
		except:
			print('model {} doesnt exists'.format(self.model_path))
			sess.run(self.init)
	def _save_model(self,sess):
		print('saving model.....')
		if not os.path.isdir('./output_model'):
			os.makedirs('./output_model')
		try:
				save_path = self.saver.save(sess, self.save_model)
		except:
				save_path = self.saver.save(sess, './output_model/temp.ckpt')
		finally:
			print('save_path{}'.format(save_path))
	def _write_to_Tfrecord(self,X_array,Y_array,filename):
		writer = tf.python_io.TFRecordWriter(filename)
		for index, each_record in enumerate(X_array):
			tensor_record = each_record.astype(np.float32).tobytes()
			tensor_result = Y_array[index].astype(np.float32).tobytes()
			#print('in _write_to_Tfrecord',X_array.shape,Y_array.shape)
			example = tf.train.Example(features = tf.train.Features (feature = {
					'index': tf.train.Feature(int64_list = tf.train.Int64List(value =[index])),
					'record':tf.train.Feature(bytes_list = tf.train.BytesList(value = [tensor_record])),
					'result':tf.train.Feature(bytes_list = tf.train.BytesList(value = [tensor_result]))
				}))
			
			writer.write(example.SerializeToString())
		writer.close()
	def _read_data_from_Tfrecord(self,filename):
		filename_queue = tf.train.string_input_producer([filename])
		reader = tf.TFRecordReader()
		_,serialized_example = reader.read(filename_queue)
		features = tf.parse_single_example(
		serialized_example,
		features ={
			'index': tf.FixedLenFeature([],tf.int64),
			'record': tf.FixedLenFeature([],tf.string),
			'result': tf.FixedLenFeature([],tf.string)
		})
		index = features['index']
		record = tf.decode_raw(features['record'],tf.float32)
		result = tf.decode_raw(features['result'],tf.float32)

		record = tf.reshape(record,[self.input_temporal,
			self.input_vertical,
			self.input_horizontal,
			self.input_channel])
		result = tf.reshape(result,[self.input_temporal,
			self.input_vertical,
			self.input_horizontal,
			self.input_channel])

		return index,record,result
	def _testing_data(self,sess,input_x,input_y):
		batch_num = int(input_x.shape[0]/self.batch_size)
		testing_data_number = input_y.shape[0]
		if  testing_data_number % self.batch_size is not 0:
			batch_len = batch_num + 1
		else:
			batch_len = batch_num

		with tf.device('/gpu:0'):
			predict_list =[]
			cum_loss = 0
			for batch_index in range(batch_len):
				
				loss,predict = sess.run([self.cost_OP,self.endecoder_OP],feed_dict={
						self.Xs:input_x[batch_index*self.batch_size:(batch_index+1)*self.batch_size],
						self.Ys:input_y[batch_index*self.batch_size:(batch_index+1)*self.batch_size],
						self.keep_prob:1,
						self.norm:0
				})
				print('predict:{},real:{}'.format(predict[0,0,50,50,0],input_y[batch_index*self.batch_size,0,50,50,0]))
				print('predict:{},real:{}'.format(predict[0,0,20,20,0],input_y[batch_index*self.batch_size,0,20,20,0]))
				print('predict:{},real:{}'.format(predict[0,0,70,70,0],input_y[batch_index*self.batch_size,0,70,70,0]))
				
				for predict_element in predict:
					predict_list.append(predict.tolist)
				cum_loss += loss
			return loss/batch_num, predict_list
	def training_data(self,restore=False):

		data = self._read_data_from_Tfrecord(self.training_file)
		batch_tuple_OP = tf.train.shuffle_batch( data,
			batch_size =self.batch_size,
			capacity = self.shuffle_capacity,
			min_after_dequeue = self.shuffle_min_after_dequeue
		)
		
		
		with tf.Session() as sess:
			if restore:
				self._reload_model(sess)
			else:
				sess.run(self.init_OP)

			coord = tf.train.Coordinator()
			treads = tf.train.start_queue_runners(sess=sess,coord=coord)
		
			epoch = 1
			cumulate_loss =0

			while epoch < self.training_iters:
				index, batch_x,batch_y = sess.run(batch_tuple_OP)
				#print('index:{}'.format(index))
				loss = 0.

				with tf.device('/gpu:0'):
					_,loss = sess.run([self.optimizer_OP,self.cost_OP], feed_dict={
									   self.Xs: batch_x,
									   self.Ys: batch_y, 
									   self.keep_prob: self.dropout, 
									   self.norm: 1})
			
				print('Epoch:%d  cost:%g'%(epoch,loss))
				cumulate_loss += loss
				if epoch % self.display_step == 0 and epoch != 0:
					average_training_loss = cumulate_loss / self.display_step
					testing_loss,_ = self._testing_data(sess,self.testing_X,self.testing_Y)
					print('testing_loss:{} average_training_loss:{}'.format(testing_loss,average_training_loss))
					cumulate_loss = 0
					self._save_model(sess)

				epoch += 1
			coord.request_stop()
			coord.join(treads)
			print('training finished!')
			_save_model(sess)
def list_all_input_file(input_dir):
	onlyfile = [f for f in os.listdir(input_dir) if (os.path.isfile(
		os.path.join(input_dir, f)) and os.path.splitext(f)[1] == ".npy")]
	return onlyfile


def load_data_format(filelist):
	def load_array(input_file):
		print('loading file from {}...'.format(input_file))
		X = np.load(input_file)
		return X

	def split_array(data_array):
		#print('data_array shape :', data_array.shape)
		split_block_size = 6  # one hour
		data_array_depth = data_array.shape[0]
		split_block_num = int(data_array_depth / split_block_size)

		# new_data_array_size = [split_block_num,data_array.shape[1:]]
		# print('new_data_array_size:',new_data_array_size)

		split_data_list = np.split(data_array, split_block_num)
		new_data_array = np.stack(split_data_list, axis=0)
		#print('new_data_array shape:', new_data_array.shape)

		return new_data_array

	def array_concatenate(x, y):
		return np.concatenate((x, y), axis=0)

	array_list = []
	for file_name in filelist:
		data_array = load_array(input_dir + file_name)
		array_list.append(split_array(data_array))

	X = fn.reduce(array_concatenate, array_list)
	print('data format shape:', X.shape)
	return X

if __name__ == '__main__':
	
	input_dir_list = [
		"/home/mldp/big_data/openbigdata/milano/SMS/11/data_preproccessing_10/",
		"/home/mldp/big_data/openbigdata/milano/SMS/12/data_preproccessing_10/"
		]

	
	
	X_array = None
	for input_dir in input_dir_list:
		filelist = list_all_input_file(input_dir)
		filelist.sort()
		temp = load_data_format(filelist)
		try:
			X_array = np.concatenate((X_array,temp),axis=0)
		except:
			X_array = temp
	
	X_array = X_array[:,:,:,:,-1,np.newaxis]
	
	network_parameter = {'conv1': 16, 'conv2': 32}
	data_shape = [6,100,100,1]
	train_CNN = CNN_autoencoder(*data_shape, **network_parameter)
	#train_CNN.reload_tfrecord('./training.tfrecoeds')
	train_CNN.set_model_name('/home/mldp/ML_with_bigdata/output_model/CNN_autoencoder_onlyinternet.ckpt','/home/mldp/ML_with_bigdata/output_model/CNN_autoencoder_onlyinternet.ckpt')
	train_CNN.set_training_data(X_array)
	del X_array
	#train_CNN.predict_data(X_array[int(9*X_array.shape[0]/10):])
	train_CNN.training_data(restore=False)
