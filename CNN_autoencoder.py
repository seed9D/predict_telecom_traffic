import pickle
import numpy as np
import os
import tensorflow as tf
import functools as fn
from glob import glob
from sys import exit
from math import sqrt
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
				#L2 regularization
				L2 = 0
				for key, value in self.weights.items():
					L2 += tf.nn.l2_loss(value)
				
				loss += tf.reduce_mean(L2*self.weight_decay)
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
			#deconv2 = tf.nn.relu(deconv2)

			endecoder_output = deconv2
			print('endecoder_output output shape :%s' %
				  endecoder_output.get_shape())

			return encode_output, endecoder_output

		def weight_variable( shape, name):
			#initial = tf.truncated_normal(shape, stddev=0.1)
			initial = np.random.randn(*shape)*sqrt(2.0/np.prod(shape))
			return tf.Variable(initial, dtype=tf.float32, name=name)

		def bias_variable( shape, name):
			#initial = tf.random_normal(shape)
			initial = np.random.randn(*shape)*sqrt(2.0/np.prod(shape))
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
		def batch_normalize(x,scope,norm=0):
			axes =  list(range(len(x.get_shape())-1))
			decay = 0.999
			with tf.variable_scope(scope):
				pop_mean = tf.get_variable('pop_mean',[x.get_shape()[-1]],
					trainable=False,
					dtype=tf.float32,
					initializer = tf.constant_initializer(0)
					)
				pop_var = tf.get_variable('pop_var',[x.get_shape()[-1]],
					trainable=False,
					dtype=tf.float32,
					initializer = tf.constant_initializer(1)
				)
				#scale = tf.get_variable(tf.ones([x.get_shape()[-1]]),name='scale')
				scale=tf.get_variable('scale',[x.get_shape()[-1]],dtype=tf.float32,initializer=tf.constant_initializer(1))
				#beta = tf.get_variable(tf.zeros([x.get_shape()[-1]]),name='beta')
				beta=tf.get_variable('beta',[x.get_shape()[-1]],dtype=tf.float32,initializer=tf.constant_initializer(0))
			if norm is 1:
				batch_mean, batch_var = tf.nn.moments(x,axes=axes)
				train_mean = tf.assign(pop_mean,
						pop_mean * decay + batch_mean * (1 - decay)
					)
				train_var = tf.assign(pop_var,
						pop_var * dacay + batch_var * (1 - decay) 
					)
				with tf.control_dependencies([train_mean,train_var]):
					return tf.nn.batch_normalization(x,batch_mean,batch_var,beta,scale,1e-3)
			else:
				return tf.nn.batch_normalization(x,pop_mean,pop_var,beta,scale,1e-3)


		self.learning_rate = 0.001
		self.training_iters = 20000
		self.batch_size = 100
		self.display_step = 50
		self.dropout = 0.6
		self.shuffle_capacity = 700
		self.shuffle_min_after_dequeue = 500
		self.weight_decay = 0.01 
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
			

			# variable, control filter size
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
			#self.mean = tf.Variable(tf.zeros([self.input_channel]),name='mean',trainable=False,)
			#self.std = tf.Variable(tf.zeros([self.input_channel]),name='std',trainable=False,)
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
		input_x,self.mean,self.std = self.feature_normalize_input_data(input_x)
		
		X_data = input_x[0:-1]
		Y_data = input_x[1:]
		
		
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
	def un_normalize_data(self,sess,input_x):
		#mean,std = sess.run([self.mean,self.std])
		#print('mean:{},std:{}'.format(self.mean,self.std))
		input_x  = input_x*self.std+self.mean
		return input_x
	def reload_tfrecord(self,tfrecored_file,tftesting_file):
		if not os.path.isfile(tfrecored_file):
			print('{} not exists'.format(tfrecored_file))
		else:
			self.training_file = tfrecored_file
			self.testing_file = tftesting_file

	def predict_data(self,input_x):
		testing_x = input_x[0:-1]
		testing_y = input_x[1:]
		with tf.Session() as sess:
			self._reload_model(sess)

			#mean,std = sess.run([self.mean,self.std])
			testing_x = (testing_x-self.mean)/self.std
			
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
	def _read_all_data_from_Tfreoced(self,filename):
		record_iterator = tf.python_io.tf_record_iterator(path=filename)
		record_list = []
		result_list = []
		for string_record in record_iterator:
			example = tf.train.Example()
			example.ParseFromString(string_record)
			index = example.features.feature['index'].int64_list.value[0]
			record = example.features.feature['record'].bytes_list.value[0]
			result = example.features.feature['result'].bytes_list.value[0]
			record = np.fromstring(record,dtype=np.float32)
			record = record.reshape((self.input_temporal,
				self.input_vertical,
				self.input_horizontal,
				self.input_channel))
			
			result = np.fromstring(record,dtype=np.float32)
			result = record.reshape((self.input_temporal,
				self.input_vertical,
				self.input_horizontal,
				self.input_channel))
			record_list.append(record)
			result_list.append(result)

		record = np.stack(record_list)
		result = np.stack(result_list)
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
				predict = self.un_normalize_data(sess,predict)
				un_normalize_input_y = self.un_normalize_data(sess,input_y[batch_index*self.batch_size:(batch_index+1)*self.batch_size])
				print('predict:{},real:{}'.format(predict[0,0,1,1,0],un_normalize_input_y[0,0,1,1,0]))
				print('predict:{},real:{}'.format(predict[0,0,10,20,0],un_normalize_input_y[0,0,10,20,0]))
				print('predict:{},real:{}'.format(predict[0,0,20,20,0],un_normalize_input_y[0,0,20,20,0]))
				
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
					index, testing_X,testing_Y = self._read_all_data_from_Tfreoced(self.testing_file)
					testing_loss,_ = self._testing_data(sess,testing_X,testing_Y)
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

def save_array(x_array,out_file):
	print('saving file to {}...'.format(out_file))
	np.save(out_file,x_array, allow_pickle=True)
def load_array(input_file):
	print('loading file from {}...'.format(input_file))
	X = np.load(input_file+'.npy')
	return X
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
	'''
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
	

	X_array = X_array[:,:,30:70,30:70,-1,np.newaxis]

	
	
	
	
	save_array(X_array,'./proccessed_raw_data')
	'''
	
	X_array = load_array('./proccessed_raw_data')
	print(np.amax(X_array),np.argmax(X_array) )
	network_parameter = {'conv1': 16, 'conv2': 32}
	data_shape = [X_array.shape[1],X_array.shape[2],X_array.shape[3],X_array.shape[4]]
	train_CNN = CNN_autoencoder(*data_shape, **network_parameter)
	#train_CNN.reload_tfrecord('./training.tfrecoeds','./testing.tfrecoeds')
	train_CNN.set_model_name('/home/mldp/ML_with_bigdata/output_model/CNN_autoencoder_onlyinternet.ckpt','/home/mldp/ML_with_bigdata/output_model/CNN_autoencoder_onlyinternet.ckpt')
	train_CNN.set_training_data(X_array)
	del X_array
	train_CNN.training_data(restore=False)
	#train_CNN.predict_data(X_array[int(9*X_array.shape[0]/10):])
	
