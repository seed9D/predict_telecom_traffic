import CNN_autoencoder as cn
import matplotlib.pyplot as plt
import numpy as np
import pytz 
from datetime import datetime, tzinfo, timedelta

def set_time_zone(timestamp):
	UTC_timezone = pytz.timezone('UTC')
	Mi_timezone = pytz.timezone('Europe/Rome')
	date_time = datetime.utcfromtimestamp(float(timestamp))
	date_time = date_time.replace(tzinfo = UTC_timezone)
	date_time = date_time.astimezone(Mi_timezone)
	return date_time
def date_time_covert_to_str(date_time):
	return date_time.strftime('%H:%M:%S')

def plot_predict_vs_real(real, predict):
	
	grid_id_1 = real[0,0,10,10,0]
	grid_id_2 = real[0,0,30,20,0]

	plot_1 ={
		'grid_id':int(real[0,0,10,10,0]),
		'predict':[],
		'real':[]
	}
	plot_2 ={
		'grid_id':int(real[0,0,30,20,0]),
		'predict':[],
		'real':[]
	}
	time_x = []
	for i in range(24):
		for j in range(real.shape[1]):
			plot_1['real'].append(real[i,j,10,10,2])
			plot_1['predict'].append(predict[i,j,10,10])
			#print('id:{},real:{},predict:{}'.format(plot_1['grid_id'],real[i,j,10,10,2],predict[i,j,10,10]))
			plot_2['real'].append(real[i,j,30,20,2])
			plot_2['predict'].append(predict[i,j,30,20])
			print('id:{},real:{},predict:{}'.format(plot_2['grid_id'],real[i,j,30,20,2],predict[i,j,30,20]))

			data_time = set_time_zone(int(real[i,j,10,10,1]))
			time_x.append(date_time_covert_to_str(data_time))


	x = range(len(time_x))
	fig = plt.figure()
	ax1 = fig.add_subplot(211)
	ax2 = fig.add_subplot(212)
	
	plt.xlabel('time sequence')
	plt.ylabel('activity strength')

	ax1.plot(x,plot_1['real'],label='real')
	ax1.plot(x,plot_1['predict'],label='predict')
	ax1.set_xticks(x,time_x)
	ax1.set_title(plot_1['grid_id'])
	#ax1.set_ylabel('time sequence')
	#ax1.set_xlabel('activity strength')
	ax1.legend()

	ax2.plot(x,plot_2['real'],label='real')
	ax2.plot(x,plot_2['predict'],label='predict')
	ax2.set_xticks(x,time_x)
	ax2.set_title(plot_2['grid_id'])
	#ax2.set_ylabel('time sequence')
	#ax2.set_xlabel('activity strength')
	ax2.legend()
	
	plt.show()





'''
input_dir_list = [
		#"/home/mldp/big_data/openbigdata/milano/SMS/11/data_preproccessing_10/",
		"/home/mldp/big_data/openbigdata/milano/SMS/12/data_preproccessing_10/"
		]

X_array = None
for input_dir in input_dir_list:
	filelist =cn.list_all_input_file(input_dir)
	filelist.sort()
	temp = cn.load_data_format(input_dir,filelist)
	try:
		X_array = np.concatenate((X_array,temp),axis=0)
	except:
		X_array = temp

X_array = X_array[:,:,30:70,30:70,:]
X_array = X_array[X_array.shape[0]-200:,:,:,:,(0,1,-1)]
print(X_array.shape)

cn.save_array(X_array,'./testing_raw_data')

'''
X_array = cn.load_array('./proccessed_raw_data')
predict_array = cn.load_array('./testing_raw_data')


network_parameter = {'conv1': 48, 'conv2': 32, 'conv3':16}
data_shape = [X_array.shape[1],X_array.shape[2],X_array.shape[3],X_array.shape[4]]

predict_CNN = cn.CNN_autoencoder(*data_shape, **network_parameter)
predict_CNN.set_model_name('/home/mldp/ML_with_bigdata/output_model/CNN_autoencoder_onlyinternet_48_32_16.ckpt','/home/mldp/ML_with_bigdata/output_model/CNN_autoencoder_onlyinternet_48_32_16.ckpt')
predict_CNN.set_training_data(X_array)
del X_array
_ , predict_y = predict_CNN.predict_data(predict_array[:,:,:,:,2,np.newaxis])

plot_predict_vs_real(predict_array[0:-1],predict_y)



