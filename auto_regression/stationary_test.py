import pandas as pd
import matplotlib.pyplot as plt
import sys
import pytz
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
sys.path.append('/home/mldp/ML_with_bigdata')
# import .data_utility as du
import data_utility as du

input_file = '/home/mldp/ML_with_bigdata/npy/11_0.npy'


def set_time_zone(timestamp):

	UTC_timezone = pytz.timezone('UTC')
	Mi_timezone = pytz.timezone('Europe/Rome')
	date_time = datetime.utcfromtimestamp(float(timestamp))
	date_time = date_time.replace(tzinfo=UTC_timezone)
	date_time = date_time.astimezone(Mi_timezone)
	return date_time


def date_time_covert_to_str(date_time):
	return date_time.strftime('%Y-%m-%d %H:%M')


def prepare_data():
	data_array = du.load_array(input_file)
	print('saving array shape:{}'.format(data_array.shape))
	# du.save_array(data_array, './npy/autoregrssion_raw_data')

	i_len = data_array.shape[0]
	j_len = data_array.shape[1]
	row = 50
	col = 50

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


def split_and_test_var_mean(data):
	 # data_internet.hist()
	split = len(data) // 2
	X1, X2 = data_internet[:split], data_internet[split:]
	mean1, mean2 = X1.mean(), X2.mean()
	var1, var2 = X1.var(), X2.var()
	print('mean1:{} mean2:{}'.format(mean1, mean2))
	print('variance1:{} variance2:{}'.format(var1, var2))


def plot_histogram(data):
	data.hist()
	plt.title('histogram of network traffic')
	plt.show()


def plot_KDE(data):
	data.plot(kind='kde')
	plt.title('KDE plot')
	plt.show()


def augmented_dickey_fuller_test(data):
	data = data.values
	result = adfuller(data)
	print('ADF stattistic:{}'.format(result[0]))
	print('p-value:{}'.format(result[1]))
	print('critical value:')
	for key, value in result[4].items():
		print('\t{}: {}'.format(key, value))


data = prepare_data()
data_internet = data['internet']
# plot_histogram(data_internet)
# plot_KDE(data_internet)
# split_and_test_var_mean(data_internet)
augmented_dickey_fuller_test(data_internet)
