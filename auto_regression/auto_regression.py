import pandas as pd
from pandas.tools.plotting import lag_plot, autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import sys
import pytz
from datetime import datetime
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


def plot_lag():
	pass


def plot_auto_correlation():
	pass


def plot_ACF():
	pass


data_frame = prepare_data()

print(len(data_frame))
data = data_frame['internet'].values
train, test = data[1:len(data) - 144], data[len(data) - 144:]
# data_frame.plot(x='date', y='internet')
# lag_plot(data_frame['internet'])
# autocorrelation_plot(data_frame['internet'])
# plot_acf(data_frame['internet'], lags=31)
# plt.show()
model = AR(train)
model_fit = model.fit()
slide_windows = model_fit.k_ar
coef = model_fit.params
print('lag: {}'.format(slide_windows))
print('coefficients: {}'.format(coef))
# make predcitions
# predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)


# walk forward over time step
history = train[len(train) - slide_windows:]  # numpy array
history = [history[i] for i in range(len(history))]  # convert to list

predictions = list()
for t in range(len(test)):
	length = len(history)
	lag = [history[i] for i in range(length - slide_windows, length)]
	yhat = coef[0]
	for d in range(slide_windows):
		yhat += coef[d + 1] * lag[slide_windows - d - 1]  # yhat = b0 + b1*X1 + b2*X2 ... bn*Xn
		# coef[0] is constant
	obs = test[t]
	predictions.append(yhat)
	history.append(obs)
	print('predict:{} expected:{}'.format(yhat, obs))


error = mean_squared_error(test, predictions)
print('Test MSE:{}'.format(error))
'''
for i in range(len(predictions)):
	print('predicted=%f, expected=%f' % (predictions[i], test[i]))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
'''

# plot results
plt.plot(test, label='test data real value', marker='.')
plt.plot(predictions, color='red', label='test data predcit value', marker='.')
plt.legend()
plt.show()

