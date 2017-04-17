import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import sys
import pytz
from datetime import datetime
from math import sqrt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
sys.path.append('/home/mldp/ML_with_bigdata')
# import .data_utility as du
import data_utility as du
import numpy as np

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


def predict(coef, history):
	yhat = 0.0
	for i in range(1, len(coef) + 1):
		yhat += coef[i - 1] * history[-i]
	return yhat


def differenc(dataset):
	diff = []
	for i in range(1, len(dataset)):
		value = dataset[i] - dataset[i - 1]
		diff.append(value)
	return np.array(diff)


def AR_model(train, test):
	history = [x for x in train]  # convert from numpy array to list
	predictions = []
	for t in range(len(test)):
		model = ARIMA(history, order=(6, 0, 0))
		model_fit = model.fit(trend='nc', disp=False)
		ar_coef = model_fit.arparams
		# ma_coef = model_fit.maparams
		# resid = model_fit.resid
		# print(resid)
		yhat = predict(ar_coef, history)
		predictions.append(yhat)
		obs = test[t]
		history.append(obs)
		print('predcition:{} expected:{}'.format(yhat, obs))
	rmse = sqrt(mean_squared_error(test, predictions))
	print('testing data RMSE: {}'.format(rmse))

	plt.plot(test, label='test data real value', marker='.')
	plt.plot(predictions, color='red', label='test data predic value', marker='.')
	plt.legend()
	plt.show()
	calculate_residual(predictions, test)

def MA_model(train, test):
	history = [x for x in train]  # convert from numpy array to list
	predictions = []
	for t in range(len(test)):
		model = ARIMA(history, order=(0, 0, 4))
		model_fit = model.fit(trend='nc', disp=False)
		# ar_coef = model_fit.arparams
		ma_coef = model_fit.maparams
		resid = model_fit.resid
		print(resid)
		yhat = predict(ma_coef, resid)
		predictions.append(yhat)
		obs = test[t]
		history.append(obs)
		print('predcition:{} expected:{}'.format(yhat, obs))
	rmse = sqrt(mean_squared_error(test, predictions))
	print('testing data RMSE: {}'.format(rmse))

	plt.plot(test, label='test data real value', marker='.')
	plt.plot(predictions, color='red', label='test data predic value', marker='.')
	plt.legend()
	plt.show()
	calculate_residual(predictions, test)


def AR_MA_model(train, test):
	history = [x for x in train]
	predictions = []
	predict_step = 6
	for t in range(len(test) // predict_step):
		model = ARIMA(history, order=(6, 0, 1))
		model_fit = model.fit(disp=False)
		yhat = model_fit.predict(start=len(history), end=len(history) + predict_step - 1, dynamic=False)
		'''
		ac_coef, ma_coef = model_fit.arparams, model_fit.maparams
		resid = model_fit.resid
		yhat = predict(ac_coef, history) + predict(ma_coef, resid)
		'''
		predictions.extend(yhat)
		obs = test[t * predict_step: (t + 1) * predict_step]
		history.extend(obs)
		print('{}: predcition:{} \n expected:{}'.format(t, yhat, obs))
	rmse = sqrt(mean_squared_error(test, predictions))
	print('testing data RMSE: {}'.format(rmse))
	plt.plot(test, label='test data real value', marker='.')
	plt.plot(predictions, color='red', label='test data predic value', marker='.')
	plt.legend()
	plt.show()
	calculate_residual(predictions, test)


def ARIMA_model(train, test):
	history = [x for x in train]
	predictions = []
	predict_step = 6
	for t in range(len(test)):
		model = ARIMA(history, order=(6, 1, 1))
		model_fit = model.fit(disp=False)
		'''
		yhat = model_fit.predict(start=len(history), end=len(history) + predict_step - 1, dynamic=False)
		predictions.extend(yhat)
		obs = test[t * predict_step: (t + 1) * predict_step]
		history.extend(obs)
		'''
		ac_coef, ma_coef = model_fit.arparams, model_fit.maparams
		# print(ma_coef)
		resid = model_fit.resid
		diff = differenc(history)
		yhat = history[-1] + predict(ac_coef, diff) + predict(ma_coef, resid)
		predictions.append(yhat)
		obs = test[t * predict_step: (t + 1) * predict_step]
		history.append(obs)
		print('{} predcition:{} expected:{}'.format(t, yhat, obs))
	rmse = sqrt(mean_squared_error(test, predictions))
	print('testing data RMSE: {}'.format(rmse))
	plt.plot(test, label='test data real value', marker='.')
	plt.plot(predictions, color='red', label='test data predic value', marker='.')
	plt.legend()
	plt.show()
	calculate_residual(predictions, test)


def plot_acf_and_pacf(data):
	plt.figure()
	plt.subplot(211)
	plot_acf(data, ax=plt.gca(), lags=31)
	plt.subplot(212)
	plot_pacf(data, ax=plt.gca(), lags=31)
	plt.show()


def calculate_residual(predictions, real):
	residuals = [real[i] - predictions[i] for i in range(len(real))]
	residuals = pd.DataFrame(residuals)
	print(residuals.describe())

	plt.figure()
	plt.subplot(211)
	residuals.hist(ax=plt.gca())
	plt.subplot(212)
	residuals.plot(kind='kde', ax=plt.gca(), label='Resudual error')
	plt.legend()
	plt.title('Residual Forecast Error Plots')
	plt.show()


data = prepare_data()
# plot_acf_and_pacf(data['internet'])
data_internet = data['internet'].values
data_internet_len = len(data_internet)
train, test = data_internet[:int(data_internet_len - 144)], data_internet[int(data_internet_len - 144):]

# AR_model(train, test)
# MA_model(train, test)
# AR_MA_model(train, test)
ARIMA_model(train, test)