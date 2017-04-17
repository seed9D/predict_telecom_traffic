import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
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


data = prepare_data()
data_internet = data['internet'].values
train, test = data_internet[:len(data_internet) - 144], data_internet[len(data_internet) - 144:]
history = [x for x in train]
# model = ARIMA(history, order=(31, 0, 0))
# model_fit = model.fit(disp=0)
# predictions = model_fit.forecast(steps=144)
# predictions = predictions[0]
predictions = list()

for t in range(len(test)):
	model = ARIMA(history, order=(0, 0, 5))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predict:{} expected:{}'.format(yhat, obs))

error = mean_squared_error(test, predictions)
print('MSE {}'.format(error))

plt.plot(test, label='test data real value', marker='.')
plt.plot(predictions, color='red', label='test data predic value', marker='.')
plt.legend()
plt.show()

'''
model = ARIMA(train, order=(5, 1, 0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
residuals = pd.DataFrame(model_fit.resid)
residuals.plot(label='residual error')
plt.show()
residuals.plot(kind='kde', label='residual error density')
plt.show()
print(residuals.describe())
'''



