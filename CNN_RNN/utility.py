from sklearn import preprocessing
from datetime import datetime
import pytz
# import numpy as np


def feature_scaling(input_datas, scaler=None, feature_range=(0.1, 255)):
	# print(input_datas.shape)
	input_shape = input_datas.shape
	input_datas = input_datas.reshape(-1, 1)
	# print(np.amin(input_datas))
	if scaler:
		output = scaler.transform(input_datas)
	else:
		scaler = preprocessing.MinMaxScaler(feature_range=feature_range)
		output = scaler.fit_transform(input_datas)

	output = output.reshape(input_shape)
	return output, scaler


def un_feature_scaling(input_data, scaler):
	input_shape = input_data.shape
	input_data = input_data.reshape(-1, 1)
	output = scaler.inverse_transform(input_data)
	output = output.reshape(input_shape)
	return output


def set_time_zone(timestamp):
	UTC_timezone = pytz.timezone('UTC')
	Mi_timezone = pytz.timezone('Europe/Rome')
	date_time = datetime.utcfromtimestamp(float(timestamp))
	date_time = date_time.replace(tzinfo=UTC_timezone)
	date_time = date_time.astimezone(Mi_timezone)
	return date_time


def date_time_covert_to_str(date_time):
	return date_time.strftime('%m-%d %H')
