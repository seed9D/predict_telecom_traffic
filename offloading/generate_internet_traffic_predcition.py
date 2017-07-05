from env import Internet_Traffic, compute_row_col, comput_grid_id


def loop_all_grid():

	def check_grid_completion(grid_list, traffic):
		Y_info = traffic.Y_info
		for row_ in range(Y_info.shape[2]):
			for col in range(Y_info.shape[3]):
				grid_id = Y_info[0, 0, row_, col, 0]
				print(grid_id)
				grid_list.append(grid_id)
		return grid_list

	# output_col_range = list(range(13, 85, 3))

	traffic = Internet_Traffic()

	row_center_list = list(range(40, 80, 3))
	col_center_list = list(range(30, 70, 3))
	'''
	gird_id, timestamp, real_min, real_avg, real_max, prediction_min, prediction_avg, prediction_max
	'''

	traffic_array = np.zeros([1487, 1, grid_row_num, grid_column_num, 8], dtype=float)

	for row_center in row_center_list:
		for col_center in col_center_list:
			print('row_center:{} col_center:{}'.format(row_center, col_center))
			grid_id = comput_grid_id(row_center, col_center)
			traffic.set_grid_id(grid_id)
			# if row_center == row_center_list[0] and col_center ==
			# col_center_list[0]:
			traffic.set_NN_mode()
			traffic.run_train()
			traffic.run_predict()
			traffic_data = traffic.fetch_real_prediction()

			for i in range(traffic_data.shape[0]):
				for j in range(traffic_data.shape[1]):
					for row in range(traffic_data.shape[2]):
						for col in range(traffic_data.shape[3]):
							grid_id = traffic_data[i, j, row, col, 0]
							row_index, col_index = compute_row_col(grid_id)
							# print(row_index, col_index)
							traffic_array[i, j, row_index, col_index] = traffic_data[
								i, j, row, col]
							# print(traffic_array[i, j, row_index, col_index])

		du.save_array(traffic_array, './result/real_prediction_traffic_array')
	du.save_array(traffic_array, './result/real_prediction_traffic_array')


def nine_grid():

	traffic = Internet_Traffic()
	traffic.set_grid_id(3257)
	traffic.set_NN_mode()
	traffic.run_train()
	traffic.run_predict()
	# traffic.fetch_real_prediction()
	# compute_row_col()

if __name__ == "__main__":
	# nine_grid()
	# loop_all_grid()