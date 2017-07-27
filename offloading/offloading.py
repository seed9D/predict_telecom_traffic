from env import Milano_env, Env_Config
from DeepQ import DeepQNetwork
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('/home/mldp/ML_with_bigdata')
import CNN_RNN.utility

logger = CNN_RNN.utility.setlog('offloading')
root_dir = '/home/mldp/ML_with_bigdata'


class Run_Offloading:
	def __init__(self, model_path, config, cell_num=867):
		self.cell_num = cell_num
		self.model_path = model_path
		self.env = Milano_env(cell_num, config)
		self.RL = DeepQNetwork(
			self.env.n_actions,
			self.env.n_features,
			learning_rate=0.001,
			reward_decay=0.9,
			e_greedy=0.9,
			replace_target_iter=200,
			memory_size=1000,
			batch_size=80,
			e_greedy_increment=None)

	def __evaluation(self, episode, mins=None):
		result_dict = self.run_test_with_RL(reload=False, mins=mins)
		training_result_effi = np.mean(result_dict['energy_effi'][:-149])
		testing_result_effi = np.mean(result_dict['energy_effi'][-149:])
		training_result_reward = np.mean(result_dict['reward'][:-149])
		testing_result_reward = np.mean(result_dict['reward'][-149:])

		training_result_macro_load = np.mean(result_dict['macro_load'][:-149])
		testing_result_macro_load = np.mean(result_dict['macro_load'][-149:])

		training_result_small_load = result_dict['small_load'][:-149]
		training_result_small_load[training_result_small_load == 0] = np.nan
		training_result_small_load = np.nanmean(training_result_small_load)

		testing_result_small_load = result_dict['small_load'][-149:]
		testing_result_small_load[testing_result_small_load == 0] = np.nan
		testing_result_small_load = np.nanmean(testing_result_small_load)
		print('episode:{} training: reward:{:.4f} macro_load:{:.4f} small load:{:.4f} effi:{:.4f} '.format(episode, training_result_reward, training_result_macro_load, training_result_small_load, training_result_effi))
		print('episode:{} testing: reward:{:.4f} macro load:{:.4f} small load:{:.4f} effi:{:.4f}'.format(episode, testing_result_reward, testing_result_macro_load, testing_result_small_load, testing_result_effi))
		print()

	def RL_train(self, mins=None):
		step = 0
		for episode in range(100):
			if mins:
				observation = self.env.reset_10_mins(training=True)
			else:
				observation = self.env.reset(training=True)
			while True:
				action = self.RL.choose_action(observation)
				# action = 0
				if mins:
					observation_, reward, done = self.env.step_10_mins(action, training=True)
				else:
					observation_, reward, done = self.env.step(action, training=True)

				if done:
					break
				# print('episode:{} step:{} obs:{} action:{}, reward:{}'.format(episode, step, observation, action, reward[0]))
				self.RL.store_transition(observation, action, reward[0], observation_)
				if (step > 200) and (step % 5 == 0):
					self.RL.learn()
				observation = observation_

				step += 1

			if episode % 10 == 0 and episode is not 0:
				# self.RL.save_model('./output_model/deepQ_500.ckpt')
				self.__evaluation(episode, mins)
		print('over')
		# self.RL.plot_cost()
		# self.RL.save_model('./output_model/deepQ_500.ckpt')
		self.RL.save_model(self.model_path)

	def run_test_with_RL(self, reload=True, mins=None):
		if reload:
			# self.RL.reload_model('./output_model/deepQ_500_10mins.ckpt')
			self.RL.reload_model(self.model_path)
		reward_list = []
		energy_effi_list = []
		if mins:
			observation = self.env.reset_10_mins(training=False)
		else:
			observation = self.env.reset(training=False)

		while True:
			action = self.RL.choose_action(observation)
			if mins:
				observation_, reward, done = self.env.step_10_mins(action, training=False)
			else:
				observation_, reward, done = self.env.step(action, training=False)
			if done:
				break
			reward_list.append(reward[0])
			energy_effi_list.append(reward[1])
			observation = observation_

		reward_array = np.array(reward_list)
		energy_effi_arry = np.array(energy_effi_list)
		result_dict = self.__get_result_information()
		result_dict['energy_effi'] = energy_effi_arry.reshape(-1, 1)
		result_dict['reward'] = reward_array.reshape(-1, 1)
		return result_dict

	def run_test_without_RL(self, action=10):
		reward_list = []
		energy_effi_list = []

		observation = self.env.reset(training=False)
		while True:
			# action = action
			observation_, reward, done = self.env.step(action, training=False)
			if done:
				break
			reward_list.append(reward[0])
			energy_effi_list.append(reward[1])
			observation = observation_

		reward_array = np.array(reward_list)
		energy_effi_arry = np.array(energy_effi_list)
		result_dict = self.__get_result_information()
		result_dict['energy_effi'] = energy_effi_arry.reshape(-1, 1)
		result_dict['reward'] = reward_array.reshape(-1, 1)
		return result_dict

	def __get_result_information(self):
		total_power_consumption = self.env.total_power_consumption
		macro_cell_load = self.env.macro_cell_load
		small_cell_load = self.env.small_cell_load
		internet_traffic_demand = self.env.internet_traffic_demand
		actions = self.env.actions
		information_dict = {
			'traffic_demand': np.array(internet_traffic_demand).reshape(-1, 1),
			'macro_load': np.array(macro_cell_load).reshape(-1, 1),
			'small_load': np.array(small_cell_load).reshape(-1, 1),
			'power_consumption': np.array(total_power_consumption).reshape(-1, 1),
			'action': np.array(actions).reshape(-1, 1)
		}

		return information_dict


def offloading_plot(with_RL, without_RL, without_RL_without_offloading):
	def plot_each_method_macro_load(fig_instance, with_RL, without_RL, without_RL_without_offloading):
		ax = fig_instance.add_subplot(1, 1, 1)
		ax.cla()
		ax.set_xlabel('Time sequence')
		ax.set_ylabel('Macro load')
		ax.plot(with_RL, label='With RL', marker='.')
		ax.plot(without_RL, label='offloading without RL', marker='')
		ax.plot(without_RL_without_offloading, label='Without offloading', color='k', linestyle='--')
		ax.grid(b=None, which='major', axis='both')
		ax.legend()

	def plot_each_method_small_load(fig_instance, with_RL, without_RL, without_RL_without_offloading):
		ax = fig_instance.add_subplot(1, 1, 1)
		ax.cla()
		ax.set_xlabel('Time sequence')
		ax.set_ylabel('Small load')
		ax.plot(with_RL, label='With RL', marker='.')
		ax.plot(without_RL, label='offloading without RL', marker='')
		ax.plot(without_RL_without_offloading, label='Without offloading', color='k', linestyle='--')
		ax.grid(b=None, which='major', axis='both')
		ax.legend()

	def plot_each_method_energy_effi(fig_instance, with_RL, without_RL, without_RL_without_offloading):
		print('with_RL mean:{} without_RL mean:{} without_RL_without_offloading mean:{}'.format(np.mean(with_RL), np.mean(without_RL), np.mean(without_RL_without_offloading)))
		ax = fig_instance.add_subplot(1, 1, 1)
		ax.cla()
		ax.set_xlabel('Time sequence')
		ax.set_ylabel('Energy efficiency(Mbps/s/J)')
		ax.plot(with_RL, label='With RL', marker='.')
		ax.plot(without_RL, label='offloading without RL', marker='')
		ax.plot(without_RL_without_offloading, label='Without offloading', color='k', linestyle='--')
		ax.grid(b=None, which='major', axis='both')
		ax.legend()

	def plot_each_method_power_consumption(fig_instance, with_RL, without_RL, without_RL_without_offloading):
		ax = fig_instance.add_subplot(1, 1, 1)
		ax.cla()
		ax.set_xlabel('Time sequence')
		ax.set_ylabel('Power consumption(W)')
		ax.plot(with_RL, label='With RL', marker='.')
		ax.plot(without_RL, label='offloading without RL', marker='')
		ax.plot(without_RL_without_offloading, label='without offloading', color='k', linestyle='--')
		ax.grid(b=None, which='major', axis='both')
		ax.legend()

	def plot_each_method_action(fig_instance, with_RL, without_RL, without_RL_without_offloading):
		ax = fig_instance.add_subplot(1, 1, 1)
		ax.cla()
		ax.set_xlabel('Time sequence')
		ax.set_ylabel('action')
		ax.plot(with_RL, label='With RL', marker='.')
		ax.plot(without_RL, label='offloading without RL', marker='')
		ax.plot(without_RL_without_offloading, label='without offloading', color='k', linestyle='--')
		ax.grid(b=None, which='major', axis='both')
		ax.legend()

	fig_energy = plt.figure('energy effi')
	fig_power = plt.figure('power consumption')
	fig_macro_load = plt.figure('macro load')
	fig_action = plt.figure('action')
	fig_small_load = plt.figure('small load')

	plot_range = -149
	plot_each_method_small_load
	plot_each_method_macro_load(fig_macro_load, with_RL['macro_load'][plot_range:], without_RL['macro_load'][plot_range:], without_RL_without_offloading['macro_load'][plot_range:])
	plot_each_method_small_load(fig_small_load, with_RL['small_load'][plot_range:], without_RL['small_load'][plot_range:], without_RL_without_offloading['small_load'][plot_range:])
	plot_each_method_energy_effi(fig_energy, with_RL['energy_effi'][plot_range:], without_RL['energy_effi'][plot_range:], without_RL_without_offloading['energy_effi'][plot_range:])
	plot_each_method_power_consumption(fig_power, with_RL['power_consumption'][plot_range:], without_RL['power_consumption'][plot_range:], without_RL_without_offloading['power_consumption'][plot_range:])
	plot_each_method_action(fig_action, with_RL['action'][plot_range:], without_RL['action'][plot_range:], without_RL_without_offloading['action'][plot_range:])
	plt.pause(0.001)
	plt.show()


if __name__ == "__main__":
	model_path = os.path.join(root_dir, 'offloading/output_model/deepQ_test.ckpt')
	config = Env_Config()
	offloading = Run_Offloading(model_path, config, 768)
	without_RL_with_offloading_result_dict = offloading.run_test_without_RL(10)
	without_RL_offloading_result_dict = offloading.run_test_without_RL(0)
	without_RL_with_offloading_test_effi = np.mean(without_RL_with_offloading_result_dict['energy_effi'][-149:])
	without_RL_offloading_test_effi = np.mean(without_RL_offloading_result_dict['energy_effi'][-149:])
	print('offloading without RL effi:{}, without_offloading effi:{}'.format(without_RL_with_offloading_test_effi, without_RL_offloading_test_effi))
	offloading.RL_train()
	with_RL_result_dict = offloading.run_test_with_RL(reload=True)
	offloading_plot(with_RL_result_dict, without_RL_with_offloading_result_dict, without_RL_offloading_result_dict)
