from env import Milano_env
from DeepQ import DeepQNetwork
import numpy as np
import matplotlib.pyplot as plt

class Run_Offloading:
	def __init__(self):
		self.env = Milano_env()
		self.RL = DeepQNetwork(
			self.env.n_actions,
			self.env.n_features,
			learning_rate=0.01,
			reward_decay=0.9,
			e_greedy=0.9,
			replace_target_iter=200,
			memory_size=1000,
			batch_size=50,
			e_greedy_increment=None)

	def RL_train(self):
		step = 0
		for episode in range(300):
			observation = self.env.reset()
			while True:
				action = self.RL.choose_action(observation)
				# action = 0
				observation_, reward, done = self.env.step(action)
				print('episode:{} step:{} obs:{} action:{}, reward:{}'.format(episode, step, observation, action, reward))

				if done:
					break
				self.RL.store_transition(observation, action, reward, observation_)
				if (step > 200) and (step % 5 == 0):
					self.RL.learn()
					# pass
				observation = observation_

				step += 1

		print('over')
		self.RL.plot_cost()
		self.RL.save_model('./output_model/deepQ.ckpt')

	def run_test_with_RL(self, reload=False):
		if reload:
			self.RL.reload_model('./output_model/deepQ.ckpt')
		reward_list = []
		observation = self.env.reset()

		while True:
			action = self.RL.choose_action(observation)
			observation_, reward, done = self.env.step(action)
			if done:
				break
			reward_list.append(reward)
			observation = observation_

		reward_array = np.array(reward_list)
		return reward_array

	def run_test_without_RL(self, action=10):
		reward_list = []

		observation = self.env.reset()
		while True:
			# action = action
			observation_, reward, done = self.env.step(action)
			if done:
				break
			reward_list.append(reward)
			observation = observation_

		reward_array = np.array(reward_list)
		return reward_array


def offloading_plot(with_RL, without_RL, without_RL_without_offloading):
	# print(without_RL_without_offloading.shape)
	plt.figure()
	plt.xlabel('time sequence')
	plt.ylabel('energy efficiency(Mbps/s/J)')
	plt.plot(with_RL, label='with RL', marker='.')
	plt.plot(without_RL, label='without RL', marker='.')
	plt.plot(without_RL_without_offloading, label='without RL without offlading', marker='.')
	plt.grid(b=None, which='major', axis='both')
	plt.legend()
	plt.show()


if __name__ == "__main__":
	offloading = Run_Offloading()
	offloading.RL_train()
	reward_with_RL = offloading.run_test_with_RL(reload=True)
	reward_without_RL_with_offloading = offloading.run_test_without_RL(10)
	reward_without_RL_without_offloading = offloading.run_test_without_RL(0)
	offloading_plot(reward_with_RL[-120:], reward_without_RL_with_offloading[-120:], reward_without_RL_without_offloading[-120:])

