import numpy as np
import tensorflow as tf


class DeepQNetwork:
	def __init__(
		self,
		n_actions,
		n_features,
		learning_rate=0.01,
		reward_decay=0.9,
		e_greedy=0.9,
		replace_target_iter=200,
		memory_size=500,
		batch_size=32,
		e_greedy_increment=None):
		tf.reset_default_graph()
		self.n_actions = n_actions
		self.n_features = n_features
		self.lr = learning_rate
		self.gamma = reward_decay
		self.epsilon_max = e_greedy
		self.replace_target_iter = replace_target_iter
		self.memory_size = memory_size
		self.batch_size = batch_size
		self.nl1 = 512
		self.nl1_5 = 512
		self.epsilon_increment = e_greedy_increment

		self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

		self.learn_step_couter = 0
		self.memory = np.zeros((self.memory_size, n_features * 2 + 2), dtype=float, order='C')
		config = tf.ConfigProto()
		config.gpu_options.per_process_gpu_memory_fraction = 0.25
		self.sess = tf.Session(config=config)
		self._build_net()
		self.saver = tf.train.Saver()
		self.sess.run(tf.global_variables_initializer())
		self.cost_his = []

	def _build_net(self):
		def build_layers(s, c_names, n_l1, w_initializer, b_initializer):
			with tf.variable_scope('l1'):
				w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
				b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
				l1 = tf.nn.relu(tf.matmul(s, w1) + b1)

			with tf.variable_scope('l1_5'):
				w1_5 = tf.get_variable('w1_5', [n_l1, self.nl1_5], initializer=w_initializer, collections=c_names)
				b1_5 = tf.get_variable('b1_5', [1, self.nl1_5], initializer=b_initializer, collections=c_names)
				l1_5 = tf.nn.relu(tf.matmul(l1, w1_5) + b1_5)

			with tf.variable_scope('l2'):
				w2 = tf.get_variable('w2', [self.nl1_5, self.n_actions], initializer=w_initializer, collections=c_names)
				b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
				out = tf.matmul(l1_5, w2) + b2
			return out

		self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
		self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')

		# self.keep_prob_ph = tf.placeholder(tf.float32, name='keep_prob')
		self.r = tf.placeholder(tf.float32, [None, ], name='r')
		self.a = tf.placeholder(tf.int32, [None, ], name='a')

		with tf.variable_scope('eval_net'):
			c_names, n_l1, w_initializer, b_initializer = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], self.nl1, tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

			self.q_eval = build_layers(self.s, c_names, n_l1, w_initializer, b_initializer)

		with tf.variable_scope('target_net'):
			c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
			self.q_next = build_layers(self.s_, c_names, n_l1, w_initializer, b_initializer)

		with tf.variable_scope('q_target'):
			q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')
			self.q_target = tf.stop_gradient(q_target)

		with tf.variable_scope('q_eval'):
			a_one_hot = tf.one_hot(self.a, depth=self.n_actions)
			self.q_eval_wrt_a = tf.reduce_sum(self.q_eval * a_one_hot, axis=1)
		with tf.variable_scope('loss'):
			self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))

		with tf.variable_scope('train'):
			self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

	def store_transition(self, s, a, r, s_):
		if not hasattr(self, 'memory_couter'):
			self.memory_couter = 0

		trainsition = np.hstack((s, [a, r], s_))
		index = self.memory_couter % self.memory_size
		self.memory[index, :] = trainsition
		self.memory_couter += 1

	def choose_action(self, observation):
		observation = observation[np.newaxis, :]
		if np.random.uniform(low=0.0, high=1.0, size=1) < self.epsilon:
			action_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
			# print('action_value shape:{}'.format(action_value.shape))
			action = np.argmax(action_value, axis=None)
		else:
			action = np.random.randint(0, high=self.n_actions, size=None)
		return action

	def _replace_target_params(self):
		t_params = tf.get_collection('target_net_params')
		e_params = tf.get_collection('eval_net_params')
		self.sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])

	def learn(self):
		if self.learn_step_couter % self.replace_target_iter == 0:
			self._replace_target_params()
			# print('\n targert_params_replaced\n')

		if self.memory_couter > self.memory_size:
			sample_index = np.random.choice(self.memory_size, size=self.batch_size, replace=True, p=None)
		else:
			sample_index = np.random.choice(self.memory_couter, size=self.batch_size)
		batch_memory = self.memory[sample_index, :]
		# print(batch_memory.shape)
		_, cost = self.sess.run(
			[self._train_op, self.loss],
			feed_dict={
				self.s: batch_memory[:, :self.n_features],
				self.a: batch_memory[:, self.n_features],
				self.r: batch_memory[:, self.n_features + 1],
				self.s_: batch_memory[:, -self.n_features:]
			})
		self.cost_his.append(cost)

		self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
		self.learn_step_couter += 1

	def plot_cost(self):
		import matplotlib.pyplot as plt
		plt.plot(np.arange(len(self.cost_his)), self.cost_his)
		plt.ylabel('Cost')
		plt.xlabel('training steps')
		plt.show()

	def save_model(self, model_path):
		print('saving_model...')
		try:
			save_path = self.saver.save(self.sess, model_path)
			# self.pre_train_saver.save(sess, model_path + '_part')
		except Exception:
			save_path = self.saver.save(self.sess, './output_model/temp.ckpt')
		finally:
			print('save_path:{}'.format(save_path))

	def reload_model(self, model_path):
		print('reloading model {}.....'.format(model_path))
		self.saver.restore(self.sess, model_path)

if __name__ == '__main__':
	DQN = DeepQNetwork(3, 4)
