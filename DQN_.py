# -*- coding = utf-8 -*-
# @Time : 2021/12/21 15:21
# @Author : qiu
# @File : DQN_.py
# @Software : PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy 经验收集时action的选取方式
GAMMA = 0.9                 # reward discount
Target_Replace_Iter: int = 100   # target update frequency
Memory_Capacity: int = 2000      # 经验池容量
env = gym.make('CartPole-v0').unwrapped         # create environment and unwrapped 创建环境打开封装 gym 一般上层包装一个env 包含所有环境通用的功能
N_Actions: int = env.action_space.n                  # 获取 agent’s action 的大小
N_States: int = env.observation_space.shape[0]       # state 的维度大小
s = env.reset()
# s_, r, done, info = env.step(s)


class Net(nn.Module):
    """
    用Net来近似 target_net 和 behavior_net
        * 使用最简单的 linear 全连接层
        * Q(s, a | w)
        * s 可用一个length 为 4 list??
    """
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_States, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(50, N_Actions)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        actions_value = self.out(x)
        return actions_value


# define DQN  this class includes choose the action of function , the storing transition and learning function
class DQN(object):
    def __init__(self):
        self.eval_net = Net()
        self.target_net = Net()
        self.learn_step_counter = 0
        self.memory_counter = 0     # memory counter
        self.memory = np.zeros((Memory_Capacity, N_States * 2 + 2))              # why is N_States * 2 + 2 根据最大容量生成经验池
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)    # 选择优化器 Adam 把eval_net 作为 更新对象
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)    # 增加维度
        if np.random.uniform() < EPSILON:               # 小于选择最大 max eval_net(x) 大于 通过均匀分布选择action
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]
        else:
            action = np.random.randint(0, N_Actions)

        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # 当经验池满了时，可以安先后顺序替换经验   == 这可以修改
        index = self.memory_counter % Memory_Capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # 更新参数网络参数
        if self.learn_step_counter % Target_Replace_Iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # 随机抽取数据（经验）
        sample_index = np.random.choice(Memory_Capacity, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_States])
        b_a = torch.LongTensor(b_memory[:, N_States: N_States+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_States+1: N_States+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_States:])

        q_eval = self.eval_net(b_s).gather(1, b_a)  # 根据经验池里的动作挑选对应的Q(s, a) 价值评估
        q_next = self.target_net(b_s_).detach()     # 用非优化网络来生成Q(s_, a)来避免自举
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)

        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def plot_reward(reward_l, episode, fig_size=(8, 6)):
    fig = plt.figure(figsize=fig_size)
    ax = plt.axes()
    ax.set_title(f"Episode {episode} Leaning Curve", fontdict={"fontsize": 20, "color": 'yellow'})
    ax.plot(np.arange(len(reward_l)), reward_l, label='reward', color='r')
    fig.legend()
    fig.show()


dqn = DQN()
learning_episode = 0
str_flag = "Collect experiences"
reward_list = []

for i in range(400):

    str_log_start = f'Episode:---{str_flag}----{i}'
    with open('log_dqn.txt', 'a') as f:
        f.writelines(str_log_start + '\n')
    print(str_log_start)
    current_state = env.reset()             # 重置环境, every episode all need to reset environment
    episode_reward_sum = 0                  # episode 的总的奖励

    while True:
        env.render()                        # 刷新屏幕
        action = dqn.choose_action(current_state)   # 选择action
        next_state, immediate_reward, done, info = env.step(action)
        # 修改奖励， 基于 next_state
        x, x_dot,  theta, theta_dot = next_state
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        dqn.store_transition(current_state, action, r, next_state)

        episode_reward_sum += r

        if dqn.memory_counter > Memory_Capacity:
            str_flag = "replaying and collecting the experiences"
            dqn.learn()
            if done:
                learning_episode += 1
                str_log_done = f'Episode: {learning_episode}, | Episode_reward_sum: {round(episode_reward_sum, 2)}'
                reward_list.append(round(episode_reward_sum, 2))
                with open('log_dqn.txt', 'a') as f:
                    f.writelines(str_log_done + '\n')
                print(str_log_done)
                if learning_episode % 30 == 0:
                    plot_reward(reward_list, learning_episode+1)

        if done:
            break

        current_state = next_state



























