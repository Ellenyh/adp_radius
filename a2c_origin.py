import gym
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
from lyh_config import *
from lyh_simulator_env import Simulator
import pandas as pd
from datetime import datetime
def datelist(beginDate, endDate):
    # beginDate, endDate是形如‘20160601’的字符串或datetime格式
    date_l=[datetime.strftime(x,'%Y-%m-%d') for x in list(pd.date_range(start=beginDate, end=endDate))]
    return date_l



class Net(nn.Module):
    def __init__(self, n_grid):
        super(Net, self).__init__()
        self.n_grid = n_grid

        self.para = nn.Linear(3*n_grid,128)

        self.mu = nn.Linear(128,64)
        self.mu_out = nn.Linear(64,self.n_grid)

        self.sig = nn.Linear(128, 64)
        self.sig_out = nn.Linear(64,self.n_grid)

        self.v = nn.Linear(3*n_grid,128)
        self.v_1 = nn.Linear(128,64)
        self.v_out = nn.Linear(64,1)

    def forward(self, x):
        order = torch.Tensor(x[0])
        driver = torch.Tensor(x[1])
        reward = torch.Tensor(x[2])

        od = torch.cat((order,driver,reward),0)
        od_1 = F.relu(self.para(od))

        mu = torch.tanh(self.mu(od_1))
        mu = F.softplus(self.mu_out(mu))
        sig = torch.tanh(self.sig(od_1))
        sig = F.softmax(self.sig_out(sig),dim=0)

        v = F.relu(self.v(od))
        v = F.relu(self.v_1(v))
        v = self.v_out(v)

        mu = mu + torch.Tensor([1e-5])
        sig = sig + torch.Tensor([1e-5])
        return mu, sig, v


class AC(nn.Module):
    def __init__(self, n_grid, args):
        super(AC, self).__init__()
        self.n_grid = n_grid
        self.gamma = args.gamma
        self.lr = args.lr

        self.model = Net(self.n_grid)
        # self.model.apply(self.init_weights)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def init_weights(self,layer):
        # 如果为卷积层，使用正态分布初始化
        if type(layer) == nn.Conv2d:
            nn.init.normal_(layer.weight, mean=0, std=0.5)
        # 如果为全连接层，权重使用均匀分布初始化，偏置初始化为0.1
        elif type(layer) == nn.Linear:
            nn.init.uniform_(layer.weight, a=-0.1, b=0.1)
            nn.init.constant_(layer.bias, 0.1)

    def choose_action(self, s):
        mu, sig, value = self.model(s)
        action = []
        torch.manual_seed(12)
        for i in range(self.n_grid):
            dist = torch.distributions.Normal(mu[i], sig[i])
            action.append(torch.tanh(dist.sample()).item())
        return np.array(action), value, mu, sig

    def critic_learn(self, s, s_, r, done):

        r = torch.FloatTensor([r])
        mu, sig, v = self.model(s)
        mu_, sig_, v_ = self.model(s_)
        v_ = v_.detach()
        v, v_ = v.squeeze(0), v_.squeeze(0)

        target = r
        if not done:
            target += self.gamma * v_

        loss_func = nn.MSELoss()
        loss = loss_func(v, torch.tensor(target.item()))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        advantage = (target - v).detach()
        return advantage

    def actor_learn(self, advantage, s, a):
        action, value, mu, sig = self.choose_action(s)
        loss = torch.Tensor([0])
        for i in range(len(mu)):
            p1 = - ((mu[i] - action[i]) ** 2) / (2*sig[i]**2)
            p2 = - torch.log(torch.sqrt(2 * math.pi * sig[i] * sig[i]))
            log_prob_v = -advantage * (p1+p2)
            loss += log_prob_v
        loss = loss/len(mu)
        print('loss: ',loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_episodes', type=int, default=500)
    parser.add_argument('--gamma', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.5)
    args = parser.parse_args()

    env_params['block_area'] = 10**6
    env = Simulator(**env_params)
    env.GS.get_grid_distance()
    grid_distance = env.GS.grid_distance
    n_grid = env.GS.num_grid
    initial_radius = radius = np.array([1] * n_grid)

    agent = AC(n_grid,args)

    x, y = [], []
    df = pd.read_pickle('data_used_0.1.pickle')
    for episode in range(args.n_episodes):
        # date = datelist('20150701','20150731')[episode//24] #date

        # cur_time = episode % 24 * 3600 # starting second
        cur_time = 0
        ep_reward = 0
        env.reset()
        s, r, done = env.step(initial_radius * 3000,grid_distance,cur_time,df['2015-07-01'])

        while True:
            a, _, mu, sig = agent.choose_action(s)
            cur_time = cur_time + 60
            s_, r, done = env.step(3000*(a+1),grid_distance,cur_time,df['2015-07-01'])
            ep_reward += r


            advantage = agent.critic_learn(s, s_, r, done)
            agent.actor_learn(advantage, s, a)

            s = s_
            if done:
                break

        print('Episode {:03d} | Reward:{:.03f}'.format(episode, ep_reward))

        x.append(episode)
        y.append(ep_reward)

    plt.plot(x, y)
    plt.savefig('ep_500.png')
