import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

print("PyTorch version:[%s]."%(torch.__version__))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device:[%s]."%(device))

# Actor layer construction

class ActorClass(nn.Module):

    def __init__(self, name='Actor', state_dim=4, action_dim=2, learning_rate=1e-2, hdims=[64, 32, 16]):

        # class initialize
        super(ActorClass, self).__init__()

        # ActorClass parameter initialize
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hdims = hdims
        # print("Action dim ", self.action_dim)

        self.layers = []

        # Dense Layer construction
        prev_hdim = self.state_dim
        for hdim in self.hdims:
            self.layers.append(nn.Linear(prev_hdim, hdim, bias=True))
            self.layers.append(nn.ReLU())  # activation function = relu
            prev_hdim = hdim

        # Final Layer (without activation)
        # print("DEBUG")
        # print(type(prev_hdim))
        # print(self.action_dim)
        self.layers.append(nn.Linear(prev_hdim, self.action_dim))

        # Conacatenate all layers
        self.net = nn.Sequential()
        for l_idx, layer in enumerate(self.layers):
            layer_name = "%s_%02d" % (type(layer).__name__.lower(), l_idx)
            self.net.add_module(layer_name, layer)

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x, softmax_dim=0):
        x = self.net(x)
        prob_out = F.softmax(x, dim=softmax_dim)  # activation function : softmax
        prob = Categorical(probs=prob_out)
        return prob, prob_out


class CriticClass(nn.Module):
    def __init__(self, name='Critic', state_dim=4, learning_rate=1e-2, hdims=[64, 32, 16]):

        # class initialize
        super(CriticClass, self).__init__()

        # ActorClass parameter initialize
        self.state_dim = state_dim
        self.hdims = hdims

        self.layers = []

        # Dense Layer construction
        prev_hdim = self.state_dim
        for hdim in self.hdims:
            self.layers.append(nn.Linear(prev_hdim, hdim, bias=True))
            self.layers.append(nn.ReLU())  # activation function = relu
            prev_hdim = hdim

        # Final Layer (without activation)
        self.layers.append(nn.Linear(prev_hdim, 1))

        # Conacatenate all layers
        self.net = nn.Sequential()
        for l_idx, layer in enumerate(self.layers):
            layer_name = "%s_%02d" % (type(layer).__name__.lower(), l_idx)
            self.net.add_module(layer_name, layer)

            # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        val = self.net(x)  # scalar
        return val


class PPO:
    def __init__(self, state_dim=4, action_dim=2, learning_rate=1e-2, eps_clip=0.2, K_epoch=3, gamma=0.98, lmbda=0.95):

        # Hyperparameter
        self.eps_clip = eps_clip
        self.K_epoch = K_epoch
        self.lmbda = lmbda
        self.gamma = gamma

        # model
        self.actor = ActorClass(state_dim=state_dim, action_dim=action_dim, learning_rate=learning_rate)
        self.critic = CriticClass(state_dim=state_dim, learning_rate=learning_rate)
        self.data = []

    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(self.K_epoch):

            # GAE calculation
            td_target = r + self.gamma * self.critic.forward(s_prime) * done_mask
            delta = td_target - self.critic.forward(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0

            for delta_t in delta[::-1]:  # delta[::-1] : delta를 뒤에서 부터 뽑아와서 축적
                advantage = self.gamma * self.lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])

            advantage_lst.reverse()  # from "reverse stacked array" to "Forward stacked array"
            advantage = torch.tensor(advantage_lst, dtype=torch.float)  # transform to tensor

            pi, pi_out = self.actor.forward(s, softmax_dim=1)
            pi_a = pi_out.gather(1, a)

            # Loss_clip을 구하기 위해 ratio(r_t(thete)) 계산
            # a/b == exp(log(a)-log(b)) : 이 형태가 연산이 효율적임, prob_a : old policy
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # new policy/ old policy

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage # PPO clip

            # Loss_clip 적용
            # -policy loss + value loss
            # policy loss : maximize(gradient_ascent), value loss : minimize(gradient_descent)
            # Policy gradient는 목적함수(J)를 최대화 시키는 것이 목표.
            # .detach의 경우 gradient flow 발생 X, TD_target이 만들어지기까지 앞의 그래프는 다 뗀다는 의미(상수로 본다)
            # (중요!).detach를 안하는 경우 loss를 줄이는 방향으로 update가 됨. target은 변하면 안됨.
            # (중요!) ratio가 clip의 범위를 넘어간다? 그때의 sample은 학습 X, 버리는 효과!
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.critic.forward(s), td_target.detach())

            self.actor.optimizer.zero_grad()
            self.critic.optimizer.zero_grad()
            loss.mean().backward()

            self.actor.optimizer.step()
            self.critic.optimizer.step()

    def make_batch(self):
        # initialize batch
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []

        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            
            # storing data
            s_lst.append(s)  # s:numpy array, shape:[...]
            a_lst.append([a])  # s와의 shape을 맞추기 위해서 [](array)를 취하고 append, pytorch unsqueeze(차원 증가)방식도 있다.
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])

            # done_mask = 0 if done else 1
            if done:
                done_mask = 0
            else:
                done_mask = 1
                
            done_lst.append([done_mask])

        #transform to tensor
        s, a, r, s_prime, done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                              torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                              torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a

    def put_data(self, transition):
        self.data.append(transition)

        
###### main ######
env = gym.make('CartPole-v1')
score = 0.0
print_interval = 20
T_horizon = 20

# print(device)
agent = PPO()

for n_epi in range(1000):
    s = env.reset()
    done = False

    # T_horizon step 만큼만 data를 모으고 학습시킨다.
    while not done:
        for t in range(T_horizon):
            prob, prob_out = agent.actor(torch.Tensor(s), -1)

            #env.render()
            a = prob.sample().item()
            s_prime, r, done, info = env.step(a)

            # prob[a].item() : 실제로 한 action의 확률값, 나중 ratio를 계산할 때 사용
            agent.put_data((s, a, r / 100.0, s_prime, prob_out[a].item(), done))
            s = s_prime
            score += r

            if done:
                break

    agent.train_net()

    if n_epi % print_interval == 0 and n_epi != 0:
        print("# of episode :{}, avg score : {:.1f}".format(n_epi, score / print_interval))
        score = 0.0

env.close()
