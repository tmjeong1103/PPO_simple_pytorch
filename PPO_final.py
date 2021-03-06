import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

print("PyTorch version:[%s]."%(torch.__version__))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device:[%s]."%(device))


# Actor Layer Construction
class ActorClass(nn.Module):

    def __init__(self, name='Actor', state_dim=4, action_dim=2, learning_rate=1e-2, hdims=[64, 32, 16]):

        # class initialize
        super(ActorClass, self).__init__()

        # ActorClass parameter initialize
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hdims = hdims

        self.layers = []

        # Dense Layer construction
        prev_hdim = self.state_dim
        for hdim in self.hdims:
            self.layers.append(nn.Linear(prev_hdim, hdim, bias=True))
            self.layers.append(nn.ReLU())  # activation function = relu
            prev_hdim = hdim

        # Final Layer (without activation)
        self.layers.append(nn.Linear(prev_hdim, self.action_dim))

        # Concatenate all layers
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


# Critic Layer Construction
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

        # Concatenate all layers
        self.net = nn.Sequential()
        for l_idx, layer in enumerate(self.layers):
            layer_name = "%s_%02d" % (type(layer).__name__.lower(), l_idx)
            self.net.add_module(layer_name, layer)

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        val = self.net(x)  # scalar
        return val

# PPO Class + train net
class PPO:
    def __init__(self, state_dim=4, action_dim=2, learning_rate=1e-2, eps_clip=0.2, K_epoch=3, gamma=0.98, lmbda=0.95):

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

            # GAE Calculation
            td_target = r + self.gamma * self.critic.forward(s_prime) * done_mask
            delta = td_target - self.critic.forward(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0

            # Advantage Calculation
            for delta_t in delta[::-1]:  # delta[::-1] : delta??? ????????? ?????? ???????????? ??????
                advantage = self.gamma * self.lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])

            advantage_lst.reverse()  # from "reverse stacked array" to "Forward stacked array"
            advantage = torch.tensor(advantage_lst, dtype=torch.float)  # transform to tensor

            pi, pi_out = self.actor.forward(s, softmax_dim=1)
            pi_a = pi_out.gather(1, a)

            # Loss_clip??? ????????? ?????? ratio(r_t(thete)) ??????
            # a/b == exp(log(a)-log(b)) : ??? ????????? ????????? ????????????, prob_a : old policy
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # new policy/ old policy

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage # PPO clip

            # Loss_clip ??????
            # -policy loss + value loss
            # policy loss : maximize(gradient_ascent), value loss : minimize(gradient_descent)
            # Policy gradient??? ????????????(J)??? ????????? ????????? ?????? ??????.
            # .detach??? ?????? gradient flow ?????? X, TD_target??? ??????????????? ?????? ?????? ???????????? ??? ????????? ??????(????????? ??????)
            # (??????!).detach??? ????????? ?????? loss??? ????????? ???????????? update??? ???. target??? ????????? ??????.
            # (??????!) ratio??? clip??? ????????? ????????????? ????????? sample??? ?????? X, ????????? ??????!
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.critic.forward(s), td_target.detach())

            self.actor.optimizer.zero_grad()
            self.critic.optimizer.zero_grad()
            loss.mean().backward()

            self.actor.optimizer.step()
            self.critic.optimizer.step()

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []

        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition

            s_lst.append(s)  # s:numpy array, shape:[...]
            a_lst.append([a])  # s??? shape??? ????????? ?????? []??? ??????./ pytorch unsqueeze(?????? ??????.)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])

            if done:
                done_mask = 0
            else:
                done_mask = 1
            done_lst.append([done_mask])

        s, a, r, s_prime, done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                              torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                              torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a

    def put_data(self, transition):
        self.data.append(transition)

def main():

    ####### Hyperparameters #######
    env = gym.make('CartPole-v1')
    score = 0.0
    print_interval = 20
    T_horizon = 20
    eps_clip = 0.2
    K_epoch = 3
    lmbda = 0.95
    gamma = 0.98
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    learning_rate = 1e-2
    ###############################

    agent = PPO(state_dim=state_dim,action_dim=action_dim,learning_rate=learning_rate,\
                eps_clip=eps_clip,K_epoch=K_epoch,gamma=gamma,lmbda=lmbda)

    for n_epi in range(1000):
        s = env.reset()
        done = False

        # T_horizon step ????????? data??? ????????? train.
        while not done:
            for t in range(T_horizon):
                prob, prob_out = agent.actor(torch.Tensor(s), -1)

                env.render()
                a = prob.sample().item()
                s_prime, r, done, info = env.step(a)

                # prob[a].item() : ????????? ??? action??? ?????????, ?????? ratio??? ????????? ??? ??????
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

if __name__ == '__main__':
    main()