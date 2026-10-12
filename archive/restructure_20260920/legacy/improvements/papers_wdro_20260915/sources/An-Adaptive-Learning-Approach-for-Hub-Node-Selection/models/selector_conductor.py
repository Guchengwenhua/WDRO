import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli
from .layers import Done, Embedder, Glimpse, Pointer
from .attention import Encoder

class Actor(nn.Module):
    def __init__(self, degree, device, kamma=0.5):
        super(Actor, self).__init__()
        self.degree = degree
        self.device = device

        # embedder args
        self.d_input = 2
        self.d_model = 128
        self.kamma = kamma
        self.embedder = Embedder(degree, self.d_input, self.d_model)
        self.embedder2 = Embedder(degree, self.d_input, self.d_model)
        # encoder args
        self.num_stacks = 3
        self.num_heads = 16
        self.d_k = 16
        self.d_v = 16
        # feedforward layer inner
        self.d_inner = 512

        self.encoder = Encoder(self.num_stacks, self.num_heads, self.d_k, self.d_v, self.d_model*2, self.d_inner)
        self.encoder2 = Encoder(self.num_stacks, self.num_heads, self.d_k, self.d_v, self.d_model*2, self.d_inner)
        # decoder args
        self.d_unit = 256
        self.d_query = 360
        self.conv1d_r = nn.Conv1d(self.d_model*2, self.d_unit, 1)
        self.conv1d_r2 = nn.Conv1d(self.d_model*2, self.d_unit, 1)
        # self.conv1d_x = nn.Conv1d(self.d_model*2, self.d_unit, 1)
        # self.conv1d_y = nn.Conv1d(self.d_model*2, self.d_unit, 1)

        self.start_ptr = Pointer(self.d_query, self.d_unit)
        self.q_root = nn.Linear(self.d_model*2, self.d_query, bias=False)
        self.q_root2 = nn.Linear(self.d_model*2, self.d_query, bias=False)
        self.q_x = nn.Linear(self.d_model*2, self.d_query, bias=False)
        self.q_x2 = nn.Linear(self.d_model*2, self.d_query, bias=False)
        self.relu = nn.ReLU()
        self.ctx_linear = nn.Linear(self.d_query, self.d_query, bias=False)
        self.ctx_linear2 = nn.Linear(self.d_query, self.d_query, bias=False)
        
        self.ptr1 = Pointer(self.d_query, self.d_unit)
        # self.ptr2 = Pointer(self.d_query, self.d_unit)
        self.done = Done(self.d_query, self.d_unit, self.d_model)
        self.to(device)
        self.train()

    def forward(self, inputs, steps, deterministic=False):  # steps [1, 3]
        # encode encode encode
        #inputs_tensor = torch.tensor(inputs, dtype=torch.float).to(self.device)
        if not isinstance(inputs, torch.Tensor):
            inputs= torch.tensor(inputs, dtype=torch.float)
        inputs_tensor = inputs.to(self.device)
        embedings = self.embedder(inputs_tensor)
        embedings_done = self.embedder2(inputs_tensor)
        root_emb = embedings[:, 0, :].unsqueeze(1).repeat(1, embedings.shape[1], 1)
        root_emb_done = embedings_done[:, 0, :].unsqueeze(1).repeat(1, embedings_done.shape[1], 1)
        embedings = torch.cat([embedings, root_emb], dim=-1)
        embedings_done = torch.cat([embedings_done, root_emb_done], dim=-1)
        encodings = self.encoder(embedings, None).permute(0, 2, 1)
        encodings_done = self.encoder2(embedings_done, None).permute(0, 2, 1)
        enc_r = self.conv1d_r(encodings).permute(0, 2, 1) # rest中选择第一个点时的W
        enc_r_done = self.conv1d_r2(encodings_done).permute(0, 2, 1)
        # enc_x = self.conv1d_x(encodings).permute(0, 2, 1) # x，y是后面选择时候的W
        # enc_y = self.conv1d_y(encodings).permute(0, 2, 1)
        encodings = encodings.permute(0, 2, 1)
        encodings_done = encodings_done.permute(0, 2, 1)
        batch_size = encodings.size()[0]
        # enc_xy = torch.cat([enc_x, enc_y], 1)
        over_flag = torch.zeros(batch_size).to(self.device)  
        reward = torch.zeros(batch_size, dtype=torch.float).to(self.device)
        ctn_p = self.done(enc_r_done, torch.zeros([batch_size, self.d_query], dtype=torch.float).to(self.device))  # continue probality
        ctn = torch.ones(batch_size).to(self.device)  # if go or not go
        done_distr = Bernoulli(probs=ctn_p)
        if deterministic:
            ctn[torch.where(ctn_p < 0.5)[0]] = 0
        else:
            ctn = done_distr.sample()
        reward = reward + ctn * 0.1
        visited = torch.zeros([batch_size, self.degree], dtype=torch.bool).to(self.device)
        visited[:, 0] = 1

        indexes, log_probs, done_probs = [], [], []

        start_logits = self.start_ptr(enc_r,
            torch.zeros([batch_size, self.d_query], dtype=torch.float).to(self.device), visited)
        distr = Categorical(logits=start_logits)
        if deterministic:
            _, start_idx = torch.max(start_logits, -1)
        else:
            start_idx = distr.sample()
        
        start_idx[torch.where(ctn == 0)[0]] = 0  # the probality of selecting point 0 is 0 too, but we need to set no grad.
        visited.scatter_(1, start_idx.unsqueeze(-1), True)  # set the node as visited status
        log_p = distr.log_prob(start_idx)
        log_p[torch.where(ctn == 0)[0]] = 0
        log_probs.append(log_p)
        indexes.append(start_idx)

        log_done = done_distr.log_prob(ctn)
        log_done[torch.where(over_flag==1)[0]] = 0
        done_probs.append(log_done)
        over_flag[torch.where(ctn == 0)[0]] = 1  # set over flag at 1, means the net selection is over.
        batch_indices = torch.arange(batch_size, device=self.device)
        qr = encodings[batch_indices, 0]
        qx = encodings[batch_indices, start_idx]

        context = torch.zeros([batch_size, self.d_query]).to(self.device)
        context2 = torch.zeros([batch_size, self.d_query]).to(self.device)
        # kamma = 0.5  # reward 衰减参数
        for step in range(1, steps):
            # print(over_flag)
            # print(ctn_p)
            residual = self.q_root(qr) + self.q_x(qx)   # 上一步连接无向边的表示
            residual_done = self.q_root2(qr) + self.q_x2(qx)
            context = torch.max(context, self.ctx_linear(self.relu(residual)))  # 环境表示
            context2 = torch.max(context2, self.ctx_linear2(self.relu(residual_done)))
            node_q = residual + context  
            node_q_done = residual_done + context2
            node_query = self.relu(node_q)  # 查询向量
            node_query_done = self.relu(node_q_done)
            
            ctn_p = self.done(enc_r_done, node_query_done)  # continue probality
            ctn = torch.ones(batch_size).to(self.device)  # if go or not go
            done_distr = Bernoulli(probs=ctn_p)
            if deterministic:
                ctn[torch.where(ctn_p < 0.5)[0]] = 0
            else:
                ctn = done_distr.sample()
            ctn[torch.where(over_flag==1)[0]] = 0
            reward = reward + ctn * 0.1 * (self.kamma ** step)
            # first_idx  unvisited
            logits = self.ptr1(enc_r, node_query, visited)
            distr = Categorical(logits=logits)
            if deterministic:
                _, node_idx = torch.max(logits, -1)
            else:
                node_idx = distr.sample()
            node_idx[torch.where(ctn == 0)[0]] = 0
            log_p = distr.log_prob(node_idx)
            log_p[torch.where(ctn == 0)[0]] = 0
            log_probs.append(log_p)
            indexes.append(node_idx)
            visited.scatter_(1, node_idx.unsqueeze(-1), True)

            log_done = done_distr.log_prob(ctn)  # ctn是0或者1，已经是代表本身的取值了，不需要再考虑1-p
            log_done[torch.where(over_flag==1)[0]] = 0  # 上一步就已经停止的置为0，本步停止的仍然计算概率
            done_probs.append(log_done)
            over_flag[torch.where(ctn == 0)[0]] = 1
            qx = encodings[batch_indices, node_idx]

        indexes = torch.stack(indexes, -1)
        # print(indexes)
        # print('*************')
        # print(log_probs)
        # print('*************')
        # print(done_probs)
        
        log_probs = sum(log_probs)
        done_probs = sum(done_probs)
        # print(log_probs)
        # print(done_probs)
        return indexes, log_probs+done_probs, reward.detach()


class Critic(nn.Module):
    def __init__(self, degree, device):
        super(Critic, self).__init__()
        self.degree = degree
        self.device = device

        # embedder args
        self.d_input = 3
        self.d_model = 128

        # encoder args
        self.num_stacks = 3
        self.num_heads = 16
        self.d_k = 16
        self.d_v = 16
        self.d_inner = 512
        self.d_unit = 256

        self.crit_embedder = Embedder(degree, self.d_input, self.d_model)
        self.crit_encoder = Encoder(self.num_stacks, self.num_heads, self.d_k, self.d_v, self.d_model, self.d_inner)
        self.glimpse = Glimpse(self.d_model, self.d_unit)
        self.critic_l1 = nn.Linear(self.d_model, self.d_unit)
        self.critic_l2 = nn.Linear(self.d_unit, 1)
        self.relu = nn.ReLU()

        self.to(device)
        self.train()

    def forward(self, inputs, deterministic=False):
        # 添加0/1标记
        pad = torch.zeros([inputs.shape[0], inputs.shape[1], 1], device=self.device)
        pad[:, 0, 0] = 1 # start is 1
        #inputs = torch.tensor(inputs, dtype=torch.float).to(self.device)
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, dtype=torch.float)
        inputs = inputs.to(self.device)
        inputs = torch.cat([inputs, pad], -1)
        critic_encode = self.crit_encoder(self.crit_embedder(inputs), None)
        glimpse = self.glimpse(critic_encode)
        critic_inner = self.relu(self.critic_l1(glimpse))
        predictions = self.relu(self.critic_l2(critic_inner)).squeeze(-1)

        return predictions


if __name__ == '__main__':
    inputs = np.random.rand(10, 8, 2)
    device = torch.device("cpu")
    actor = Actor(8, device)
    a, b, reward = actor(inputs, 7, False)
    print(a)
    print(b)
    print(reward)
    # critic = Critic(4, 'cpu')
    # c = critic(inputs)
    # print(c)
