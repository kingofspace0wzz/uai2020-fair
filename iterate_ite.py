import argparse
import time
import random
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import RandomSampler

from model.models.model import CausalFair
from data import get_adult, get_german, get_crime, get_bank
from model.models.losses import kl_standard_normal

from sinkhorn import sinkhorn_loss_primal, sinkhorn_loss_dual

from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='adult.data')
## train ##
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-03)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--critic_iter', type=int, default=10)
parser.add_argument('--wstart', type=int, default=10)
parser.add_argument('--alpha', type=float, default=1.)
## model parameters ##
parser.add_argument('--latent_size', type=int, default=32)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--disc_size', type=int, default=0)
parser.add_argument('--det', action='store_true')
parser.add_argument('--critic_dim', type=int, default=5)
## main ##
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--nogender', action='store_true')
parser.add_argument('--test_size', type=float, default=0.5)
parser.add_argument('--supervise', action='store_true')
parser.add_argument('--label', action='store_true')
parser.add_argument('--ite', action='store_true', help='ite as loss')
parser.add_argument('--weight_cliping_limit', type=float, default=0.8)
parser.add_argument('--niter', type=int, default=20)
parser.add_argument('--eps', type=float, default=1.)
parser.add_argument('--log', type=str, default='results/log.txt')
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(args.seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.cuda.set_device(args.cuda)

class Critic(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim=100):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.critic(x)

def train(model, ite):
    model.train()
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    # dis_optimizer = torch.optim.Adam(critic.parameters(), lr=args.lr, betas=(0.9, 0.98))
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    dis_optimizer = torch.optim.RMSprop(critic.parameters(), lr=args.lr)
   
    cont_dim = latent_spec['cont']
    disc_dim = latent_spec['disc']
    writer = SummaryWriter("runs/")
    print('Start training.')
    try:
        for epoch in range(args.epochs):
            re_loss = 0
            re_loss2 = 0
            size = 0
            correct = 0
            correct_g = 0
            kld_loss = 0
            kl2_loss = 0
            mmd_loss = 0
            w_loss = 0
            it_estimate = 0
            for i, data in enumerate(tqdm(train_iter)):
                # data_iter = iter(train_iter)
                data_iter = train_iter.dataset
                inputs, labels, factor = [d.to(device) for d in data]
                batch_size = inputs.size(0)
                x = torch.cat((factor, inputs), dim=-1)
                labels = labels.long().squeeze(1)
                out, z, y, q_z = model(x)
                closs = F.cross_entropy(y, labels)  # mean
                reloss = F.mse_loss(out*255, x*255, reduction='mean') / 255 # mean
                kld = kl_standard_normal(q_z)   # mean
                loss = closs 
                if ite and epoch >= args.wstart:
                    for _ in range(args.critic_iter):
                        for p in critic.parameters():
                            p.data.clamp_(-args.weight_cliping_limit, args.weight_cliping_limit)
                
                        # inputs_next = data_iter.next()[0].to(device)
                        inputs_next = data_iter[list(sampler)[0]][0].to(device)
                        # print(inputs_next)
                        x_0 = torch.cat((torch.zeros(inputs_next.size(0), 1).to(inputs.device), inputs_next), dim=-1)
                        z_0 = model.encode(x_0).rsample() # z ~ q(z|do(s), x)
                        dis_int = critic(z_0)
                        x_1 = torch.cat((torch.ones(inputs_next.size(0), 1).to(inputs.device), inputs_next), dim=-1)
                        z_1 = model.encode(x_1).rsample()
                        dis_real = critic(z_1)
                        sink_f = 2*sinkhorn_loss_dual(dis_int, dis_real, args.eps, inputs_next.size(0), args.niter) \
                                - sinkhorn_loss_dual(dis_int, dis_int, args.eps, inputs_next.size(0), args.niter) \
                                - sinkhorn_loss_dual(dis_real, dis_real, args.eps, inputs_next.size(0), args.niter)
                       
                        dis_optimizer.zero_grad()
                        sink_f.backward(retain_graph=True)
                        dis_optimizer.step()

                    y_0 = model.z_to_y(z_0).mean() # p(y|do(s), x) = \int_z p(y|z)p(z|do(s), x)
                    y_1 = model.z_to_y(z_1).mean()
                    # loss += wloss

                    # inputs_next = data_iter.next()[0].to(device)
                    inputs_next = data_iter[list(sampler)[0]][0].to(device)
                    x_0 = torch.cat((torch.zeros(inputs_next.size(0), 1).to(inputs.device), inputs_next), dim=-1)
                    z_0 = model.encode(x_0).rsample() # z ~ q(z|do(s), x)
                    dis_int = critic(z_0)
                    x_1 = torch.cat((torch.ones(inputs_next.size(0), 1).to(inputs.device), inputs_next), dim=-1)
                    z_1 = model.encode(x_1).rsample()
                    dis_real = critic(z_1)
                    wloss = 2*sinkhorn_loss_dual(dis_int, dis_real, args.eps, inputs_next.size(0), args.niter) \
                            - sinkhorn_loss_dual(dis_int, dis_int, args.eps, inputs_next.size(0), args.niter) \
                            - sinkhorn_loss_dual(dis_real, dis_real, args.eps, inputs_next.size(0), args.niter)
                    # print(wloss)
                    loss += args.alpha * wloss
                else:
                    wloss = torch.zeros(1)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                predicted = torch.max(y, dim=-1)[1]
                correct += (predicted == labels).sum().item()
                # predicted_g = torch.max(logits, dim=-1)[1]
                # correct_g += (predicted_g == label_g).sum().item()
                size += batch_size
                re_loss += reloss.item() * batch_size
                kld_loss += kld.item() * batch_size 
                w_loss += wloss.item() * batch_size
            re_loss = re_loss / size
            re_loss2 /= size
            kld_loss /= size
            mmd_loss /= size
            w_loss /= size
            acc = correct / size * 100
            print('-'*90)
            print('Epoch: {:3d} | reloss: {:5.2f} | kld: {:5.2f} | wloss {:5.4f} | acc {:5.2f}'.format(epoch, re_loss, kld, w_loss, acc))
            
            model = model.to('cpu')
            data = test_iter.dataset
            inputs, _, factors = data[:]
            x_0 = torch.cat((torch.zeros(inputs.size(0), 1), inputs), dim=-1)
            x_1 = torch.cat((torch.ones(inputs.size(0), 1), inputs), dim=-1)
            z_0 = model.encode(x_0).sample()
            z_1 = model.encode(x_1).sample()
            y_0 = model.z_to_y(z_0)
            y_1 = model.z_to_y(z_1)
            y_0 = torch.max(y_0, dim=-1)[1]
            y_1 = torch.max(y_1, dim=-1)[1]
            it_estimate += (y_0 - y_1).float().abs().sum()
            it_estimate = it_estimate / inputs.size(0)
            model = model.to(device)
            if ite:
                writer.add_scalars('ACE {}'.format(args.data), {'ours': it_estimate}, epoch * len(train_iter)+i)
                writer.close()
                ite2.append(it_estimate)
            else:
                writer.add_scalars('ACE {}'.format(args.data), {'baseline': it_estimate}, epoch * len(train_iter)+i)
                ite1.append(it_estimate)
            print('ite: {:5.2f}'.format(it_estimate))
        # for epoch in range(args.epochs):


    except KeyboardInterrupt:
        print('-' * 90)
        print('Exiting from training early')

def evaluate(model):
    model.eval()
    print('Start evaluation.')
    disc_dim = latent_spec['disc']
    re_loss = 0
    it_estimate = 0
    it_estimate_z = 0
    correct = 0
    correct_g = 0
    w_loss = 0
    size = 0
    for i, data in enumerate(test_iter):
        data_iter = iter(test_iter)
        inputs, labels, factor = [d.to(device) for d in data]
        batch_size = inputs.size(0)
        x = torch.cat((factor, inputs), dim=-1)
        labels = labels.long().squeeze(1)
        out, z, y, q_z = model(x)
        closs = F.cross_entropy(y, labels)
        reloss = F.mse_loss(out, x)
        kld = kl_standard_normal(q_z)

        predicted = torch.max(y, dim=-1)[1]
        correct += (predicted == labels).sum().item()
        
        re_loss += reloss * batch_size
        # w_loss += wloss * batch_size
        size += batch_size
    model = model.to('cpu')
    data = test_iter.dataset
    inputs, _, factors = data[:]
    
    # for i in range(100):
    x_0 = torch.cat((torch.zeros(inputs.size(0), 1), inputs), dim=-1)
    x_1 = torch.cat((torch.ones(inputs.size(0), 1), inputs), dim=-1)
    z_0 = model.encode(x_0).sample()
    z_1 = model.encode(x_1).sample()
    y_0 = model.z_to_y(z_0)
    y_1 = model.z_to_y(z_1)
    
    # print(y_0)
    y_0 = torch.max(y_0, dim=-1)[1]
    y_1 = torch.max(y_1, dim=-1)[1]
    it_estimate += (y_0 - y_1).float().abs().sum()

    nonzero = torch.nonzero(factors)
    nonone = torch.nonzero(1 - factors)
    _, _, y, _ = model(torch.cat((factors, inputs), dim=-1))
    y_one = y[nonzero]
    y_zero = y[nonone]
    dem_parity = (y_one.sum() / factors.sum() - y_zero.sum() / (1 - (1-factors).sum())).abs()

    it_estimate = it_estimate / inputs.size(0)
    re_loss = re_loss / size
    w_loss = w_loss / size
    acc = correct / size * 100
    # acc_g = correct_g / size * 100
    print('-'*90)
    print('Test | reloss {:5.2f} | wloss {:5.2f} | acc {:5.2f} | ite {:5.4f} | dp {:5.4f}'.format(re_loss, w_loss, acc, it_estimate, dem_parity))
    with open(args.log, 'a') as fd:
        print('Test | reloss {:5.2f} | wloss {:5.2f} | acc {:5.2f} | ite {:5.4f} | dp {:5.4f}'.format(re_loss, w_loss, acc, it_estimate, dem_parity), file=fd)

if __name__ == "__main__":
    
    latent_spec = {'cont': args.latent_size, 'disc': args.disc_size}
    dataset = args.data.rstrip('/').split('/')[-1]
    if dataset == 'adult.data':
        train_iter, test_iter = get_adult(args.data, args.batch_size, args.nogender, args.test_size)
    elif dataset == 'german.data':
        train_iter, test_iter = get_german(args.data, args.batch_size, args.test_size)
    elif dataset == 'communities.data':
        train_iter, test_iter = get_crime(args.data, args.batch_size, args.test_size)
    elif dataset == 'bank.csv':
        train_iter, test_iter = get_bank(args.data, args.batch_size, args.test_size)

    for _, (batch, _, _) in enumerate(train_iter):
        input_dim = batch.size(-1)
        break
    # dataset = train_iter.dataset
    sampler = torch.utils.data.BatchSampler(RandomSampler(range(len(train_iter.dataset))), batch_size=args.batch_size, drop_last=False)
    model = CausalFair(input_dim+1, args.hidden_dim, 2, latent_spec, args.det).to(device)
    code_size = args.latent_size + args.disc_size
    critic = Critic(code_size, args.critic_dim).to(device)
    ite1, ite2 = [], []
    train(model, False)
    model = CausalFair(input_dim+1, args.hidden_dim, 2, latent_spec, args.det).to(device)
    code_size = args.latent_size + args.disc_size
    critic = Critic(code_size, args.critic_dim).to(device)
    train(model, True)
    # evaluate(model)
    print(len(ite1))
    print(len(ite2))
    df = pd.DataFrame({'x': range(args.epochs), 'baseline':ite1, 'ours':ite2})
    palette = plt.get_cmap('Set1')

    num = 0
    for column in df.drop('x', axis=1):
        num += 1
        plt.plot(df['x'], df[column], marker='', color=palette(num),
                 linewidth=1, alpha=0.9, label=column)

    plt.legend(loc=2, ncol=2, fontsize='xx-large')

    plt.xlabel("Epoch", fontsize='xx-large')
    plt.ylabel("ACE", fontsize='xx-large')
    plt.title("ACE vs Epoch (Adult)", fontsize='xx-large')

    plt.savefig('results/iterate-ite-adult.png', bbox_inches='tight')