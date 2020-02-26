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

from model.models.model import FairVAE, CausalFair
from data import get_adult, get_german, get_crime
from model.models.losses import kl_standard_normal, kl_normal_normal
from model.models.losses import compute_mmd

from sinkhorn import sinkhorn_loss_primal

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='adult.data')
## train ##
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--lr', type=float, default=1e-03)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--critic_iter', type=int, default=10)
parser.add_argument('--wstart', type=int, default=10)
## model parameters ##
parser.add_argument('--latent_size', type=int, default=32)
parser.add_argument('--z1_dim', type=int, default=50)
parser.add_argument('--z2_dim', type=int, default=50)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--disc_size', type=int, default=0)
## main ##
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--nocuda', action='store_true')
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--nogender', action='store_true')
parser.add_argument('--test_size', type=float, default=0.5)
parser.add_argument('--supervise', action='store_true')
parser.add_argument('--label', action='store_true')
parser.add_argument('--ite', action='store_true', help='ite as loss')
parser.add_argument('--weight_cliping_limit', type=float, default=0.8)
parser.add_argument('--log', type=str, default='results/vfae.txt')
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(args.seed)

device = 'cuda' if torch.cuda.is_available() and args.nocuda is False else 'cpu'
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

def train(model):
    model.train()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    dis_optimizer = torch.optim.RMSprop(critic.parameters(), lr=args.lr)
    writer = SummaryWriter("runs/fair")
    print('Start training.')
    try:
        for epoch in range(args.epochs):
            re_loss = 0
            c_loss = 0
            size = 0
            correct = 0
            correct_g = 0
            kld_loss = 0
            kl2_loss = 0
            mmd_loss = 0
            w_loss = 0
            for i, data in enumerate(tqdm(train_iter)):
                data_iter = iter(train_iter)
                inputs, labels, factor = [d.to(device) for d in data]
                batch_size = inputs.size(0)
                x = torch.cat((factor, inputs), dim=-1)
                
                # out, z1, z2, z1_z2y, logit, probs = model(inputs, factor, labels)
                # labels = labels.long().squeeze(1)
                
                # # reloss = F.mse_loss(out, inputs)    # mean
                # reloss = F.binary_cross_entropy_with_logits(out, inputs)
                # kld_z1 = kl_normal_normal(probs['z1_xu'], probs['z1_z2y'])
                # kld_z2 = kl_standard_normal(probs['z2_z1y'])    # mean
                # closs = F.cross_entropy(logit, labels)  # mean
                # z1_0 = model._q_z1_xu(inputs, torch.zeros_like(factor))
                # z1_1 = model._q_z1_xu(inputs, torch.ones_like(factor))
                # mmd = compute_mmd(z1_0.rsample(), z1_1.rsample())
                # loss = reloss + kld_z2 - kld_z1 + closs + 5 * mmd # (z1.log().mean() - z1_z2y.log().mean()) does not work
                # print('| reloss: {:5.2f} | kld: {:5.2f} | closs {:5.4f} | mmd {:5.2f} '.format(reloss, kld_z2, closs, mmd))
                
                labels = labels.long().squeeze(1)
                out, z, y, q_z = model(x)
                closs = F.cross_entropy(y, labels)  # mean
                reloss = F.mse_loss(out*255, x*255, reduction='mean') / 255 # mean
                kld = kl_standard_normal(q_z)   # mean
                z_0 = model.encode(torch.cat((torch.zeros(inputs.size(0), 1).to(device), inputs), dim=-1)).rsample()
                z_1 = model.encode(torch.cat((torch.ones(inputs.size(0), 1).to(device), inputs), dim=-1)).rsample()
                mmd = compute_mmd(z_0, z_1)
                loss = 1* closs + 1 * reloss + 1 * kld + 1 * batch_size * mmd 
                
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
                c_loss += closs.item() * batch_size
                mmd_loss += mmd.item() * batch_size

            re_loss = re_loss / size
            c_loss /= size
            kld_loss /= size
            mmd_loss /= size
            acc = correct / size * 100
            print('-'*90)
            print('Epoch: {:3d} | reloss: {:5.2f} | kld: {:5.2f} | closs {:5.4f} | mmd {:5.2f} | acc {:5.2f}'.format(epoch, re_loss, kld_loss, c_loss, mmd_loss, acc))
            # writer.add_scalar('train/reloss', re_loss, epoch * len(train_iter)+i)
            # writer.add_scalar('train/kld', kld_loss, epoch * len(train_iter)+i)
            # writer.add_scalar('train/wloss', w_loss, epoch * len(train_iter)+i)
            # writer.add_scalar('train/acc', acc, epoch * len(train_iter)+i)
        # for epoch in range(args.epochs):


    except KeyboardInterrupt:
        print('-' * 90)
        print('Exiting from training early')

def evaluate(model):
    model.eval()
    print('Start evaluation.')
    
    re_loss = 0
    it_estimate = 0
    correct = 0
    correct_g = 0
    size = 0
    for i, data in enumerate(test_iter):
        inputs, labels, factor = [d.to(device) for d in data]
        batch_size = inputs.size(0)
        x = torch.cat((factor, inputs), dim=-1)
        
        # out, z1, z2, z1_z2y, logit, probs = model(inputs, factor, labels)
        # labels_ = labels.long().squeeze(1)
        
        # reloss = F.mse_loss(out, inputs)    # mean
        # kld_z2 = kl_standard_normal(probs['z2_z1y'])    # mean
        # closs = F.cross_entropy(logit, labels_)  # mean
        # z1_0 = model._q_z1_xu(inputs, torch.zeros_like(factor))
        # z1_1 = model._q_z1_xu(inputs, torch.ones_like(factor))
        # mmd = compute_mmd(z1_0.rsample(), z1_1.rsample())
        # loss = reloss + kld_z2 + closs + (z1.log().mean() - z1_z2y.log().mean()) + mmd
              

        labels = labels.long().squeeze(1)
        out, z, y, q_z = model(x)
        closs = F.cross_entropy(y, labels)  # mean
        reloss = F.mse_loss(out*255, x*255, reduction='mean') / 255 # mean
        kld = kl_standard_normal(q_z)   # mean
        z_0 = model.encode(torch.cat((torch.zeros(inputs.size(0), 1).to(device), inputs), dim=-1)).rsample()
        z_1 = model.encode(torch.cat((torch.ones(inputs.size(0), 1).to(device), inputs), dim=-1)).rsample()
        mmd = compute_mmd(z_0, z_1)
        loss = 0 * closs + reloss + kld + 5 * mmd 

        ys = []
        # _, _, _, _, y_0, _ = model(inputs, torch.zeros_like(factor), labels)
        # _, _, _, _, y_1, _ = model(inputs, torch.ones_like(factor), labels)

        predicted = torch.max(y, dim=-1)[1]
        correct += (predicted == labels).sum().item()
        re_loss += reloss * batch_size
        size += batch_size
    
    model = model.to('cpu')
    data = test_iter.dataset
    inputs, labels, factors = data[:]
    
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

    it_estimate = it_estimate / size
    re_loss = re_loss / size
    acc = correct / size * 100
    # acc_g = correct_g / size * 100
    print('-'*90)
    print('Test | reloss {:5.2f} | acc {:5.2f} | ite {:5.4f} | dp {:5.2f}'.format(re_loss, acc, it_estimate, dem_parity))
    with open(args.log, 'a') as fd:
        print('data {} | reloss {:5.2f} | acc {:5.2f} | ite {:5.4f} | dp {:5.2f}'.format(args.data, re_loss, acc, it_estimate, dem_parity), file=fd)
    
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
    model = CausalFair(input_dim+1, args.hidden_dim, 2, latent_spec).to(device)
    code_size = args.latent_size + args.disc_size
    critic = Critic(code_size, 1).to(device)
    train(model)
    evaluate(model)