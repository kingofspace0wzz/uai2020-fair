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

from model.models.model import CausalFair
from data import get_adult, get_german
from model.models.losses import kl_standard_normal

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='adult.data')
## train ##
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-03)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--critic_iter', type=int, default=10)
parser.add_argument('--wstart', type=int, default=10)
## model parameters ##
parser.add_argument('--latent_size', type=int, default=32)
parser.add_argument('--hidden_dim', type=int, default=100)
parser.add_argument('--disc_size', type=int, default=0)
## main ##
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--nogender', action='store_true')
parser.add_argument('--test_size', type=float, default=0.5)
parser.add_argument('--supervise', action='store_true')
parser.add_argument('--label', action='store_true')
parser.add_argument('--ite', action='store_true', help='ite as loss')
parser.add_argument('--weight_cliping_limit', type=float, default=0.8)
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

def train(model):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    dis_optimizer = torch.optim.Adam(critic.parameters(), lr=args.lr, betas=(0.9, 0.98))
    cont_dim = latent_spec['cont']
    disc_dim = latent_spec['disc']
    writer = SummaryWriter()
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
            for i, data in enumerate(tqdm(train_iter)):
                inputs, labels, factor = [d.to(device) for d in data]
                batch_size = inputs.size(0)
                x = torch.cat((factor, inputs), dim=-1)
                labels = labels.long().squeeze(1)
                out, z, y, q_z = model(x)
                closs = F.cross_entropy(y, labels)
                reloss = F.mse_loss(out, inputs)
                kld = kl_standard_normal(q_z)
                if args.ite and epoch >= args.wstart:
                    for _ in range(args.critic_iter):
                        for p in critic.parameters():
                            p.data.clamp_(-args.weight_cliping_limit, args.weight_cliping_limit)
                        disloss = 0
                        wloss = 0
                        # for n in range(s.size(-1)):
                        x_0 = torch.cat((torch.zeros(batch_size, 1).to(inputs.device), inputs), dim=-1)
                        z_0 = model.encode(x_0).rsample() # z ~ q(z|do(s), x)
                        dis_int = critic(z_0).mean()
                        x_1 = torch.cat((torch.ones(batch_size, 1).to(inputs.device), inputs), dim=-1)
                        z_1 = model.encode(x_1).rsample()
                        dis_real = critic(z_1).mean()
                        disloss = dis_int - dis_real
                        wloss = dis_real - dis_int
                        dis_optimizer.zero_grad()
                        disloss.backward()
                        dis_optimizer.step()
                    y_0 = model.z_to_y(z_0).mean() # p(y|do(s), x) = \int_z p(y|z)p(z|do(s), x)
                    y_1 = model.z_to_y(z_1).mean()
                    loss += wloss
                else:
                    wloss = torch.zeros(1)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                predicted = torch.max(out, dim=-1)[1]
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
            writer.add_scalar('train/reloss', re_loss)
            writer.add_scalar('train/kld', kld_loss)
            writer.add_scalar('train/wloss', w_loss)
            writer.add_scalar('train/acc', acc)
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
    correct = 0
    correct_g = 0
    size = 0
    for i, data in enumerate(test_iter):
        inputs, labels, factor = [d.to(device) for d in data]
        batch_size = inputs.size(0)
        x = torch.cat((factor, inputs), dim=-1)
        labels = labels.long().squeeze(1)
        out, z, y, q_z = model(x)
        closs = F.cross_entropy(y, labels)
        reloss = F.mse_loss(out, inputs)
        kld = kl_standard_normal(q_z)

        ys = []
        x_0 = torch.cat((torch.zeros(batch_size, 1).to(inputs.device), inputs), dim=-1)
        z_0 = model.encode(x_0).rsample() # z ~ q(z|do(s), x)
        x_1 = torch.cat((torch.ones(batch_size, 1).to(inputs.device), inputs), dim=-1)
        z_1 = model.encode(x_1).rsample()
        y_0 = model.z_to_y(z_0) # p(y|do(s), x) = \int_z p(y|z)p(z|do(s), x)
        y_1 = model.z_to_y(z_1)   

        predicted = torch.max(out, dim=-1)[1]
        correct += (predicted == labels).sum().item()
        logits = model.s_to_p(s)
        predicted_g = torch.max(logits, dim=-1)[1]
        correct_g += (predicted_g == label_g).sum().item()
        # it_estimate += (ys[0] - ys[1]).float().abs().sum()
        it_estimate += (y_0 - y_1).float().abs().sum()
        re_loss += reloss * batch_size
        size += batch_size
    it_estimate = it_estimate / size
    re_loss = re_loss / size
    acc = correct / size * 100
    acc_g = correct_g / size * 100
    print('-'*90)
    print('Test | reloss {:5.2f} | acc {:5.2f} | acc g {:5.2f} | ite {:5.4f}'.format(re_loss, acc, acc_g, it_estimate))

if __name__ == "__main__":
    if args.disc_size == 0:
        latent_spec = {'cont': args.latent_size}
    else:
        latent_spec = {'cont': args.latent_size, 'disc': args.disc_size}
    dataset = args.data.rstrip('/').split('/')[-1]
    if dataset == 'adult.data':
        train_iter, test_iter = get_adult(args.data, args.batch_size, args.nogender, args.test_size)
    else:
        train_iter, test_iter = get_german(args.data, args.batch_size, args.test_size)

    for _, (batch, _, _) in enumerate(train_iter):
        input_dim = batch.size(-1)
        break
    model = CausalFair(input_dim, args.hidden_dim, 2, latent_spec).to(device)
    critic = Critic(2, 1).to(device)
    train(model)
    evaluate(model)