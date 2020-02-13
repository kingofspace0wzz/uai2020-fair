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

from model.models.model import FairVAE
from data import get_adult, get_german
from model.models.losses import kl_standard_normal, kl_normal_normal
from model.models.losses import compute_mmd

from sinkhorn import sinkhorn_loss_primal

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
parser.add_argument('--z1_dim', type=int, default=50)
parser.add_argument('--z2_dim', type=int, default=50)
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
                
                out, z1, z2, z1_z2y, logit, probs = model(inputs, factor, labels)
                labels = labels.long().squeeze(1)
                
                # reloss = F.mse_loss(out, inputs)    # mean
                reloss = F.binary_cross_entropy_with_logits(out, inputs)
                kld_z1 = kl_normal_normal(probs['z1_xu'], probs['z1_z2y'])
                kld_z2 = kl_standard_normal(probs['z2_z1y'])    # mean
                closs = F.cross_entropy(logit, labels)  # mean
                z1_0 = model._q_z1_xu(inputs, torch.zeros_like(factor))
                z1_1 = model._q_z1_xu(inputs, torch.ones_like(factor))
                mmd = compute_mmd(z1_0.rsample(), z1_1.rsample())
                loss = reloss + kld_z2 + kld_z1 + closs + mmd # (z1.log().mean() - z1_z2y.log().mean()) does not work
                # print('| reloss: {:5.2f} | kld: {:5.2f} | closs {:5.4f} | mmd {:5.2f} '.format(reloss, kld_z2, closs, mmd))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                predicted = torch.max(logit, dim=-1)[1]
                correct += (predicted == labels).sum().item()
                # predicted_g = torch.max(logits, dim=-1)[1]
                # correct_g += (predicted_g == label_g).sum().item()
                size += batch_size
                re_loss += reloss.item() * batch_size 
                kld_loss += kld_z2.item() * batch_size
                c_loss += closs.item() * batch_size
                mmd_loss += mmd.item() * batch_size

            re_loss = re_loss / size
            c_loss /= size
            kld_loss /= size
            mmd_loss /= size
            acc = correct / size * 100
            print('-'*90)
            print('Epoch: {:3d} | reloss: {:5.2f} | kld: {:5.2f} | closs {:5.4f} | mmd {:5.2f} | acc {:5.2f}'.format(epoch, re_loss, kld_loss, c_loss, mmd_loss, acc))
            writer.add_scalar('train/reloss', re_loss, epoch * len(train_iter)+i)
            writer.add_scalar('train/kld', kld_loss, epoch * len(train_iter)+i)
            writer.add_scalar('train/wloss', w_loss, epoch * len(train_iter)+i)
            writer.add_scalar('train/acc', acc, epoch * len(train_iter)+i)
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
        
        out, z1, z2, z1_z2y, logit, probs = model(inputs, factor, labels)
        labels_ = labels.long().squeeze(1)
        
        reloss = F.mse_loss(out, inputs)    # mean
        kld_z2 = kl_standard_normal(probs['z2_z1y'])    # mean
        closs = F.cross_entropy(logit, labels_)  # mean
        z1_0 = model._q_z1_xu(inputs, torch.zeros_like(factor))
        z1_1 = model._q_z1_xu(inputs, torch.ones_like(factor))
        mmd = compute_mmd(z1_0.rsample(), z1_1.rsample())
        loss = reloss + kld_z2 + closs + (z1.log().mean() - z1_z2y.log().mean()) + mmd
              
        ys = []
        _, _, _, _, y_0, _ = model(inputs, torch.zeros_like(factor), labels)
        _, _, _, _, y_1, _ = model(inputs, torch.ones_like(factor), labels)

        predicted = torch.max(logit, dim=-1)[1]
        correct += (predicted == labels_).sum().item()
        # logits = model.s_to_p(s)
        # predicted_g = torch.max(logits, dim=-1)[1]
        # correct_g += (predicted_g == label_g).sum().item()
        # it_estimate += (ys[0] - ys[1]).float().abs().sum()
        it_estimate += (y_0 - y_1).float().abs().sum()
        re_loss += reloss * batch_size
        size += batch_size
    it_estimate = it_estimate / size
    re_loss = re_loss / size
    acc = correct / size * 100
    # acc_g = correct_g / size * 100
    print('-'*90)
    print('Test | reloss {:5.2f} | acc {:5.2f} | ite {:5.4f}'.format(re_loss, acc, it_estimate))

if __name__ == "__main__":
    
    latent_spec = {'z1': args.z1_dim, 'z2': args.z2_dim}
    dataset = args.data.rstrip('/').split('/')[-1]
    if dataset == 'adult.data':
        train_iter, test_iter = get_adult(args.data, args.batch_size, args.nogender, args.test_size)
    else:
        train_iter, test_iter = get_german(args.data, args.batch_size, args.test_size)

    for _, (batch, _, _) in enumerate(train_iter):
        input_dim = batch.size(-1)
        break
    model = FairVAE(input_dim, args.hidden_dim, 2, latent_spec).to(device)
    code_size = args.latent_size + args.disc_size
    critic = Critic(code_size, 1).to(device)
    train(model)
    evaluate(model)