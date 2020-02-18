import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal

class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, latent_spec, det=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.latent_spec = latent_spec
        self.det = det
        self.is_continuous = latent_spec['cont']
        self.is_discrete = latent_spec['disc']
        if self.is_continuous:
            self.cont_dim = latent_spec['cont']
        self.disc_dim = 0
        if self.is_discrete:
            self.disc_dim = latent_spec['disc']
            self.num_disc_latents = len(latent_spec)
        self.code_dim = self.cont_dim + self.disc_dim
        self.fcmu = nn.Linear(self.cont_dim, self.cont_dim)
        self.fcvar = nn.Linear(self.cont_dim, self.cont_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.code_dim),
            # nn.Dropout(0.2),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.code_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x, labels=None):
        q_z = self.encode(x)
        if self.det:
            z = q_z.mean
        else:
            z = q_z.rsample()
        out = self.decoder(z)
        return out, z

    def encode(self, x):
        h = self.encoder(x)
        mu, lv = self.fcmu(h), self.fcvar(h)
        return Normal(mu, (0.5 * lv).exp())
  
class CausalFair(Model):
    def __init__(self, input_dim, hidden_dim, out_dim, latent_spec, det=False):
        super().__init__(input_dim, hidden_dim, out_dim, latent_spec, det)
        self.z_to_y = nn.Sequential(
            nn.Linear(self.code_dim, hidden_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_dim, out_dim),
        )
        self.z_to_s = nn.Linear(self.code_dim, self.disc_dim)

    def forward(self, x):
        q_z = self.encode(x)
        if self.det:
            z = q_z.mean
        else:
            z = q_z.rsample()
        y = self.z_to_y(z)
        out = self.decoder(z)
        return out, z, y, q_z

# class GeneralFair(nn.Module):
#     '''
#     We need q(z|x, u), p(u|z)
#     '''
#     def __init__():
        
#         super().__init__()
        

#     def forward(self, x, u):
#         pass

class FairVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, latent_spec, det=False):
        super().__init__()
        self.z1_dim = latent_spec['z1']
        self.z2_dim = latent_spec['z2']
        self.out_dim = out_dim
        self.xu_mu1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, self.z1_dim)
        )
        self.xu_logvar1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, self.z1_dim)
        )
        self.xu_to_mu1 = nn.Sequential(
            nn.Linear(input_dim+1, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, self.z1_dim)
        )
        self.xu_to_logvar1 = nn.Sequential(
            nn.Linear(input_dim+1, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, self.z1_dim)
        )
        self.z1_to_logit = nn.Linear(self.z1_dim, out_dim)
        self.z1y_to_mu2 = nn.Sequential(
            nn.Linear(self.z1_dim+1, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, self.z2_dim)
        )
        self.z1y_to_logvar2 = nn.Sequential(
            nn.Linear(self.z1_dim+1, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, self.z2_dim)
        )
        self.z2y_to_mu1 = nn.Sequential(
            nn.Linear(self.z2_dim+1, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, self.z1_dim)
        )
        self.z2y_to_logvar1 = nn.Sequential(
            nn.Linear(self.z2_dim+1, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, self.z1_dim)
        )
        self.z1u_to_x = nn.Sequential(
            nn.Linear(self.z1_dim+1, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def _q_z1_xu(self, x, u):    # q(z|x, u)
        mu = self.xu_to_mu1(torch.cat((u, x), dim=-1))
        logvar = self.xu_to_logvar1(torch.cat((u, x), dim=-1))
        return Normal(mu, (0.5 * logvar).exp())

    def _q_y_z1(self, z1):   # q(y|z1)
        logit = self.z1_to_logit(z1)
        return logit

    def _q_z2_z1y(self, z1, y): # q(z2|z1, y)
        mu = self.z1y_to_mu2(torch.cat((z1, y), dim=-1))
        logvar = self.z1y_to_logvar2(torch.cat((z1, y), dim=-1))
        return Normal(mu, (0.5 * logvar).exp())

    def _q_z1_z2y(self, z2, y):  # q(z1|z2, y)
        mu = self.z2y_to_mu1(torch.cat((z2, y), dim=-1))
        logvar = self.z2y_to_logvar1(torch.cat((z2, y), dim=-1))
        return Normal(mu, (0.5 * logvar).exp())

    def decoder(self, z1, u):
        out = self.z1u_to_x(torch.cat((z1, u), dim=-1))
        return out

    def forward(self, x, u, y): # x: input, u: sensitive, y: label
        q_z1_xu = self._q_z1_xu(x, u)
        z1 = q_z1_xu.rsample()

        logit = self._q_y_z1(z1)

        q_z2_z1y = self._q_z2_z1y(z1, y)
        z2 = q_z2_z1y.rsample()
        
        q_z1_z2y = self._q_z1_z2y(z2, y)
        z1_z2y = q_z1_z2y.rsample()
        
        out = self.decoder(z1, u)
        
        probs = {"z1_xu":q_z1_xu, "z2_z1y":q_z2_z1y, "z1_z2y":q_z1_z2y}

        return out, z1, z2, z1_z2y, logit, probs

# class SCM(Model):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x):
#         z = self.encoder(x)
#         p_ez = Normal(torch.zeros_like(z), torch.ones_like(z))
#         ez = p_ez.sample()
#         z = z + ez
#         return z, p_ez