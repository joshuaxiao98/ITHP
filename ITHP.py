import torch
import torch.nn as nn
import global_configs


# from global_configs import *

class ITHP(nn.Module):
    def __init__(self, ITHP_args):
        super(ITHP, self).__init__()
        TEXT_DIM, ACOUSTIC_DIM, VISUAL_DIM = (global_configs.TEXT_DIM, global_configs.ACOUSTIC_DIM,
                                              global_configs.VISUAL_DIM)

        self.X0_dim = ITHP_args['X0_dim']
        self.X1_dim = ITHP_args['X1_dim']
        self.X2_dim = ITHP_args['X2_dim']
        self.inter_dim = ITHP_args['inter_dim']
        self.drop_prob = ITHP_args['drop_prob']
        self.max_sen_len = ITHP_args['max_sen_len']
        self.B0_dim = ITHP_args['B0_dim']
        self.B1_dim = ITHP_args['B1_dim']
        self.p_beta = ITHP_args['p_beta']
        self.p_gamma = ITHP_args['p_gamma']
        self.p_lambda = ITHP_args['p_lambda']

        self.encoder1 = nn.Sequential(
            nn.Linear(self.X0_dim, self.inter_dim),
            nn.ReLU(),
            nn.Dropout(self.drop_prob),
            nn.Linear(self.inter_dim, self.B0_dim * 2),
        )

        self.MLP1 = nn.Sequential(
            nn.Linear(self.B0_dim, self.inter_dim),
            nn.ReLU(),
            nn.Dropout(self.drop_prob),
            nn.Linear(self.inter_dim, self.X1_dim),
            nn.Sigmoid(),
            nn.Dropout(self.drop_prob),
        )

        self.encoder2 = nn.Sequential(
            nn.Linear(self.B0_dim, self.inter_dim),
            nn.ReLU(),
            nn.Dropout(self.drop_prob),
            nn.Linear(self.inter_dim, self.B1_dim * 2),
        )

        self.MLP2 = nn.Sequential(
            nn.Linear(self.B1_dim, self.inter_dim),
            nn.ReLU(),
            nn.Dropout(self.drop_prob),
            nn.Linear(self.inter_dim, self.X2_dim),
            nn.Sigmoid(),
            nn.Dropout(self.drop_prob),
        )

        self.criterion = nn.MSELoss()

    def kl_loss(self, mu, logvar):
        kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        kl_mean = torch.mean(kl_div)
        return kl_mean

    def reparameterise(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)

    def forward(self, x, visual, acoustic):
        h1 = self.encoder1(x)
        mu1, logvar1 = h1.chunk(2, dim=-1)
        kl_loss_0 = self.kl_loss(mu1, logvar1)
        b0 = self.reparameterise(mu1, logvar1)
        output1 = self.MLP1(b0)
        mse_0 = self.criterion(output1, acoustic)
        IB0 = kl_loss_0 + self.p_beta * mse_0

        b0_sample = b0
        h2 = self.encoder2(b0_sample)
        mu2, logvar2 = h2.chunk(2, dim=-1)
        kl_loss_1 = self.kl_loss(mu2, logvar2)
        b1 = self.reparameterise(mu2, logvar2)
        output2 = self.MLP2(b1)
        mse_1 = self.criterion(output2, visual)
        IB1 = kl_loss_1 + self.p_gamma * mse_1
        IB_total = IB0 + self.p_lambda * IB1

        return b1, IB_total, kl_loss_0, mse_0, kl_loss_1, mse_1
