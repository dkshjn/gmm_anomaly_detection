import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import itertools
from utils import *
import math


class Cholesky(torch.autograd.Function):
    def forward(ctx, a):
        l = torch.linalg.cholesky(a)
        ctx.save_for_backward(l)
        return l

    def backward(ctx, grad_output):
        (l,) = ctx.saved_variables
        linv = l.inverse()
        inner = torch.tril(torch.mm(l.t(), grad_output)) * torch.tril(
            1.0 - Variable(l.data.new(l.size(1)).fill_(0.5).diag())
        )
        s = torch.mm(linv.t(), torch.mm(inner, linv))
        return s


class DaGMM(nn.Module):
    """Residual Block."""

    def __init__(self, hparams, n_gmm=2, latent_dim=3):
        super(DaGMM, self).__init__()
        input_dim = hparams["input_dim"]

        layers = []
        layers += [nn.Linear(input_dim, 60)]
        layers += [nn.Tanh()]
        layers += [nn.Linear(60, 30)]
        layers += [nn.Tanh()]
        layers += [nn.Linear(30, 10)]
        layers += [nn.Tanh()]
        layers += [nn.Linear(10, 1)]

        self.encoder = nn.Sequential(*layers)

        layers = []
        layers += [nn.Linear(1, 10)]
        layers += [nn.Tanh()]
        layers += [nn.Linear(10, 30)]
        layers += [nn.Tanh()]
        layers += [nn.Linear(30, 60)]
        layers += [nn.Tanh()]
        layers += [nn.Linear(60, input_dim)]

        self.decoder = nn.Sequential(*layers)

        layers = []
        layers += [nn.Linear(latent_dim, 10)]
        layers += [nn.Tanh()]
        layers += [nn.Dropout(p=0.5)]
        layers += [nn.Linear(10, n_gmm)]
        layers += [nn.Softmax(dim=1)]

        self.estimation = nn.Sequential(*layers)

        self.register_buffer("phi", torch.zeros(n_gmm))
        self.register_buffer("mu", torch.zeros(n_gmm, latent_dim))
        self.register_buffer("cov", torch.zeros(n_gmm, latent_dim, latent_dim))

        self.dist_type = hparams["dist_type"]
        self.student_nu = hparams["student_nu"]

    def relative_euclidean_distance(self, a, b):
        return (a - b).norm(2, dim=1) / a.norm(2, dim=1)

    def forward(self, x):
        enc = self.encoder(x)

        dec = self.decoder(enc)

        rec_cosine = F.cosine_similarity(x, dec, dim=1)
        rec_euclidean = self.relative_euclidean_distance(x, dec)

        z = torch.cat(
            [enc, rec_euclidean.unsqueeze(-1), rec_cosine.unsqueeze(-1)], dim=1
        )

        gamma = self.estimation(z)

        return enc, dec, z, gamma

    def compute_gmm_params(self, z, gamma):
        N = gamma.size(0)
        # K
        sum_gamma = torch.sum(gamma, dim=0)

        # K
        phi = sum_gamma / N

        self.phi = phi.data

        # K x D
        mu = torch.sum(
            gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0
        ) / sum_gamma.unsqueeze(-1)
        self.mu = mu.data

        # z_mu = N x K x D
        z_mu = z.unsqueeze(1) - mu.unsqueeze(0)

        # z_mu_outer = N x K x D x D
        z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)

        # K x D x D
        cov = torch.sum(
            gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer, dim=0
        ) / sum_gamma.unsqueeze(-1).unsqueeze(-1)
        self.cov = cov.data

        return phi, mu, cov

    def _component_log_prob(self, z, mu, cov_inv, log_det_cov):
        delta = z - mu  # B x K x D
        m_dist = torch.einsum("bkd,kde,bke->bk", delta, cov_inv, delta)

        if self.dist_type == "gaussian":
            log_norm = -0.5 * (mu.size(-1) * math.log(2 * math.pi) + log_det_cov)
            return log_norm - 0.5 * m_dist

        elif self.dist_type == "laplace":
            scale = torch.sqrt(
                torch.diagonal(cov_inv.inverse(), dim1=-2, dim2=-1) / 2.0
            )
            l1_norm = torch.sum(torch.abs(delta) / scale, dim=-1)
            log_norm = -(
                mu.size(-1) * math.log(2.0) + torch.sum(torch.log(scale), dim=-1)
            )
            return log_norm - l1_norm

        elif self.dist_type == "student_t":
            ν = self.student_nu
            d = mu.size(-1)
            ν_tensor = torch.tensor(
                ν, dtype=log_det_cov.dtype, device=log_det_cov.device
            )
            d_tensor = torch.tensor(
                d, dtype=log_det_cov.dtype, device=log_det_cov.device
            )

            log_norm = (
                torch.lgamma((ν_tensor + d_tensor) / 2)
                - torch.lgamma(ν_tensor / 2)
                - 0.5 * (d_tensor * math.log(ν * math.pi) + log_det_cov)
            )

            return log_norm - 0.5 * (ν + d) * torch.log1p(m_dist / ν)

        else:
            raise ValueError(f"Unknown dist_type {self.dist_type}")

    def compute_energy(self, z, phi=None):
        z = z.unsqueeze(1)  # B x 1 x D
        phi = phi if phi is not None else self.phi  # K
        log_phi = torch.log(phi + 1e-12)

        cov_inv = torch.inverse(self.cov)  # K x D x D
        log_det = torch.logdet(
            self.cov + 1e-6 * torch.eye(z.size(-1), device=z.device)
        )  # K

        log_pdf = self._component_log_prob(z, self.mu, cov_inv, log_det)  # B x K
        log_mix = torch.logsumexp(log_phi + log_pdf, dim=1)  # B

        return -log_mix

    def loss_function(self, x, x_hat, z, gamma, lambda_energy, lambda_cov_diag):
        recon_error = torch.mean((x - x_hat) ** 2)

        phi, mu, cov = self.compute_gmm_params(z, gamma)

        sample_energy = self.compute_energy(z, phi)
        cov_diag = torch.sum(1.0 / self.cov.diagonal(dim1=-2, dim2=-1))

        loss = recon_error + lambda_energy * sample_energy + lambda_cov_diag * cov_diag

        return loss, sample_energy, recon_error, cov_diag
