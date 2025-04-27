import torch
import numpy as np
import os
import time
from torch.autograd import Variable
from model import *
import matplotlib.pyplot as plt
from utils import *
from data_loader import *
from tqdm import tqdm


class Solver(object):
    DEFAULTS = {
        'pretrained_model': None,
        'sample_step': 100,
        'model_save_step': 100
    }

    def __init__(self, data_loader, config):
        # Data loader
        self.__dict__.update(Solver.DEFAULTS, **config)
        self.data_loader = data_loader
        self.config = config

        self.build_model()

    def build_model(self):
        # Define model
        self.dagmm = DaGMM(self.config, n_gmm=self.gmm_k)

        # Optimizers
        self.optimizer = torch.optim.Adam(self.dagmm.parameters(), lr=self.lr)

    def reset_grad(self):
        self.dagmm.zero_grad()

    def to_var(self, x, volatile=False):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, volatile=volatile)

    def train(self):
        # Start with trained model if exists
        if self.pretrained_model:
            start = int(self.pretrained_model.split("_")[0])
        else:
            start = 0

        # Start training
        iter_ctr = 0

        self.ap_global_train = np.array([0, 0, 0])
        for e in range(start, self.num_epochs):
            for i, (input_data, labels) in enumerate(tqdm(self.data_loader)):
                iter_ctr += 1
                start = time.time()

                input_data = self.to_var(input_data)

                total_loss, sample_energy, recon_error, cov_diag = self.dagmm_step(
                    input_data
                )
                # Loss
                loss = {}
                loss["total_loss"] = total_loss.data.item()
                loss["sample_energy"] = sample_energy.mean().item()
                loss["recon_error"] = recon_error.mean().item()
                loss["cov_diag"] = (
                    cov_diag.item() if isinstance(cov_diag, torch.Tensor) else cov_diag
                )

                # Save model checkpoints
                if (i + 1) % self.model_save_step == 0:
                    torch.save(
                        self.dagmm.state_dict(),
                        os.path.join(
                            self.model_save_path, "{}_{}_dagmm.pth".format(e + 1, i + 1)
                        ),
                    )

    def dagmm_step(self, input_data):
        self.dagmm.train()
        enc, dec, z, gamma = self.dagmm(input_data)

        total_loss, sample_energy, recon_error, cov_diag = self.dagmm.loss_function(
            input_data, dec, z, gamma, self.lambda_energy, self.lambda_cov_diag
        )

        self.reset_grad()

        total_loss = total_loss.mean()
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.dagmm.parameters(), 5)
        self.optimizer.step()

        return total_loss, sample_energy, recon_error, cov_diag

    def test(self):
        self.dagmm.eval()
        self.data_loader.dataset.mode = "train"

        N = 0
        mu_sum = 0
        cov_sum = 0
        gamma_sum = 0

        for it, (input_data, labels) in enumerate(self.data_loader):
            input_data = self.to_var(input_data)
            enc, dec, z, gamma = self.dagmm(input_data)
            phi, mu, cov = self.dagmm.compute_gmm_params(z, gamma)

            batch_gamma_sum = torch.sum(gamma, dim=0)

            gamma_sum += batch_gamma_sum
            mu_sum += mu * batch_gamma_sum.unsqueeze(
                -1
            )  # keep sums of the numerator only
            cov_sum += cov * batch_gamma_sum.unsqueeze(-1).unsqueeze(
                -1
            )  # keep sums of the numerator only

            N += input_data.size(0)

        train_phi = gamma_sum / N

        train_energy = []
        train_labels = []
        train_z = []
        for it, (input_data, labels) in enumerate(self.data_loader):
            input_data = self.to_var(input_data)
            enc, dec, z, gamma = self.dagmm(input_data)
            # sample_energy, cov_diag = self.dagmm.compute_energy(z, phi=train_phi, mu=train_mu, cov=train_cov, size_average=False)
            sample_energy = self.dagmm.compute_energy(z, phi=train_phi)

            train_energy.append(sample_energy.data.cpu().numpy())
            train_z.append(z.data.cpu().numpy())
            train_labels.append(labels.numpy())

        train_energy = np.concatenate(train_energy, axis=0)
        train_z = np.concatenate(train_z, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)

        self.data_loader.dataset.mode = "test"
        test_energy = []
        test_labels = []
        test_z = []
        for it, (input_data, labels) in enumerate(self.data_loader):
            input_data = self.to_var(input_data)
            enc, dec, z, gamma = self.dagmm(input_data)
            # sample_energy, cov_diag = self.dagmm.compute_energy(z, size_average=False)
            sample_energy = self.dagmm.compute_energy(z)
            test_energy.append(sample_energy.data.cpu().numpy())
            test_z.append(z.data.cpu().numpy())
            test_labels.append(labels.numpy())

        test_energy = np.concatenate(test_energy, axis=0)
        test_z = np.concatenate(test_z, axis=0)
        test_labels = np.concatenate(test_labels, axis=0)

        combined_energy = np.concatenate([train_energy, test_energy], axis=0)

        thresh = np.percentile(combined_energy, 100 - 20)
        print("Threshold :", thresh)

        pred = (test_energy > thresh).astype(int)
        gt = test_labels.astype(int)

        from sklearn.metrics import (
            precision_recall_fscore_support as prf,
            accuracy_score,
        )

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = prf(gt, pred, average="binary")

        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}".format(
                accuracy, precision, recall, f_score
            )
        )

        return accuracy, precision, recall, f_score
