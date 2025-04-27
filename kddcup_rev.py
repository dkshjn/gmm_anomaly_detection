import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from model import DaGMM
from solver import Solver

class KDDCupRevLoader(Dataset):
    def __init__(self, data_path, mode="train"):
        # Load raw KDD Cup 99 data (10% subset expected)
        column_names = list(range(42))  # KDD 99 has 41 features + label
        data = pd.read_csv(data_path, names=column_names)  # comma-separated values
        # Separate features and label
        features = data.iloc[:, :-1]
        labels_text = data.iloc[:, -1]
        # Binary label: 0 for normal, 1 for attack
        labels = (labels_text != 'normal.').astype(int).values
        # One-hot encode categorical features (protocol type, service, flag)
        features = pd.get_dummies(features, columns=[1, 2, 3])
        features = features.values
        # Construct KDDCup-Rev: include all normals and a subset of attacks (20% anomaly ratio)
        normal_data = features[labels == 0]; normal_labels = labels[labels == 0]
        attack_data = features[labels == 1]; attack_labels = labels[labels == 1]
        # Determine number of attacks to sample (anomaly ratio 0.2 -> attacks ≈ 0.25 * normals)
        num_normal = normal_data.shape[0]
        target_num_attacks = int(0.25 * num_normal)
        if target_num_attacks > attack_data.shape[0]:
            target_num_attacks = attack_data.shape[0]
        rand_idx = np.random.permutation(attack_data.shape[0])
        selected_attacks = attack_data[rand_idx[:target_num_attacks]]
        selected_attack_labels = attack_labels[rand_idx[:target_num_attacks]]
        # Split normals for train/test
        rand_norm_idx = np.random.permutation(num_normal)
        half_norm = num_normal // 2
        train_norm_data = normal_data[rand_norm_idx[:half_norm]]
        train_norm_labels = normal_labels[rand_norm_idx[:half_norm]]
        test_norm_data = normal_data[rand_norm_idx[half_norm:]]
        test_norm_labels = normal_labels[rand_norm_idx[half_norm:]]
        # Test set: remaining normals + all selected attacks
        test_attack_data = selected_attacks
        test_attack_labels = selected_attack_labels
        self.train = train_norm_data;      self.train_labels = train_norm_labels
        self.test  = np.concatenate((test_norm_data, test_attack_data), axis=0)
        self.test_labels = np.concatenate((test_norm_labels, test_attack_labels), axis=0)
        self.mode = mode
    def __len__(self):
        return self.train.shape[0] if self.mode == 'train' else self.test.shape[0]
    def __getitem__(self, index):
        if self.mode == 'train':
            return np.float32(self.train[index]), np.float32(self.train_labels[index])
        else:
            return np.float32(self.test[index]), np.float32(self.test_labels[index])

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # Hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--gmm_k', type=int, default=4)
    parser.add_argument('--lambda_energy', type=float, default=0.1)
    parser.add_argument('--lambda_cov_diag', type=float, default=0.005)
    # Mode and paths
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--data_path', type=str, default='kddcup.data_10_percent')
    parser.add_argument('--model_save_path', type=str, default='./dagmm_kddrev/models')
    # Logging step size
    parser.add_argument('--sample_step', type=int, default=194)
    parser.add_argument('--model_save_step', type=int, default=194)
    # Distribution choice for DAGMM
    parser.add_argument('--dist_type', type=str, default='gaussian', choices=['gaussian','laplace','student_t'])
    parser.add_argument('--student_nu', type=float, default=4.0)
    config = parser.parse_args()
    # Data loader
    dataset = KDDCupRevLoader(config.data_path, mode=config.mode)
    shuffle = True if config.mode == 'train' else False
    data_loader = DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=shuffle)
    # Initialize solver and model
    solver = Solver(data_loader, vars(config))
    # Adjust DAGMM model input/output dimensions (118 features after encoding)
    input_dim = dataset.train.shape[1] if config.mode == 'train' else dataset.test.shape[1]
    solver.model.encoder[0] = torch.nn.Linear(input_dim, solver.model.encoder[0].out_features)
    solver.model.decoder[-1] = torch.nn.Linear(solver.model.decoder[-1].in_features, input_dim)
    # Run training or testing
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()
