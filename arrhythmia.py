import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from model import DaGMM
from solver import Solver

class ArrhythmiaLoader(Dataset):
    def __init__(self, data_path, mode="train"):
        # Load arrhythmia dataset (treat '?' as NaN for missing values)
        data = pd.read_csv(data_path, header=None, na_values='?')
        features = data.iloc[:, :-1]
        labels = data.iloc[:, -1].astype(int).values  # class labels 1-16
        # Impute missing values: numeric with mean, categorical with mode
        for col in features.columns:
            if features[col].dtype == object:
                features[col].fillna(features[col].mode()[0], inplace=True)
            else:
                features[col].fillna(features[col].mean(), inplace=True)
        features = features.values
        # Label 1 for anomaly if class in {3,4,5,7,8,9,14,15}, else 0 for normal
        anomaly_classes = {3,4,5,7,8,9,14,15}
        binary_labels = np.isin(labels, list(anomaly_classes)).astype(int)
        # Split normal data for training/test
        normal_data = features[binary_labels == 0]; normal_labels = binary_labels[binary_labels == 0]
        anomaly_data = features[binary_labels == 1]; anomaly_labels = binary_labels[binary_labels == 1]
        num_normal = normal_data.shape[0]
        rand_idx = np.random.permutation(num_normal)
        half_norm = num_normal // 2
        train_norm_data = normal_data[rand_idx[:half_norm]]
        train_norm_labels = normal_labels[rand_idx[:half_norm]]
        test_norm_data = normal_data[rand_idx[half_norm:]]
        test_norm_labels = normal_labels[rand_idx[half_norm:]]
        test_anom_data = anomaly_data
        test_anom_labels = anomaly_labels
        self.train = train_norm_data;      self.train_labels = train_norm_labels
        self.test  = np.concatenate((test_norm_data, test_anom_data), axis=0)
        self.test_labels = np.concatenate((test_norm_labels, test_anom_labels), axis=0)
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
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gmm_k', type=int, default=4)
    parser.add_argument('--lambda_energy', type=float, default=0.1)
    parser.add_argument('--lambda_cov_diag', type=float, default=0.005)
    parser.add_argument('--mode', type=str, default='train', choices=['train','test'])
    parser.add_argument('--data_path', type=str, default='arrhythmia.data')
    parser.add_argument('--model_save_path', type=str, default='./dagmm_arrhythmia/models')
    parser.add_argument('--sample_step', type=int, default=20)
    parser.add_argument('--model_save_step', type=int, default=20)
    parser.add_argument('--dist_type', type=str, default='gaussian', choices=['gaussian','laplace','student_t'])
    parser.add_argument('--student_nu', type=float, default=4.0)
    config = parser.parse_args()
    dataset = ArrhythmiaLoader(config.data_path, mode=config.mode)
    data_loader = DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=(config.mode=='train'))
    solver = Solver(data_loader, vars(config))
    # Adjust input/output dimensions for Arrhythmia features
    input_dim = dataset.train.shape[1] if config.mode == 'train' else dataset.test.shape[1]
    solver.dagmm.encoder[0] = torch.nn.Linear(input_dim, solver.dagmm.encoder[0].out_features)
    solver.dagmm.decoder[-1] = torch.nn.Linear(solver.dagmm.decoder[-1].in_features, input_dim)
    if config.mode == 'train':
        solver.train()
    else:
        solver.test()
