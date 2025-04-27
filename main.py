import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
from utils import *


def str2bool(v):
    return v.lower() in ("true")


def main(config):
    # For fast training
    cudnn.benchmark = True

    # Create directories if not exist
    mkdir(config.model_save_path)

    data_loader = get_loader(
        config.data_path, batch_size=config.batch_size, mode=config.mode
    )

    # Solver
    solver = Solver(data_loader, vars(config))

    if config.mode == "train":
        solver.train()
    elif config.mode == "test":
        solver.test()

    return solver


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument("--lr", type=float, default=1e-4)

    # Training settings
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--gmm_k", type=int, default=4)
    parser.add_argument("--lambda_energy", type=float, default=0.1)
    parser.add_argument("--lambda_cov_diag", type=float, default=0.005)

    # Misc
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])

    # Path
    parser.add_argument("--data_path", type=str, default="kdd_cup.npz")
    parser.add_argument(
        "--model_save_path", type=str, default="./dagmm_gaussian/models"
    )

    # Step size
    parser.add_argument("--sample_step", type=int, default=194)
    parser.add_argument("--model_save_step", type=int, default=194)

    # Distribution choice
    parser.add_argument(
        "--dist_type",
        type=str,
        default="gaussian",
        choices=["gaussian", "laplace", "student_t"],
    )
    parser.add_argument("--student_nu", type=float, default=4.0)

    config = parser.parse_args()

    args = vars(config)

    main(config)
