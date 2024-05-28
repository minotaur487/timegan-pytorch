import numpy as np
import torch
from discriminator import Discriminator
from embedder import Embedder
from generator import Generator
from recovery import Recovery
from supervisor import Supervisor
from torch import nn, optim
from torch.utils.data import DataLoader


def TimeGAN(dataset, parameters):
    """
    TimeGAN: Combines all the networks to generate synthetic data

    Args:
      - dataset: pytorch dataset containing processed input time-series data;
        np array of shape (num of seq, seq length, dim of each sample)
      - parameters:

    Returns:
      - generated_data: synthetic time-series data
    """
    # Data Parameters
    # num_of_seq, seq_len, dim = np.asarray(data).shape

    # Network Parameters
    iterations = parameters["iterations"]
    batch_size = parameters["batch_size"]
    gamma = parameters["gamma"]
    EMBEDDER_LR = 0.001
    RECOVERY_LR = 0.001
    SUPERVISOR_LR = 0.001
    GENERATOR_LR = 0.001
    DISCRIMINATOR_LR = 0.001

    # Calculate number of epochs
    num_of_batches = int(np.ceil(len(dataset) / batch_size))
    num_of_epochs = int(iterations / num_of_batches)

    # Instantiate networks
    embedder = Embedder(parameters)
    recovery = Recovery(parameters)
    supervisor = Supervisor(parameters)
    generator = Generator(parameters)
    discriminator = Discriminator(parameters)

    # Instantiate optimizers
    embedder_optimizer = optim.Adam(embedder.parameters(), lr=EMBEDDER_LR)
    recovery_optimizer = optim.Adam(recovery.parameters(), lr=RECOVERY_LR)
    supervisor_optimizer = optim.Adam(supervisor.parameters(), lr=SUPERVISOR_LR)
    generator_optimizer = optim.Adam(generator.parameters(), lr=GENERATOR_LR)
    discriminator_optimizer = optim.Adam(
        discriminator.parameters(), lr=DISCRIMINATOR_LR
    )

    # Define loss functions
    def embedder_loss(self, X, X_tilde, G_loss_S, is_joint_training=False):
        E_loss_T0 = nn.MSELoss(X, X_tilde)
        E_loss0 = 10 * torch.sqrt(E_loss_T0)
        # calculate E_loss if joint training
        if is_joint_training:
            return E_loss0 + 0.1 * G_loss_S
        return E_loss0

    def supervisor_loss(self, H, H_hat_supervise):
        G_loss_S = nn.MSELoss(H[:, 1:, :], H_hat_supervise[:, :-1, :])
        return G_loss_S

    def generator_loss(self, Y_fake, Y_fake_e, X, X_hat, H, H_hat_supervise):
        G_loss_S = nn.MSELoss(H[:, 1:, :], H_hat_supervise[:, :-1, :])

        X_mean = torch.mean(X, [0])
        X_std = torch.std(X, [0])
        X_hat_mean = torch.mean(X_hat, [0])
        X_hat_std = torch.std(X_hat, [0])
        G_loss_V1 = torch.mean(
            torch.abs(torch.sqrt(X_hat_std + 1e-6) - torch.sqrt(X_std + 1e-6))
        )
        G_loss_V2 = torch.mean(torch.abs(X_hat_mean - X_mean))
        G_loss_V = G_loss_V1 + G_loss_V2

        G_loss_U = nn.BCEWithLogitsLoss(torch.ones_like(Y_fake), Y_fake)
        G_loss_U_e = nn.BCEWithLogitsLoss(torch.ones_like(Y_fake_e), Y_fake_e)
        G_loss = (
            G_loss_U + gamma * G_loss_U_e + 100 * torch.sqrt(G_loss_S) + 100 * G_loss_V
        )
        return G_loss

    def discriminator_loss(self, Y_fake, Y_real, Y_fake_e):
        D_loss_real = nn.BCEWithLogitsLoss(torch.ones_like(Y_real), Y_real)
        D_loss_fake = nn.BCEWithLogitsLoss(torch.zeros_like(Y_fake), Y_fake)
        D_loss_fake_e = nn.BCEWithLogitsLoss(torch.zeros_like(Y_fake_e), Y_fake_e)
        D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e
        return D_loss

    def train_embedded_network():
        for i in range(num_of_epochs):
            train_loader = DataLoader(
                dataset, batch_size=parameters["batch_size"], shuffle=True
            )
            for X in train_loader:
                return

    def train_supervised_loss():
        for i in range(num_of_epochs):
            train_loader = DataLoader(
                dataset, batch_size=parameters["batch_size"], shuffle=True
            )
            for X in train_loader:
                return

    def train_joint():
        for i in range(num_of_epochs):
            train_loader = DataLoader(
                dataset, batch_size=parameters["batch_size"], shuffle=True
            )
            for X in train_loader:
                return

    def train_discriminator():
        for i in range(num_of_epochs):
            train_loader = DataLoader(
                dataset, batch_size=parameters["batch_size"], shuffle=True
            )
            for X in train_loader:
                return

    def generate_synthetic_data():
        # TODO
        pass

    train_embedded_network()
    train_supervised_loss()
    train_joint()
    train_discriminator()
    generated_data = generate_synthetic_data()
    return generate_synthetic_data
