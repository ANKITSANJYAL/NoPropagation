# experiments/run_mnist_dt.py

import torch
from models.no_prop_dt import NoPropDT
from trainer.train_nopropdt import train_nopropdt
from data.mnist_loader import get_mnist_loaders

def main():
    # Hyperparameters
    T = 10
    eta = 0.1
    embedding_dim = 256
    num_classes = 10
    batch_size = 128
    lr = 1e-3
    weight_decay = 1e-3
    epochs = 10

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load data
    train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)

    # Build model
    model = NoPropDT(num_classes=num_classes, embedding_dim=embedding_dim, T=T, eta=eta)

    # Train
    train_nopropdt(model, train_loader, test_loader, epochs, lr, weight_decay, device=device)

if __name__ == "__main__":
    main()
