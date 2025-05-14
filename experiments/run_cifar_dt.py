# experiments/run_cifar_dt.py

import torch
from models.no_prop_dt import NoPropDT
from trainer.train_nopropdt import train_nopropdt
from data.cifar_loader import get_cifar_loaders

def main():
    # Hyperparameters
    T = 10
    eta = 0.1
    embedding_dim = 256
    num_classes = 10
    batch_size = 128
    lr = 1e-3
    weight_decay = 1e-3
    epochs = 50

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load CIFAR-10
    train_loader, test_loader = get_cifar_loaders(batch_size=batch_size)

    # Init model with 3 input channels (RGB)
    model = NoPropDT(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        T=T,
        eta=eta,
        num_input_channels=3, # key change
        use_decoder=True
    )

    # Train
    train_nopropdt(model, train_loader, test_loader, epochs, lr, weight_decay, device=device)

if __name__ == "__main__":
    main()
