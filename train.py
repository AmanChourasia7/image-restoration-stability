import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config
from dataset import NoisyDataset
from models.dncnn import DnCNN
from metrics import psnr


def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_dataset = NoisyDataset(train=True)
    test_dataset = NoisyDataset(train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=0
    )

    model = DnCNN().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate)

    for epoch in range(Config.epochs):

        model.train()
        total_loss = 0

        print("\nEpoch", epoch, "starting")

        for i, (noisy, clean) in enumerate(train_loader):

            noisy = noisy.to(device)
            clean = clean.to(device)

            output = model(noisy)

            loss = criterion(output, clean)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 100 == 0:
                print("Batch", i, "Loss:", loss.item())

        print("Epoch", epoch, "Loss:", total_loss)

        evaluate(model, test_loader, device)


def evaluate(model, loader, device):

    model.eval()

    total_psnr = 0
    count = 0

    with torch.no_grad():

        for noisy, clean in loader:

            noisy = noisy.to(device)
            clean = clean.to(device)

            output = model(noisy)

            total_psnr += psnr(output, clean)
            count += 1

    print("Test PSNR:", (total_psnr / count).item())


if __name__ == "__main__":
    train()
