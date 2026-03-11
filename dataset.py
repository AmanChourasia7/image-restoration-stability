import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from config import Config


class NoisyDataset(Dataset):

    def __init__(self, train=True):

        transform = transforms.Compose([
            transforms.Resize((Config.image_size, Config.image_size)),
            transforms.ToTensor()
        ])

        self.dataset = CIFAR10(
            root="./data",
            train=train,
            download=True,
            transform=transform
        )

    def add_noise(self, image):

        noise = torch.randn_like(image) * (Config.noise_sigma / 255.0)
        noisy_image = image + noise
        noisy_image = torch.clamp(noisy_image, 0., 1.)

        return noisy_image

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        clean_image, _ = self.dataset[idx]
        noisy_image = self.add_noise(clean_image)

        return noisy_image, clean_image
