#! /usr/bin/python3

"""
Autoencoder is a type of neural network that is responsible for encoding and decoding data.
Analogy can be morse code where text is manually converted into a sequence of signals.
Autoencoder would rather learn this encoding format from the text and then decode it back.
Another example of what an autoencoder can do is lower is the dimensionality of the data.
- Compared to PCA, autoencoder can also learn non-linear transformations.

Applications:
- Compressing the images, denoising the images, generating new images, etc.
- Reconstructing or filling out parts of the image.
- Removing parts of the image.

Architecture:
- Special type of neural network
- First layers helps it to ingest the data. Could be flattened or CNNs
- Dimensions reduce until we reach bottleneck or CODE
- Then dimensions expand in successive layers till output decoded format is obtained

Types:
Stacked: Layers additional to encoding, bottleneck, and decoding
Variational: Uses mean and variance to generate new data
Sparse: 
Denoising: Trains on corrupted data to denoise and reconstruct the data

Stacked autoencoder that takes original image and encodes it then decodes again.
- no labels
- use original image to compute MSE loss

Denoising example: original -> gaussian noise -> noise -> autoencoder -> reconstructed -> MSE loss

Variational:
- Original image -> learn mean/stddev -> decoded original image -> MSE with original image

Sparse:
- Used in pretraining networks for classification
- hidden layers with partially active neurons
"""

import os
import numpy as np
import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.nn import Module
import matplotlib.pyplot as plt
import multiprocessing
from torch.autograd import Variable
from IPython import embed

multiprocessing.set_start_method("fork")

DEVICE = "mps" if torch.mps.is_available() else "cpu"
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
BATCH_SIZE = 500
LEARNING_RATE = 0.001
EPOCHS = 10
NOISE_LEVEL = 0.1


def display_images(original: np.array, noisy: np.array, denoised: np.array):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[1].imshow(noisy)
    axes[1].set_title("Noisy")
    axes[2].imshow(denoised)
    axes[2].set_title("Denoised")
    plt.show()


tf_composed = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)
train_data = CIFAR10(
    train=True,
    root=os.path.join(ROOT, "data"),
    download=True,
    transform=tf_composed,
)
test_data = CIFAR10(
    train=False,
    root=os.path.join(ROOT, "data"),
    download=True,
    transform=tf_composed,
)
train_loader = DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)
test_loader = DataLoader(
    dataset=test_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)


class DenoisingAutoencoder(Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        # Input 32x32x3
        self._conv1 = torch.nn.Conv2d(
            in_channels=3, out_channels=24, kernel_size=3, padding=2
        )  # 30x30x24
        self._conv2 = torch.nn.Conv2d(
            in_channels=24, out_channels=48, kernel_size=3, padding=2
        )  # 28x28x48
        self._conv3 = torch.nn.Conv2d(
            in_channels=48, out_channels=96, kernel_size=3, padding=2
        )  # 26x26x96
        self._conv4 = torch.nn.Conv2d(
            in_channels=96, out_channels=128, kernel_size=3, padding=2
        )  # 24x24x128
        self._conv5 = torch.nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, padding=2
        )  # 22x22x256
        self._code = torch.nn.MaxPool2d(kernel_size=2, return_indices=True)  # 11x11x256
        self._expand = torch.nn.MaxUnpool2d(kernel_size=2)
        self._deconv1 = torch.nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=3, padding=2
        )
        self._deconv2 = torch.nn.ConvTranspose2d(
            in_channels=128, out_channels=96, kernel_size=3, padding=2
        )
        self._deconv3 = torch.nn.ConvTranspose2d(
            in_channels=96, out_channels=48, kernel_size=3, padding=2
        )
        self._deconv4 = torch.nn.ConvTranspose2d(
            in_channels=48, out_channels=24, kernel_size=3, padding=2
        )
        self._deconv5 = torch.nn.ConvTranspose2d(
            in_channels=24, out_channels=3, kernel_size=3, padding=2
        )

    def forward(self, x):
        x = self._conv1(x)
        x = torch.nn.functional.relu(x)
        x = self._conv2(x)
        x = torch.nn.functional.relu(x)
        x = self._conv3(x)
        x = torch.nn.functional.relu(x)
        x = self._conv4(x)
        x = torch.nn.functional.relu(x)
        x = self._conv5(x)
        x, indices = self._code(x)
        x = self._expand(x, indices=indices)
        x = self._deconv1(x)
        x = torch.nn.functional.relu(x)
        x = self._deconv2(x)
        x = torch.nn.functional.relu(x)
        x = self._deconv3(x)
        x = torch.nn.functional.relu(x)
        x = self._deconv4(x)
        x = torch.nn.functional.relu(x)
        x = self._deconv5(x)
        x = torch.nn.functional.relu(x)
        return x


if __name__ == "__main__":

    # Test image plotting
    # image, _ = train_data[0]
    # original = image.numpy().transpose(1, 2, 0)
    # noisy = original + np.random.normal(0, 0.1, original.shape)
    # display_images(original, noisy, original)

    # Main code starts here
    autoencoder = DenoisingAutoencoder().to(DEVICE)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)
    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}")
        autoencoder.train()
        train_loss = 0
        for i, (images, _) in enumerate(train_loader):
            optimizer.zero_grad()
            # Add noise to image
            noisy_images = images + torch.randn_like(images) * NOISE_LEVEL
            noisy_images = torch.clamp(noisy_images, 0, 1)
            images = Variable(data=noisy_images).to(DEVICE)
            output = autoencoder(images)
            loss = criterion(output, images)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        torch.save(
            autoencoder.state_dict(),
            os.path.join(ROOT, "model", "autoencoder.model.pth"),
        )
        # Test the autoencoder
        with torch.no_grad():
            test_loss = 0
            autoencoder.eval()
            for i, (images, _) in enumerate(test_loader):
                images = Variable(data=images).to(DEVICE)
                output = autoencoder(images)
                loss = criterion(output, images)
                test_loss += loss.item()
                if False:
                    original = images[0].cpu().numpy().transpose(1, 2, 0)
                    noisy = original + np.random.normal(0, 0.1, original.shape)
                    denoised = output[0].cpu().numpy().transpose(1, 2, 0)
                    display_images(original, noisy, denoised)
        print(
            f"Train Loss: {train_loss / len(train_loader):.2f}, Test Loss: {test_loss / len(test_loader):.2f}\n"
        )
    embed()
