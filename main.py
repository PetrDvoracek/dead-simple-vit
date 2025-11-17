import torch
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt

import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingLayer(torch.nn.Module):
    def __init__(self, embed_dim, patch_size):
        super(EmbeddingLayer, self).__init__()
        # Patchification is done using convolution with
        # big kernel_size and stride.
        self.conv = torch.nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        x = self.conv(x)
        # merge `height` and `width` dimensions into one `tokens` dimension
        # batch x tokens x features
        x = x.reshape(x.shape[0], -1, x.shape[1])
        return x


class TransformerBlock(torch.nn.Module):
    def __init__(self, embed, embed_inner):
        super().__init__()
        self.q = torch.nn.Linear(embed, embed_inner)
        self.k = torch.nn.Linear(embed, embed_inner)
        self.v = torch.nn.Linear(embed, embed_inner)
        self.fc_attn = torch.nn.Linear(embed_inner, embed)
        self.fc_block = torch.nn.Linear(embed, embed)
        self.norm1 = torch.nn.LayerNorm(embed)
        self.norm2 = torch.nn.LayerNorm(embed)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x_skip_1 = x.clone()
        x = self.norm1(x)

        # self attention
        q, k, v = self.q(x), self.k(x), self.v(x)
        qk = torch.matmul(q, k.transpose(-2, -1))
        qk = self.softmax(qk)
        qkv = torch.matmul(qk, v)
        x = self.fc_attn(qkv)

        x = x + x_skip_1
        x_skip_2 = x.clone()
        x = self.norm2(x)
        x = self.fc_block(x)
        x = x + x_skip_2
        return x


class Transformer(torch.nn.Module):
    def __init__(self, embed_dim, patch_size, num_classes):
        super(Transformer, self).__init__()
        self.embedding = EmbeddingLayer(embed_dim, patch_size)
        self.cls_token = torch.nn.Parameter(torch.rand(embed_dim))
        self.blocks = torch.nn.Sequential(
            TransformerBlock(embed_dim, int(embed_dim * 1.5)),
            TransformerBlock(embed_dim, int(embed_dim * 1.5)),
            TransformerBlock(embed_dim, int(embed_dim * 1.5)),
        )
        self.cls_layer = torch.nn.Linear(embed_dim, out_features=num_classes)

    def forward(self, x):
        bs = x.shape[0]
        cls_token = torch.stack([self.cls_token] * bs).unsqueeze(1)

        x = self.embedding(x)
        x = torch.cat([cls_token, x], dim=1)

        x = self.blocks(x)

        cls_token = x[:, 0, :]

        out = self.cls_layer(cls_token)

        return out


if __name__ == "__main__":
    # TODO Change this to your dataset path
    dataset_root = "/Volumes/External/data/torch_datasets"
    # dataset_root = "~/torch_datasets"
    name = "1_block"
    # TODO change device
    device = "mps"  # for Apple Silicon GPU (MacBooks with M* chips)
    # device = "cuda" # for NVIDIA GPU
    # device = "cpu" # use if you do not have a compatible GPU

    epochs = 10
    batch_size = 128
    embed = 512
    patch_size = 16
    num_classes = 10
    model = Transformer(embed, patch_size, num_classes).to(device)

    dataset = torchvision.datasets.CIFAR10(
        root=dataset_root,
        transform=T.Compose([T.ToTensor()]),
        download=True,
    )
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5)
    lossfn = torch.nn.CrossEntropyLoss()

    # Split dataset into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    dataloader_train = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    dataloader_val = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    # Lists to store metrics for plotting
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        correct = 0
        total = 0
        train_loss = 0.0

        progress_bar = tqdm(
            dataloader_train, desc=f"Epoch {epoch+1}/{epochs} - Training"
        )
        for batch_idx, (x, y) in enumerate(progress_bar):
            x = x.to(torch.float32).to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_hat = model(x)
            l = lossfn(
                y_hat.to(torch.float32),
                y.to(torch.long),
            )
            l.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(y_hat.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

            # Update training loss
            train_loss += l.item()

            # Update progress bar
            accuracy = 100 * correct / total
            avg_loss = train_loss / (batch_idx + 1)
            progress_bar.set_postfix(
                {"Loss": f"{avg_loss:.4f}", "Accuracy": f"{accuracy:.2f}%"}
            )

        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            progress_bar_val = tqdm(
                dataloader_val, desc=f"Epoch {epoch+1}/{epochs} - Validation"
            )
            for batch_idx, (x, y) in enumerate(progress_bar_val):
                x = x.to(torch.float32).to(device)
                y = y.to(device)
                y_hat = model(x)
                l = lossfn(
                    y_hat.to(torch.float32),
                    y.to(torch.long),
                )

                # Calculate accuracy
                _, predicted = torch.max(y_hat.data, 1)
                val_total += y.size(0)
                val_correct += (predicted == y).sum().item()
                val_loss += l.item()

                # Update progress bar
                val_accuracy = 100 * val_correct / val_total
                avg_val_loss = val_loss / (batch_idx + 1)
                progress_bar_val.set_postfix(
                    {
                        "Val Loss": f"{avg_val_loss:.4f}",
                        "Val Accuracy": f"{val_accuracy:.2f}%",
                    }
                )

        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

    # Plot training metrics
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot loss on left y-axis
    color = "tab:red"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color=color)
    ax1.plot(range(1, epochs + 1), train_losses, "r-", label="Train Loss")
    ax1.plot(range(1, epochs + 1), val_losses, "r--", label="Val Loss")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.legend(loc="upper left")

    # Create second y-axis for accuracy
    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Accuracy (%)", color=color)
    ax2.plot(range(1, epochs + 1), train_accuracies, "b-", label="Train Accuracy")
    ax2.plot(range(1, epochs + 1), val_accuracies, "b--", label="Val Accuracy")
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.legend(loc="upper right")

    plt.title("Training and Validation Metrics")
    plt.tight_layout()
    plt.savefig(f"{name}_e-{epochs}_acc-{val_accuracies[-1]}.png")
    plt.show()
