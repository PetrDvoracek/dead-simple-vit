import torch
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.append(".")

from vit_simple import ViT



if __name__ == "__main__":
    # TODO Change this to your dataset path
    dataset_root = "/Volumes/External/data/torch_datasets"
    name = "1_block_optimized"
    # TODO change device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    epochs = 100  # Increased epochs for better convergence
    learning_rate = 1e-5  # Slightly higher learning rate for faster training
    batch_size = 128  # Reduced batch size to fit into GPU memory
    embed = 512
    patch_size = 16
    num_classes = 10

    model = ViT(patch_size, embed, num_classes).to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate, weight_decay=0.05)  # Added weight decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)  # Added learning rate scheduler
    lossfn = torch.nn.CrossEntropyLoss()

    train_dataset = torchvision.datasets.CIFAR10(
        root=dataset_root,
        transform=T.Compose(
            [
                T.RandomHorizontalFlip(),  # Data augmentation: horizontal flip
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),  # Normalize with CIFAR-10 stats
            ]
        ),
        download=True,
        train=True,
    )
    val_dataset = torchvision.datasets.CIFAR10(
        root=dataset_root,
        transform=T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),  # Normalize with CIFAR-10 stats
            ]
        ),
        download=True,
        train=False,
    )
    dataloader_train = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True
    )
    dataloader_val = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True
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

        # Step the scheduler
        scheduler.step()

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
    ax2.set_ylim(0, 100)
    ax2.plot(range(1, epochs + 1), train_accuracies, "b-", label="Train Accuracy")
    ax2.plot(range(1, epochs + 1), val_accuracies, "b--", label="Val Accuracy")
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.legend(loc="upper right")

    plt.title("Training and Validation Metrics")
    plt.tight_layout()
    plt.savefig(f"{name}_e-{epochs}_acc-{val_accuracies[-1]:.2f}.png")
    plt.show()