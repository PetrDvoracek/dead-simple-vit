import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

import sys
sys.path.append(".")

# from vit_simple import ViT
# from vit_position import ViT
from vit_optimized import ViT


class Cutout:
    """Randomly mask out a square region."""

    def __init__(self, size=8):
        self.size = size

    def __call__(self, img):
        h, w = img.shape[1], img.shape[2]
        y = torch.randint(0, h, (1,)).item()
        x = torch.randint(0, w, (1,)).item()

        y1 = max(0, y - self.size // 2)
        y2 = min(h, y + self.size // 2)
        x1 = max(0, x - self.size // 2)
        x2 = min(w, x + self.size // 2)

        img[:, y1:y2, x1:x2] = 0
        return img


def mixup(x, y, alpha=0.8):
    """Mixup data augmentation."""
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine annealing."""

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]["lr"]

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr


if __name__ == "__main__":
    # TODO Change this to your dataset path
    dataset_root = "/Volumes/External/data/torch_datasets"
    name = "vit_optimized"
    # TODO change device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Hyperparameters
    epochs = 200
    warmup_epochs = 20
    learning_rate = 1e-3
    min_lr = 1e-5
    batch_size = 128
    weight_decay = 0.05
    grad_clip = 1.0
    label_smoothing = 0.1
    mixup_alpha = 0.8

    # Model parameters - must match vit_optimized.py expectations
    img_size = 32
    patch_size = 4      # Smaller patches = more tokens (64 vs 4)
    embed_dim = 384     # Divisible by num_heads
    depth = 6
    num_heads = 8
    num_classes = 10

    model = ViT(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        num_classes=num_classes,
    ).to(device)

    # Print model info
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    print(f"Number of patches: {(img_size // patch_size) ** 2}")
    print(f"Device: {device}")

    # Optimizer with proper weight decay (no decay on bias and layernorm)
    decay_params = []
    no_decay_params = []
    for name_p, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or "bias" in name_p:
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(0.9, 0.999),
    )

    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs, epochs, min_lr)
    lossfn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Strong data augmentation for training
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.RandAugment(num_ops=2, magnitude=9),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        Cutout(size=8),
    ])

    val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=dataset_root,
        transform=train_transform,
        download=True,
        train=True,
    )
    val_dataset = torchvision.datasets.CIFAR10(
        root=dataset_root,
        transform=val_transform,
        download=True,
        train=False,
    )
    dataloader_train = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True
    )
    dataloader_val = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True
    )

    # Lists to store metrics for plotting
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_val_acc = 0.0

    for epoch in range(epochs):
        # Update learning rate
        current_lr = scheduler.step(epoch)

        # Training phase
        model.train()
        correct = 0
        total = 0
        train_loss = 0.0

        progress_bar = tqdm(
            dataloader_train, desc=f"Epoch {epoch+1}/{epochs} [LR: {current_lr:.2e}]"
        )
        for batch_idx, (x, y) in enumerate(progress_bar):
            x = x.to(device)
            y = y.to(device)

            # Apply mixup
            x_mixed, y_a, y_b, lam = mixup(x, y, alpha=mixup_alpha)

            optimizer.zero_grad()
            y_hat = model(x_mixed)

            # Mixup loss
            l = mixup_criterion(lossfn, y_hat, y_a, y_b, lam)
            l.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            # Calculate accuracy on clean data for monitoring
            with torch.no_grad():
                y_hat_clean = model(x)
                _, predicted = torch.max(y_hat_clean, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

            # Update training loss
            train_loss += l.item()

            # Update progress bar
            accuracy = 100 * correct / total
            avg_loss = train_loss / (batch_idx + 1)
            progress_bar.set_postfix(
                {"Loss": f"{avg_loss:.4f}", "Acc": f"{accuracy:.2f}%"}
            )

        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for x, y in dataloader_val:
                x = x.to(device)
                y = y.to(device)
                y_hat = model(x)
                l = lossfn(y_hat, y)

                # Calculate accuracy
                _, predicted = torch.max(y_hat, 1)
                val_total += y.size(0)
                val_correct += (predicted == y).sum().item()
                val_loss += l.item()

        val_accuracy = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(dataloader_val)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        # Track best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), f"{name}_best.pth")

        print(f"Epoch {epoch+1}: Train Acc: {accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%, Best: {best_val_acc:.2f}%")

    # Plot training metrics
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot loss on left y-axis
    color = "tab:red"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color=color)
    ax1.plot(range(1, epochs + 1), train_losses, "r-", label="Train Loss", alpha=0.7)
    ax1.plot(range(1, epochs + 1), val_losses, "r--", label="Val Loss", alpha=0.7)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.legend(loc="upper left")

    # Create second y-axis for accuracy
    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Accuracy (%)", color=color)
    ax2.set_ylim(0, 100)
    ax2.plot(range(1, epochs + 1), train_accuracies, "b-", label="Train Accuracy", alpha=0.7)
    ax2.plot(range(1, epochs + 1), val_accuracies, "b--", label="Val Accuracy", alpha=0.7)
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.legend(loc="upper right")

    plt.title(f"Training Metrics - Best Val Acc: {best_val_acc:.2f}%")
    plt.tight_layout()
    plt.savefig(f"{name}_e-{epochs}_acc-{best_val_acc:.2f}.png")
    plt.show()

    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%")
