"""
COMP0197 Coursework 1 — Task 2: Robust Representation via MixUp and Label Smoothing

train.py
--------
Downloads CIFAR-10, implements custom MixUp data augmentation and Label Smoothing
cross-entropy loss from scratch using basic tensor operations, trains a baseline
CNN and a CNN incorporating both techniques with manual validation-based early
stopping, and saves both models.

GenAI Usage Statement
---------------------
Claude was used to assist with code structuring and drafting the
technical analysis. All model design decisions, hyperparameter choices, and
from-scratch implementations were verified by the author against the COMP0197
lecture material. One specific correction: Claude initially applied label
smoothing only to hard targets, whereas the correct approach applies smoothing
on top of MixUp's already-soft labels during training, ensuring both
regularisation techniques compose correctly.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import json
import copy


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

class ConvNet(nn.Module):
    """CNN for CIFAR-10 classification built from layer primitives.

    Architecture (input: 3x32x32):
        Conv2d(3,  32, 3, pad=1) -> BatchNorm2d(32)  -> ReLU           [32x32]
        Conv2d(32, 64, 3, pad=1) -> BatchNorm2d(64)  -> ReLU -> MaxPool(2) [16x16]
        Conv2d(64,128, 3, pad=1) -> BatchNorm2d(128) -> ReLU -> MaxPool(2) [8x8]
        Conv2d(128,256,3, pad=1) -> BatchNorm2d(256) -> ReLU -> MaxPool(2) [4x4]
        Flatten -> Linear(4096, 256) -> ReLU -> Dropout(p) -> Linear(256, 10)

    Args:
        num_classes  (int):   Number of output classes.  Default: 10.
        dropout_rate (float): Dropout probability in the classifier head.
                              Default: 0.3.
    """

    def __init__(self, num_classes=10, dropout_rate=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): Images of shape (batch_size, 3, 32, 32).

        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes).
        """
        return self.classifier(self.features(x))


# ──────────────────────────────────────────────────────────────────────────────
# From-Scratch: Custom MixUp
# ──────────────────────────────────────────────────────────────────────────────

def mixup_data(x, y, num_classes=10, alpha=0.2):
    """Apply MixUp augmentation to a batch of inputs and labels.

    Samples a single mixing coefficient lambda from Beta(alpha, alpha) per
    batch and blends image pairs with their one-hot encoded labels.  Pairs
    are formed by randomly permuting batch indices.

    Implemented from scratch using basic tensor operations only:
    - torch.distributions.Beta for lambda sampling
    - Element-wise multiplication and addition for blending
    - scatter_ for one-hot encoding

    Reference: Lecture 4 — mixup regularisation (vicinal risk minimisation).
    Following the original MixUp paper (Zhang et al., 2018): one lambda
    per batch.

    Args:
        x           (torch.Tensor): Input batch, shape (N, C, H, W).
        y           (torch.Tensor): Integer class labels, shape (N,).
        num_classes (int):          Total number of classes.  Default: 10.
        alpha       (float):        Beta distribution parameter.  Default: 0.2.

    Returns:
        tuple: (mixed_x, mixed_y, lam, indices) where:
            mixed_x  (torch.Tensor): Blended inputs, same shape as x.
            mixed_y  (torch.Tensor): Blended soft labels, shape (N, num_classes).
            lam      (float):        Mixing coefficient.
            indices  (torch.Tensor): Permuted indices, shape (N,).
    """
    batch_size = x.size(0)

    # Sample single lambda per batch from Beta(alpha, alpha)
    # Beta sampling on CPU (not supported on MPS); .item() extracts scalar
    if alpha > 0.0:
        lam = torch.distributions.Beta(
            torch.tensor(alpha),
            torch.tensor(alpha),
        ).sample().item()
    else:
        lam = 1.0

    # Random permutation for pairing
    indices = torch.randperm(batch_size, device=x.device)

    # Blend inputs: x_tilde = lam * x_i + (1 - lam) * x_j
    mixed_x = lam * x + (1.0 - lam) * x[indices]

    # One-hot encode labels using scatter_
    y_onehot = torch.zeros(batch_size, num_classes, device=x.device)
    y_onehot.scatter_(1, y.unsqueeze(1), 1.0)

    y_perm_onehot = torch.zeros(batch_size, num_classes, device=x.device)
    y_perm_onehot.scatter_(1, y[indices].unsqueeze(1), 1.0)

    # Blend labels: y_tilde = lam * y_i + (1 - lam) * y_j
    mixed_y = lam * y_onehot + (1.0 - lam) * y_perm_onehot

    return mixed_x, mixed_y, lam, indices


# ──────────────────────────────────────────────────────────────────────────────
# From-Scratch: Label Smoothing Cross-Entropy
# ──────────────────────────────────────────────────────────────────────────────

def label_smoothing_cross_entropy(logits, targets, epsilon=0.1):
    """Compute cross-entropy loss with label smoothing.

    Converts target distributions to smoothed targets and computes the loss
    using a manually implemented log-softmax with numerical stability.

    Soft target formula (Lecture 4 — label smoothing under randomness):
        y_smooth = (1 - epsilon) * y + epsilon / K
    where K is the number of classes and y is the input target distribution.

    Log-softmax computed from scratch with the max-subtraction trick:
        log_softmax(z) = z - max(z) - log( sum( exp(z - max(z)) ) )

    Loss = - (1/N) * sum_i sum_k [ y_smooth_ik * log_softmax(z_ik) ]

    Does NOT use nn.CrossEntropyLoss, F.cross_entropy, or F.log_softmax.

    Args:
        logits  (torch.Tensor): Raw model outputs, shape (N, K).
        targets (torch.Tensor): Target distribution (one-hot or soft labels
                                from MixUp), shape (N, K).
        epsilon (float):        Smoothing factor in [0, 1].  Default: 0.1.

    Returns:
        torch.Tensor: Scalar mean loss value.
    """
    num_classes = logits.size(1)

    # Apply label smoothing: (1-eps)*target + eps/K
    smooth_targets = (1.0 - epsilon) * targets + epsilon / num_classes

    # Manual log-softmax with numerical stability
    max_logits = logits.max(dim=1, keepdim=True).values
    shifted = logits - max_logits
    log_sum_exp = torch.log(shifted.exp().sum(dim=1, keepdim=True))
    log_probs = shifted - log_sum_exp

    # Cross-entropy: -mean over batch of sum over classes
    loss = -(smooth_targets * log_probs).sum(dim=1).mean()

    return loss


# ──────────────────────────────────────────────────────────────────────────────
# Training utilities
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(model, loader, criterion, device):
    """Compute average loss and accuracy on a dataset.

    Uses standard cross-entropy for loss computation, providing a
    consistent scale for early stopping comparison between models.

    Args:
        model     (nn.Module):    Model (set to eval mode internally).
        loader    (DataLoader):   Provides (images, labels) batches.
        criterion (nn.Module):    Loss function (typically nn.CrossEntropyLoss).
        device    (torch.device): Computation device.

    Returns:
        (float, float): (mean_loss, accuracy) where accuracy is in [0, 1].
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item() * images.size(0)
            correct += logits.argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one standard training epoch.

    Args:
        model     (nn.Module):            Model.
        loader    (DataLoader):           Training data.
        criterion (nn.Module):            Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device    (torch.device):         Computation device.

    Returns:
        (float, float): (mean_loss, accuracy).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += logits.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def train_one_epoch_mixup(model, loader, optimizer, device,
                          alpha=0.2, epsilon=0.1, num_classes=10):
    """Run one training epoch with MixUp augmentation and Label Smoothing.

    Each batch is augmented with MixUp before computing the label-smoothed
    cross-entropy loss.  Training accuracy is approximate (compared against
    the dominant class in the mixed label).

    Args:
        model       (nn.Module):            Model.
        loader      (DataLoader):           Training data.
        optimizer   (torch.optim.Optimizer): Optimizer.
        device      (torch.device):         Computation device.
        alpha       (float):               MixUp Beta parameter.  Default: 0.2.
        epsilon     (float):               Label smoothing factor.  Default: 0.1.
        num_classes (int):                 Number of classes.  Default: 10.

    Returns:
        (float, float): (mean_loss, approximate_accuracy).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        # Apply MixUp augmentation
        mixed_x, mixed_y, lam, _ = mixup_data(
            images, labels, num_classes=num_classes, alpha=alpha)

        optimizer.zero_grad()
        logits = model(mixed_x)
        loss = label_smoothing_cross_entropy(logits, mixed_y, epsilon)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        # Approximate accuracy: compare prediction to dominant mixed class
        correct += logits.argmax(1).eq(mixed_y.argmax(1)).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def train_model(model, train_loader, val_loader, optimizer, device,
                num_epochs, patience, use_mixup=False, alpha=0.2,
                epsilon=0.1, num_classes=10):
    """Training loop with manual validation-based early stopping.

    Monitors validation loss (standard cross-entropy) and stops when no
    improvement is observed for *patience* consecutive epochs.  Restores
    the best model weights after stopping.

    Args:
        model        (nn.Module):            Model to train.
        train_loader (DataLoader):           Training data.
        val_loader   (DataLoader):           Validation data.
        optimizer    (torch.optim.Optimizer): Optimizer.
        device       (torch.device):         Computation device.
        num_epochs   (int):                  Maximum number of epochs.
        patience     (int):                  Early stopping patience.
        use_mixup    (bool):  Use MixUp + Label Smoothing.  Default: False.
        alpha        (float): MixUp Beta parameter.         Default: 0.2.
        epsilon      (float): Label smoothing factor.       Default: 0.1.
        num_classes  (int):   Number of classes.            Default: 10.

    Returns:
        dict: Training history with keys 'train_loss', 'train_acc',
              'val_loss', 'val_acc' (lists), and 'stopped_epoch' (int).
    """
    criterion = nn.CrossEntropyLoss()
    history = {k: [] for k in ('train_loss', 'train_acc', 'val_loss', 'val_acc')}

    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        # Training phase
        if use_mixup:
            tr_loss, tr_acc = train_one_epoch_mixup(
                model, train_loader, optimizer, device,
                alpha, epsilon, num_classes)
        else:
            tr_loss, tr_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device)

        # Validation phase (standard CE for fair early stopping)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"  Epoch [{epoch+1:3d}/{num_epochs}]  "
              f"Train Loss: {tr_loss:.4f}  Train Acc: {tr_acc:.4f}  "
              f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n  Early stopping at epoch {epoch+1} "
                      f"(no val_loss improvement for {patience} epochs)")
                break

    # Restore best model weights
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  Restored best model (val_loss = {best_val_loss:.4f})")

    history['stopped_epoch'] = epoch + 1
    return history


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """Download CIFAR-10, train baseline and MixUp+LS models with early
    stopping, and save both models and training history."""

    # ── reproducibility ───────────────────────────────────────────────────
    torch.manual_seed(42)

    # ── configuration ─────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    num_epochs  = 100
    batch_size  = 128
    lr          = 1e-3
    num_classes = 10
    alpha       = 0.2       # MixUp Beta parameter
    epsilon     = 0.1       # Label smoothing factor
    patience    = 10        # Early stopping patience

    print(f"Device: {device}")
    print(f"Max Epochs: {num_epochs}  Batch: {batch_size}  LR: {lr}")
    print(f"MixUp alpha: {alpha}  Label Smoothing epsilon: {epsilon}")
    print(f"Early Stopping patience: {patience}\n")

    # ── data ──────────────────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    full_train = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    test_data  = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)

    train_set, val_set = torch.utils.data.random_split(
        full_train, [45000, 5000],
        generator=torch.Generator().manual_seed(42))

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    # ── baseline model (standard CE, no MixUp / Label Smoothing) ─────────
    print("=" * 70)
    print("BASELINE MODEL  (standard cross-entropy, no MixUp / Label Smoothing)")
    print("=" * 70)
    torch.manual_seed(42)
    baseline = ConvNet(num_classes=num_classes, dropout_rate=0.3).to(device)
    param_count = sum(p.numel() for p in baseline.parameters())
    print(f"Parameters: {param_count:,}\n")

    bl_optim = torch.optim.Adam(baseline.parameters(), lr=lr)
    bl_hist  = train_model(
        baseline, train_loader, val_loader, bl_optim, device,
        num_epochs=num_epochs, patience=patience)

    _, bl_test = evaluate(baseline, test_loader, criterion, device)
    print(f"\n  -> Baseline test accuracy: {bl_test:.4f}")

    # ── MixUp + Label Smoothing model ────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"MIXUP + LABEL SMOOTHING MODEL  (alpha={alpha}, epsilon={epsilon})")
    print("=" * 70)
    torch.manual_seed(42)
    mixup_model = ConvNet(num_classes=num_classes, dropout_rate=0.3).to(device)
    print(f"Parameters: {param_count:,}\n")

    mx_optim = torch.optim.Adam(mixup_model.parameters(), lr=lr)
    mx_hist  = train_model(
        mixup_model, train_loader, val_loader, mx_optim, device,
        num_epochs=num_epochs, patience=patience,
        use_mixup=True, alpha=alpha, epsilon=epsilon, num_classes=num_classes)

    _, mx_test = evaluate(mixup_model, test_loader, criterion, device)
    print(f"\n  -> MixUp+LS test accuracy: {mx_test:.4f}")

    # ── save models ──────────────────────────────────────────────────────
    torch.save({
        'baseline_state': baseline.state_dict(),
        'mixup_state':    mixup_model.state_dict(),
    }, 'models.pth')
    print("\n  -> Saved models.pth")

    # ── save training history ────────────────────────────────────────────
    history = {
        'baseline':          bl_hist,
        'mixup':             mx_hist,
        'baseline_test_acc': bl_test,
        'mixup_test_acc':    mx_test,
        'param_count':       param_count,
        'config': {
            'num_epochs':  num_epochs,
            'batch_size':  batch_size,
            'lr':          lr,
            'alpha':       alpha,
            'epsilon':     epsilon,
            'patience':    patience,
        },
    }
    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print("  -> Saved training_history.json")
    print("Done.\n")


if __name__ == '__main__':
    main()
