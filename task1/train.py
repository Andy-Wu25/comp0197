"""
COMP0197 Coursework 1 — Task 1: The Dynamics of Generalization

train.py
--------
Downloads CIFAR-10, builds baseline and regularized deep feedforward networks,
trains both, and saves model weights and per-epoch training history.

GenAI Usage Statement
---------------------
Claude was used as an assistive tool for code and terminal output structuring 
and Pillow-based visualisation. All code implemented by Claude was reviewed, 
tested, and edited by the author to ensure correctness.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import json



# Model Selection
class DeepNetwork(nn.Module):
    """Deep feedforward network built exclusively from nn.Linear primitives.

    Constructs a multi-layer perceptron with ReLU activations, optional
    batch normalisation after each hidden linear layer, and optional
    dropout after each hidden activation.

    Architecture pattern per hidden layer:
        Linear -> [BatchNorm1d] -> ReLU -> [Dropout]

    Args:
        input_dim     (int):       Flattened input size (e.g. 3072 for CIFAR-10).
        num_classes   (int):       Number of output classes.
        hidden_dims   (list[int]): Sizes of each hidden layer.
        dropout_rate  (float):     Dropout probability after each hidden layer.
                                   0.0 disables dropout.  Default: 0.0.
        use_batchnorm (bool):      Whether to insert BatchNorm1d after each
                                   hidden Linear layer.  Default: False.
    """

    def __init__(self, input_dim, num_classes, hidden_dims,
                 dropout_rate=0.0, use_batchnorm=False):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(p=dropout_rate))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): Float32 tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes).
        """
        return self.network(x)

def evaluate(model, loader, criterion, device):
    """Compute average loss and accuracy on a dataset.

    Args:
        model     (nn.Module):    Model (set to eval mode internally).
        loader    (DataLoader):   Provides (images, labels) batches.
        criterion (nn.Module):    Loss function.
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
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item() * images.size(0)
            correct += logits.argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one training epoch.

    Args:
        model     (nn.Module):            Model (set to train mode internally).
        loader    (DataLoader):           Training data.
        criterion (nn.Module):            Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device    (torch.device):         Computation device.

    Returns:
        (float, float): (mean_training_loss, training_accuracy).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.view(images.size(0), -1).to(device)
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


def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs, device, scheduler=None, train_eval_loader=None):
    """Full training loop with per-epoch validation logging.

    Args:
        model            (nn.Module):            Model to train.
        train_loader     (DataLoader):           Training split.
        val_loader       (DataLoader):           Validation split.
        criterion        (nn.Module):            Loss function.
        optimizer        (torch.optim.Optimizer): Optimizer.
        num_epochs       (int):                  Number of epochs.
        device           (torch.device):         Computation device.
        scheduler        (lr_scheduler, optional): Learning rate scheduler.
                                                   Default: None.
        train_eval_loader (DataLoader, optional): Clean (non-augmented) loader
                         over training set for accurate train accuracy
                         measurement.

    Returns:
        dict: Keys 'train_loss', 'train_acc', 'val_loss', 'val_acc',
              each a list[float] of length num_epochs.
    """
    history = {k: [] for k in ('train_loss', 'train_acc', 'val_loss', 'val_acc')}

    for epoch in range(num_epochs):
        train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Evaluate training accuracy on clean data
        if train_eval_loader is not None:
            tr_loss, tr_acc = evaluate(
                model, train_eval_loader, criterion, device)
        else:
            tr_loss, tr_acc = evaluate(
                model, train_loader, criterion, device)

        va_loss, va_acc = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(va_loss)
        history['val_acc'].append(va_acc)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch [{epoch+1:3d}/{num_epochs}]  LR: {current_lr:.5f}  "
              f"Train Loss: {tr_loss:.4f}  Train Acc: {tr_acc:.4f}  "
              f"Val Loss: {va_loss:.4f}  Val Acc: {va_acc:.4f}")

        if scheduler is not None:
            scheduler.step()

    return history



# Main
def main():
    """Download CIFAR-10, train baseline & regularized models, save artefacts."""
    torch.manual_seed(42)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    num_epochs   = 50
    batch_size   = 128
    lr           = 0.01
    momentum     = 0.9
    input_dim    = 3 * 32 * 32          
    num_classes  = 10
    hidden_dims  = [1024, 512, 512, 256, 128]   

    print(f"Device: {device}")
    print(f"Architecture: {input_dim} -> {' -> '.join(map(str, hidden_dims))} -> {num_classes}")
    print(f"Epochs: {num_epochs}  Batch: {batch_size}  LR: {lr}  Momentum: {momentum}\n")

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    # Data augmentation for regularized model
    aug_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    full_train = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=eval_transform)
    full_train_aug = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=False, transform=aug_transform)
    test_data = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=eval_transform)

    # Same split indices for both augmented and non-augmented datasets
    gen = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(full_train), generator=gen).tolist()
    train_indices = indices[:45000]
    val_indices = indices[45000:]

    train_set = torch.utils.data.Subset(full_train, train_indices)
    train_set_aug = torch.utils.data.Subset(full_train_aug, train_indices)
    val_set = torch.utils.data.Subset(full_train, val_indices)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True)
    train_loader_aug = torch.utils.data.DataLoader(
        train_set_aug, batch_size=batch_size, shuffle=True)
    
    # Clean evaluation loader over training set
    train_eval_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    # Baseline model
    print("=" * 70)
    print("BASELINE MODEL  (no regularization)")
    print("=" * 70)
    torch.manual_seed(42)
    baseline = DeepNetwork(input_dim, num_classes, hidden_dims,
                           dropout_rate=0.0, use_batchnorm=False)
    baseline.to(device)
    bl_param_count = sum(p.numel() for p in baseline.parameters())
    print(f"Parameters: {bl_param_count:,}\n")

    bl_optim = torch.optim.SGD(baseline.parameters(), lr=lr, momentum=momentum)
    
    # LR scheduler: halve learning rate every 15 epochs
    bl_sched = torch.optim.lr_scheduler.StepLR(bl_optim, step_size=15, gamma=0.5)
    bl_hist  = train_model(
        baseline, train_loader, val_loader, criterion, bl_optim,
        num_epochs, device, scheduler=bl_sched)

    _, bl_test = evaluate(baseline, test_loader, criterion, device)
    print(f"\n  -> Baseline test accuracy: {bl_test:.4f}")
    torch.save(baseline.state_dict(), 'baseline_model.pth')
    print("  -> Saved baseline_model.pth\n")

    # Regularized model
    print("=" * 70)
    print("REGULARIZED MODEL  (Augmentation+ColorJitter + BatchNorm + Dropout p=0.3 + WD 1e-3)")
    print("=" * 70)
    torch.manual_seed(42)
    reg_model = DeepNetwork(input_dim, num_classes, hidden_dims,
                            dropout_rate=0.3, use_batchnorm=True)
    reg_model.to(device)
    reg_param_count = sum(p.numel() for p in reg_model.parameters())
    print(f"Parameters: {reg_param_count:,}\n")

    reg_optim = torch.optim.SGD(
        reg_model.parameters(), lr=lr, momentum=momentum, weight_decay=1e-3)
    reg_sched = torch.optim.lr_scheduler.StepLR(reg_optim, step_size=15, gamma=0.5)

    reg_hist = train_model(
        reg_model, train_loader_aug, val_loader, criterion, reg_optim,
        num_epochs, device, scheduler=reg_sched,
        train_eval_loader=train_eval_loader)

    _, reg_test = evaluate(reg_model, test_loader, criterion, device)
    print(f"\n  -> Regularized test accuracy: {reg_test:.4f}")
    torch.save(reg_model.state_dict(), 'regularized_model.pth')
    print("  -> Saved regularized_model.pth\n")

    history = {
        'baseline':          bl_hist,
        'regularized':       reg_hist,
        'baseline_test_acc': bl_test,
        'reg_test_acc':      reg_test,
        'baseline_param_count': bl_param_count,
        'reg_param_count':      reg_param_count,
        'config': {
            'num_epochs':      num_epochs,
            'batch_size':      batch_size,
            'lr':              lr,
            'momentum':        momentum,
            'hidden_dims':     hidden_dims,
            'dropout_rate':    0.3,
            'weight_decay':    1e-3,
            'use_batchnorm':   True,
            'use_augmentation': True,
            'lr_scheduler':    'StepLR(step_size=15, gamma=0.5)',
        },
    }
    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print("Saved training_history.json")
    print("Done.\n")


if __name__ == '__main__':
    main()
