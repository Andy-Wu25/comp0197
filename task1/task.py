"""
COMP0197 Coursework 1 — Task 1: The Dynamics of Generalization

task.py
-------
Loads the saved baseline and regularized models, generates a
generalization_gap.png plot (using Pillow), and prints a ~500-word
technical analysis discussing the generalization gap, implicit
regularization through optimizer choice, and the bias-variance trade-off.

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
from PIL import Image, ImageDraw, ImageFont

from train import DeepNetwork, evaluate


# Pillow-based plotting
def _load_font(size):
    """Return a Pillow font object at the requested pixel size.

    Falls back to the default bitmap font on older Pillow versions that
    do not support the *size* parameter in load_default().

    Args:
        size (int): Desired font size in pixels.

    Returns:
        PIL.ImageFont.FreeTypeFont or PIL.ImageFont.ImageFont
    """
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _get_text_size(draw, text, font):
    """Measure rendered text width and height using textbbox.

    Args:
        draw (ImageDraw.Draw): Draw context.
        text (str):            Text to measure.
        font (ImageFont):      Font to use.

    Returns:
        (int, int): (width, height) in pixels.
    """
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def create_generalization_gap_plot(history, filename='generalization_gap.png'):
    """Render training-vs-validation accuracy curves for both models as a PNG.

    Draws four curves (baseline train/val, regularized train/val) on a
    single plot with annotated generalization gaps.  Uses only Pillow.

    Args:
        history  (dict): Must contain 'baseline' and 'regularized' sub-dicts
                         each with 'train_acc' and 'val_acc' lists (float).
        filename (str):  Output PNG path.  Default: 'generalization_gap.png'.
    """
    W, H = 920, 560
    margin = {'l': 75, 'r': 210, 't': 55, 'b': 60}
    pw = W - margin['l'] - margin['r']     
    ph = H - margin['t'] - margin['b']    

    img  = Image.new('RGB', (W, H), '#FFFFFF')
    draw = ImageDraw.Draw(img)

    font_sm    = _load_font(12)
    font_md    = _load_font(14)
    font_title = _load_font(18)

    series = [
        ('Baseline Train',    history['baseline']['train_acc'],    '#D32F2F'),
        ('Baseline Val',      history['baseline']['val_acc'],      '#EF9A9A'),
        ('Regularized Train', history['regularized']['train_acc'], '#1565C0'),
        ('Regularized Val',   history['regularized']['val_acc'],   '#90CAF9'),
    ]

    n_epochs = len(series[0][1])
    all_vals = [v for _, data, _ in series for v in data]
    y_lo = max(0.0, min(all_vals) - 0.05)
    y_hi = min(1.0, max(all_vals) + 0.05)

    def xpx(epoch):
        """Map 1-indexed epoch to pixel x."""
        return margin['l'] + int((epoch - 1) / max(n_epochs - 1, 1) * pw)

    def ypx(val):
        """Map accuracy value to pixel y."""
        return margin['t'] + int((1.0 - (val - y_lo) / (y_hi - y_lo)) * ph)

    n_grid = 10
    for i in range(n_grid + 1):
        yv = y_lo + (y_hi - y_lo) * i / n_grid
        py = ypx(yv)
        draw.line([(margin['l'], py), (margin['l'] + pw, py)],
                  fill='#E0E0E0', width=1)
        draw.text((margin['l'] - 55, py - 7), f"{yv:.2f}",
                  fill='#333333', font=font_sm)

    tick = max(1, n_epochs // 10)
    for e in range(1, n_epochs + 1, tick):
        px = xpx(e)
        draw.line([(px, margin['t'] + ph), (px, margin['t'] + ph + 5)],
                  fill='#333333', width=1)
        draw.text((px - 8, margin['t'] + ph + 10), str(e),
                  fill='#333333', font=font_sm)

    draw.rectangle([margin['l'], margin['t'],
                    margin['l'] + pw, margin['t'] + ph],
                   outline='#333333', width=2)

    for _, data, colour in series:
        pts = [(xpx(i + 1), ypx(v)) for i, v in enumerate(data)]
        for j in range(len(pts) - 1):
            draw.line([pts[j], pts[j + 1]], fill=colour, width=2)
        for j in range(0, len(pts), 5):
            x, y = pts[j]
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=colour)

    lx = margin['l'] + pw + 18
    ly = margin['t'] + 15
    for i, (name, _, colour) in enumerate(series):
        y = ly + i * 28
        draw.rectangle([lx, y, lx + 16, y + 14], fill=colour)
        draw.text((lx + 22, y - 1), name, fill='#333333', font=font_sm)

    bl_gap = (history['baseline']['train_acc'][-1]
              - history['baseline']['val_acc'][-1])
    rg_gap = (history['regularized']['train_acc'][-1]
              - history['regularized']['val_acc'][-1])
    gy = ly + len(series) * 28 + 18
    draw.text((lx, gy),      f"Baseline gap:  {bl_gap:.3f}",
              fill='#D32F2F', font=font_sm)
    draw.text((lx, gy + 22), f"Reg. gap:      {rg_gap:.3f}",
              fill='#1565C0', font=font_sm)

    title_str = "Generalization Gap: Training vs Validation Accuracy"
    tw, _ = _get_text_size(draw, title_str, font_title)
    draw.text(((W - tw) // 2, 12), title_str,
              fill='#111111', font=font_title)

    x_label = "Epoch"
    xw, _ = _get_text_size(draw, x_label, font_md)
    draw.text(((margin['l'] + margin['l'] + pw - xw) // 2, H - 28),
              x_label, fill='#333333', font=font_md)

    for i, ch in enumerate("Accuracy"):
        draw.text((12, margin['t'] + ph // 2 - 55 + i * 16),
                  ch, fill='#333333', font=font_md)

    img.save(filename)
    print(f"Saved: {filename}")


# Technical analysis
def print_technical_analysis(history):
    """Print ~500-word technical analysis of the generalization gap.

    Discusses the observed gap, implicit regularization through optimizer
    choice, explicit regularization techniques (data augmentation, BatchNorm,
    Dropout, weight decay), hyperparameter justification, and the bias-variance
    trade-off.

    Args:
        history (dict): Full training history including 'config',
                        'baseline_test_acc', 'reg_test_acc',
                        'baseline_param_count', and 'reg_param_count'.
    """
    c  = history['config']
    bl = history['baseline']
    rg = history['regularized']

    bl_tr  = bl['train_acc'][-1]
    bl_va  = bl['val_acc'][-1]
    rg_tr  = rg['train_acc'][-1]
    rg_va  = rg['val_acc'][-1]
    bl_gap = bl_tr - bl_va
    rg_gap = rg_tr - rg_va
    bl_te  = history['baseline_test_acc']
    rg_te  = history['reg_test_acc']
    nparams_bl = history['baseline_param_count']
    nparams_rg = history['reg_param_count']
    hdims  = ' -> '.join(map(str, c['hidden_dims']))

    text = f"""\
================================================================================
Technical Justification — The Dynamics of Generalization
================================================================================

1. The Generalization Gap

Our baseline is a 6-layer MLP ({hdims}, {nparams_bl:,} parameters) trained with
no explicit regularization. It reaches {bl_tr:.1%} training accuracy but only
{bl_va:.1%} validation accuracy, having a gap of {bl_gap:.1%}. This shows
overfitting where the model has enough capacity to memorise the training set but 
the learned function does not transfer to unseen data.

The regularised model ({nparams_rg:,} parameters) applies data augmentation,
BatchNorm, Dropout, and weight decay. It achieves {rg_tr:.1%} train / {rg_va:.1%}
val (gap = {rg_gap:.1%}). Test accuracy: baseline {bl_te:.1%} vs regularised
{rg_te:.1%}. Train/val/test splits are strictly separate.

2. Implicit regularization via the optimizer

Both models use mini-batch SGD with momentum {c['momentum']} and batch size
{c['batch_size']}. Mini-batch SGD estimates gradients from a random subset of
the data, injecting noise into each update. This noise biases the optimizer toward 
flat minima (regions where the loss surface varies slowly under small parameter 
perturbations) which tend to generalise better than sharp minima. Even the baseline 
benefits from this as full-batch gradient descent would overfit more severely.

We apply a StepLR schedule that halves the learning rate every 15 epochs. This lets 
the optimizer take large steps early for fast progress, then smaller steps to settle 
into a minimum rather than oscillating around it. The baseline's validation loss keeps 
rising after epoch 15 (1.66 to 4.55 by epoch 50), confirming that implicit regularization 
alone cannot control overfitting in a high-capacity network.

3. Hyperparameter Justification

Architecture [{hdims}]: A funnel topology that progressively compresses
representations across 6 layers. Depth enables compositional feature learning
while the narrowing forces the network to distil information.

Data augmentation: RandomCrop with 4px padding, RandomHorizontalFlip, and ColorJitter (0.2)
apply spatial and colour perturbations, expanding the effective training distribution.
Training accuracy is measured on clean data for an honest gap measurement.

BatchNorm: Normalises activations to zero mean and unit variance
within each mini-batch, then applies a learned affine transform. This stabilises
gradient flow through our 6 layers and introduces stochastic noise via batch
statistics, acting as implicit regularization.

Dropout p={c['dropout_rate']}: Randomly zeros {int(c['dropout_rate']*100)}% of activations
each forward pass. This prevents co-adaptation and approximates an ensemble of
sub-networks, reducing variance.

Weight decay {c['weight_decay']}: Adds lambda*||w||^2 to the loss.
The L2 gradient scales with w, so weights shrink toward zero without becoming
exactly sparse — favouring smoother decision boundaries.

LR={c['lr']}, momentum={c['momentum']}, batch size={c['batch_size']}: Standard
SGD settings. The batch size is small enough to maintain gradient noise for
regularization but large enough for stable training.

4. Bias-Variance Trade-off

The baseline sits in the high-variance regime: {bl_tr:.1%} train vs {bl_va:.1%}
val shows it fits noise in the training set rather than the underlying pattern.
Each regularization technique constrains effective capacity, trading some ability
to fit training data (higher bias) for much lower sensitivity to the specific training
sample (lower variance). The result {rg_va:.1%} val vs {bl_va:.1%} confirms the variance
reduction more than compensates for the bias increase, moving the model closer to
the optimal point on the bias-variance curve.
"""
    print(text)


# Main
def main():
    """Load saved models, generate plot, and print analysis."""

    # Training configuration and per-epoch history (produced by train.py)
    cfg = {
        'num_epochs': 50, 'batch_size': 128, 'lr': 0.01, 'momentum': 0.9,
        'hidden_dims': [1024, 512, 512, 256, 128],
        'dropout_rate': 0.3, 'weight_decay': 0.001,
        'use_batchnorm': True, 'use_augmentation': True,
        'lr_scheduler': 'StepLR(step_size=15, gamma=0.5)',
    }
    history = {
        'baseline': {
            'train_acc': [
                0.3579, 0.4644, 0.5212, 0.5627, 0.5878, 0.6331, 0.6581, 0.6890, 0.6938, 0.7426,
                0.7586, 0.7905, 0.8024, 0.8315, 0.8434, 0.9376, 0.9521, 0.9474, 0.9520, 0.9538,
                0.9558, 0.9606, 0.9529, 0.9624, 0.9570, 0.9676, 0.9677, 0.9767, 0.9820, 0.9712,
                0.9985, 0.9998, 0.9995, 0.9999, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            ],
            'val_acc': [
                0.3562, 0.4426, 0.4864, 0.5036, 0.5106, 0.5264, 0.5244, 0.5366, 0.5262, 0.5486,
                0.5286, 0.5404, 0.5392, 0.5436, 0.5336, 0.5530, 0.5592, 0.5502, 0.5524, 0.5374,
                0.5430, 0.5386, 0.5458, 0.5464, 0.5374, 0.5522, 0.5506, 0.5474, 0.5548, 0.5420,
                0.5632, 0.5628, 0.5632, 0.5608, 0.5600, 0.5622, 0.5596, 0.5594, 0.5588, 0.5598,
                0.5590, 0.5604, 0.5600, 0.5598, 0.5604, 0.5594, 0.5586, 0.5594, 0.5598, 0.5600,
            ],
        },
        'regularized': {
            'train_acc': [
                0.3660, 0.4007, 0.4226, 0.4507, 0.4468, 0.4649, 0.4727, 0.4891, 0.4908, 0.4825,
                0.4972, 0.5051, 0.5093, 0.5133, 0.5110, 0.5306, 0.5356, 0.5368, 0.5362, 0.5498,
                0.5495, 0.5468, 0.5465, 0.5525, 0.5565, 0.5581, 0.5632, 0.5622, 0.5628, 0.5633,
                0.5709, 0.5765, 0.5749, 0.5719, 0.5817, 0.5779, 0.5854, 0.5838, 0.5807, 0.5875,
                0.5853, 0.5892, 0.5918, 0.5899, 0.5917, 0.5960, 0.5979, 0.6022, 0.6020, 0.6010,
            ],
            'val_acc': [
                0.3650, 0.4000, 0.4220, 0.4476, 0.4458, 0.4684, 0.4656, 0.4856, 0.4848, 0.4760,
                0.4940, 0.4960, 0.5054, 0.5024, 0.5078, 0.5242, 0.5236, 0.5290, 0.5262, 0.5436,
                0.5468, 0.5414, 0.5370, 0.5464, 0.5490, 0.5516, 0.5504, 0.5518, 0.5526, 0.5470,
                0.5580, 0.5652, 0.5634, 0.5564, 0.5724, 0.5704, 0.5764, 0.5688, 0.5706, 0.5750,
                0.5742, 0.5780, 0.5722, 0.5728, 0.5796, 0.5802, 0.5840, 0.5844, 0.5874, 0.5804,
            ],
        },
        'config': cfg,
    }

    # Load models and evaluate on test set
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    input_dim   = 3 * 32 * 32
    num_classes = 10

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    test_data = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=128, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    # Baseline (no BatchNorm, no Dropout)
    baseline = DeepNetwork(
        input_dim, num_classes, cfg['hidden_dims'],
        dropout_rate=0.0, use_batchnorm=False)
    baseline.load_state_dict(
        torch.load('baseline_model.pth', map_location=device, weights_only=True))
    baseline.to(device)
    _, bl_test = evaluate(baseline, test_loader, criterion, device)

    # Regularized (data augmentation + BatchNorm + Dropout)
    regularized = DeepNetwork(
        input_dim, num_classes, cfg['hidden_dims'],
        dropout_rate=cfg['dropout_rate'],
        use_batchnorm=cfg['use_batchnorm'])
    regularized.load_state_dict(
        torch.load('regularized_model.pth', map_location=device, weights_only=True))
    regularized.to(device)
    _, rg_test = evaluate(regularized, test_loader, criterion, device)

    print(f"Baseline     test accuracy: {bl_test:.4f}")
    print(f"Regularized  test accuracy: {rg_test:.4f}\n")

    history['baseline_test_acc'] = bl_test
    history['reg_test_acc']      = rg_test
    history['baseline_param_count'] = sum(p.numel() for p in baseline.parameters())
    history['reg_param_count'] = sum(p.numel() for p in regularized.parameters())

    create_generalization_gap_plot(history, 'generalization_gap.png')
    print_technical_analysis(history)


if __name__ == '__main__':
    main()
