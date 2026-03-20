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
import json
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
sample (lower variance). The result — {rg_va:.1%} val vs {bl_va:.1%} — confirms the variance
reduction more than compensates for the bias increase, moving the model closer to
the optimal point on the bias-variance curve.
"""
    print(text)


# Main
def main():
    """Load saved models, generate plot, and print analysis."""

    # Load training history
    with open('training_history.json', 'r') as f:
        history = json.load(f)
    cfg = history['config']

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

    create_generalization_gap_plot(history, 'generalization_gap.png')
    print_technical_analysis(history)


if __name__ == '__main__':
    main()
