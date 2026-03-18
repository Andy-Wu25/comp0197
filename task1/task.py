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
Claude was used to assist with code structuring, Pillow-based
plotting, and drafting the technical analysis. All model design decisions,
hyperparameter choices, and theoretical content were verified by the author
against the COMP0197 lecture material. One specific correction: Claude initially
omitted model.eval() during per-epoch validation, which would have left dropout
active and artificially depressed the regularized model's validation accuracy.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import json
from PIL import Image, ImageDraw, ImageFont

from train import DeepNetwork, evaluate


# ──────────────────────────────────────────────────────────────────────────────
# Pillow-based plotting  (matplotlib is NOT allowed in submission)
# ──────────────────────────────────────────────────────────────────────────────

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


def create_generalization_gap_plot(history, filename='generalization_gap.png'):
    """Render training-vs-validation accuracy curves for both models as a PNG.

    Draws four curves (baseline train/val, regularized train/val) on a
    single plot with annotated generalization gaps.  Uses only Pillow.

    Args:
        history  (dict): Must contain 'baseline' and 'regularized' sub-dicts
                         each with 'train_acc' and 'val_acc' lists (float).
        filename (str):  Output PNG path.  Default: 'generalization_gap.png'.
    """
    # ── canvas layout ────────────────────────────────────────────────────────
    W, H = 920, 560
    margin = {'l': 75, 'r': 210, 't': 55, 'b': 60}
    pw = W - margin['l'] - margin['r']     # plot width in pixels
    ph = H - margin['t'] - margin['b']     # plot height in pixels

    img  = Image.new('RGB', (W, H), '#FFFFFF')
    draw = ImageDraw.Draw(img)

    font_sm    = _load_font(12)
    font_md    = _load_font(14)
    font_title = _load_font(18)

    # ── data preparation ─────────────────────────────────────────────────────
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

    # ── grid lines + y-axis labels ───────────────────────────────────────────
    n_grid = 10
    for i in range(n_grid + 1):
        yv = y_lo + (y_hi - y_lo) * i / n_grid
        py = ypx(yv)
        draw.line([(margin['l'], py), (margin['l'] + pw, py)],
                  fill='#E0E0E0', width=1)
        draw.text((margin['l'] - 55, py - 7), f"{yv:.2f}",
                  fill='#333333', font=font_sm)

    # ── x-axis tick labels ───────────────────────────────────────────────────
    tick = max(1, n_epochs // 10)
    for e in range(1, n_epochs + 1, tick):
        px = xpx(e)
        draw.line([(px, margin['t'] + ph), (px, margin['t'] + ph + 5)],
                  fill='#333333', width=1)
        draw.text((px - 8, margin['t'] + ph + 10), str(e),
                  fill='#333333', font=font_sm)

    # ── plot border ──────────────────────────────────────────────────────────
    draw.rectangle([margin['l'], margin['t'],
                    margin['l'] + pw, margin['t'] + ph],
                   outline='#333333', width=2)

    # ── plot each series ─────────────────────────────────────────────────────
    for _, data, colour in series:
        pts = [(xpx(i + 1), ypx(v)) for i, v in enumerate(data)]
        for j in range(len(pts) - 1):
            draw.line([pts[j], pts[j + 1]], fill=colour, width=2)
        for j in range(0, len(pts), 5):
            x, y = pts[j]
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=colour)

    # ── legend ───────────────────────────────────────────────────────────────
    lx = margin['l'] + pw + 18
    ly = margin['t'] + 15
    for i, (name, _, colour) in enumerate(series):
        y = ly + i * 28
        draw.rectangle([lx, y, lx + 16, y + 14], fill=colour)
        draw.text((lx + 22, y - 1), name, fill='#333333', font=font_sm)

    # ── annotate gaps ────────────────────────────────────────────────────────
    bl_gap = (history['baseline']['train_acc'][-1]
              - history['baseline']['val_acc'][-1])
    rg_gap = (history['regularized']['train_acc'][-1]
              - history['regularized']['val_acc'][-1])
    gy = ly + len(series) * 28 + 18
    draw.text((lx, gy),      f"Baseline gap:  {bl_gap:.3f}",
              fill='#D32F2F', font=font_sm)
    draw.text((lx, gy + 22), f"Reg. gap:      {rg_gap:.3f}",
              fill='#1565C0', font=font_sm)

    # ── title & axis labels ──────────────────────────────────────────────────
    draw.text((margin['l'] + pw // 2 - 150, 12),
              "Generalization Gap: Training vs Validation Accuracy",
              fill='#111111', font=font_title)
    draw.text((margin['l'] + pw // 2 - 15, H - 28),
              "Epoch", fill='#333333', font=font_md)
    for i, ch in enumerate("Accuracy"):
        draw.text((12, margin['t'] + ph // 2 - 55 + i * 16),
                  ch, fill='#333333', font=font_md)

    img.save(filename)
    print(f"Saved: {filename}")


# ──────────────────────────────────────────────────────────────────────────────
# Technical analysis
# ──────────────────────────────────────────────────────────────────────────────

def print_technical_analysis(history):
    """Print ~500-word technical analysis of the generalization gap.

    Discusses the observed gap, implicit regularization through optimizer
    choice, explicit regularization techniques (BatchNorm, Dropout, weight
    decay), hyperparameter justification, and the bias-variance trade-off.
    All claims are grounded in COMP0197 lecture material.

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
TECHNICAL ANALYSIS — THE DYNAMICS OF GENERALIZATION
================================================================================

1. GENERALIZATION GAP

The generalization gap — the difference between training and validation accuracy
— quantifies overfitting. Our baseline model, a 6-layer MLP ({hdims},
{nparams_bl:,} parameters) with no explicit regularization, reaches {bl_tr:.1%}
training accuracy but only {bl_va:.1%} validation accuracy (gap = {bl_gap:.1%}).
Although the universal approximation theorem (Lecture 1) guarantees that even a
single hidden layer can approximate any function, depth enables more efficient
compositional feature learning — our 6-layer funnel architecture exploits this,
but the resulting capacity also enables severe overfitting when unconstrained,
placing the baseline firmly in the high-variance regime (Lecture 4).

The regularized model ({nparams_rg:,} parameters including BatchNorm) employing
BatchNorm, Dropout (p=0.3), and L2 weight decay (lambda=1e-3) achieves
{rg_tr:.1%} training / {rg_va:.1%} validation accuracy
(gap = {rg_gap:.1%}). Test-set results: baseline {bl_te:.1%} vs regularized
{rg_te:.1%}. We maintain strict train/validation/test separation to avoid data
leakage (Lecture 6).

2. IMPLICIT REGULARIZATION VIA OPTIMIZER CHOICE

Both models use SGD with momentum (beta={c['momentum']}) and mini-batch size
{c['batch_size']}. Mini-batch SGD provides implicit regularization through
gradient noise (Lecture 2). At each step the gradient is estimated from a random
subset, injecting noise that biases the optimizer toward flat minima — regions
where loss varies slowly under parameter perturbation — which are empirically
associated with better generalization. Momentum smooths the trajectory while
preserving stochastic exploration. Even the baseline benefits; full-batch
gradient descent would overfit more severely.

3. EXPLICIT REGULARIZATION TECHNIQUES

BatchNorm (Lecture 4, slides 17-18): Normalises each hidden layer's activations
to zero mean and unit variance within each mini-batch, then applies a learnable
affine transform. This introduces stochastic noise through batch statistics,
acting as an implicit regulariser. It also stabilises gradient flow through our
6-layer network and is recommended as a "no harm trick" (Lecture 6).

Dropout p={c['dropout_rate']} (Lecture 4): Randomly zeros {int(c['dropout_rate']*100)}%
of hidden activations, preventing feature co-adaptation. Dropout approximates an
ensemble of exponentially many sub-networks, reducing variance through model
averaging. At inference, all units are active with appropriately scaled outputs.

Weight decay lambda={c['weight_decay']} (Lecture 4): Adds lambda*||w||^2 to the
loss, penalising large weights and favouring smoother decision boundaries.
L2 gradient scales with w, so weights shrink without reaching exactly zero — the
network retains all features at reduced magnitude.

4. HYPERPARAMETER JUSTIFICATION

* Architecture [{hdims}]: funnel topology through 6 linear layers, encouraging
  hierarchical feature abstraction (Lecture 1). Appropriate weight
  initialization maintains stable gradient flow (Lecture 2).
* LR {c['lr']} + momentum {c['momentum']}: standard SGD configuration balancing
  convergence speed with stability (Lecture 2).
* Batch size {c['batch_size']}: sufficient gradient noise for implicit
  regularization while keeping variance manageable (Lecture 2).

5. BIAS-VARIANCE TRADE-OFF

The baseline operates in the low-bias, high-variance regime: it fits training
data well but generalises poorly (Lecture 4). BatchNorm, Dropout, and weight
decay constrain effective capacity, shifting toward higher bias and substantially
lower variance. The validation improvement ({rg_va:.1%} vs {bl_va:.1%}) confirms
that variance reduction outweighs the bias increase, yielding a more favourable
total error at the optimal operating point on the bias-variance curve.

================================================================================
GenAI Correction: Claude initially suggested omitting model.eval() in the
per-epoch validation loop. This would have left Dropout active during
validation, making the regularized model's validation accuracy artificially
low and the gap comparison misleading. The error was caught and corrected.
================================================================================
"""
    print(text)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """Load saved models, generate plot, and print analysis."""

    # ── load training history ────────────────────────────────────────────────
    with open('training_history.json', 'r') as f:
        history = json.load(f)
    cfg = history['config']

    # ── load models and evaluate on test set ─────────────────────────────────
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

    # baseline (no BatchNorm, no Dropout)
    baseline = DeepNetwork(
        input_dim, num_classes, cfg['hidden_dims'],
        dropout_rate=0.0, use_batchnorm=False)
    baseline.load_state_dict(
        torch.load('baseline_model.pth', map_location=device, weights_only=True))
    baseline.to(device)
    _, bl_test = evaluate(baseline, test_loader, criterion, device)

    # regularized (BatchNorm + Dropout)
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

    # update history with fresh test numbers
    history['baseline_test_acc'] = bl_test
    history['reg_test_acc']      = rg_test

    # ── generate plot ────────────────────────────────────────────────────────
    create_generalization_gap_plot(history, 'generalization_gap.png')

    # ── print technical analysis ─────────────────────────────────────────────
    print_technical_analysis(history)


if __name__ == '__main__':
    main()
