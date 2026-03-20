"""
COMP0197 Coursework 1 — Task 2: Robust Representation via MixUp and Label Smoothing

task.py
-------
Loads the saved baseline and MixUp+LS models, evaluates robustness under
Gaussian noise at multiple intensities, generates a robustness_demo.png
montage of 16 MixUp-processed images (using Pillow), and prints a ~500-word
technical justification comparing both models.

GenAI Usage Statement
---------------------
Claude was used to assist with code structuring and Pillow-based
visualisation. One specific correction: Claude
initially applied label smoothing only to hard targets, whereas the correct
approach applies smoothing on top of MixUp's already-soft labels during
training, ensuring both regularisation techniques compose correctly.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import json
from PIL import Image, ImageDraw, ImageFont

from train import ConvNet, evaluate

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck',
]


def _load_font(size):
    """Return a Pillow font object at the requested pixel size.

    Falls back to the default bitmap font on older Pillow versions.

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


# Noisy evaluation
def evaluate_noisy(model, loader, device, noise_std=0.1):
    """Evaluate model accuracy on data corrupted with additive Gaussian noise.

    For each batch, samples noise ~ N(0, noise_std^2) and adds it to the
    normalised input images before computing predictions.

    Args:
        model     (nn.Module):    Trained model.
        loader    (DataLoader):   Test data loader.
        device    (torch.device): Computation device.
        noise_std (float):        Standard deviation of Gaussian noise.
                                  Default: 0.1.

    Returns:
        float: Accuracy on the noisy inputs, in [0, 1].
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            # Add Gaussian noise
            noisy_images = images + noise_std * torch.randn_like(images)

            logits = model(noisy_images)
            correct += logits.argmax(1).eq(labels).sum().item()
            total += labels.size(0)

    return correct / total


# Robustness demo image
def create_robustness_demo(test_data, alpha=0.2, filename='robustness_demo.png'):
    """Create a 4x4 montage of 16 images processed by MixUp logic.

    Selects 32 test images, pairs them (image i with image i+16), blends
    each pair using a lambda sampled from Beta(alpha, alpha), and arranges
    the 16 mixed images in a grid.  Each cell shows the blended image
    with the mixing coefficient and the two source class names.

    Uses the same MixUp formula and Beta distribution as the training
    implementation.  Individual lambdas are sampled per cell to
    demonstrate the range of blending produced by Beta(alpha, alpha).

    Args:
        test_data (torchvision.datasets.CIFAR10): Test dataset
                  (with ToTensor + Normalize transform).
        alpha     (float): MixUp Beta distribution parameter.  Default: 0.2.
        filename  (str):   Output PNG path.  Default: 'robustness_demo.png'.
    """
    torch.manual_seed(123)

    indices = torch.randperm(len(test_data))[:32]
    images = []
    labels = []
    for idx in indices:
        img, lbl = test_data[idx.item()]
        images.append(img)
        labels.append(lbl)

    beta_dist = torch.distributions.Beta(
        torch.tensor(alpha), torch.tensor(alpha))

    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std  = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)

    cell_w, cell_h = 120, 120
    label_h  = 36
    padding  = 8
    cols, rows = 4, 4
    title_h = 40

    W = cols * (cell_w + padding) + padding
    H = title_h + rows * (cell_h + label_h + padding) + padding

    canvas = Image.new('RGB', (W, H), '#FFFFFF')
    draw   = ImageDraw.Draw(canvas)
    font_sm    = _load_font(11)
    font_title = _load_font(16)
    title_str = "MixUp Augmentation Demo (16 Samples)"
    tw, _ = _get_text_size(draw, title_str, font_title)
    draw.text(((W - tw) // 2, 10), title_str,
              fill='#111111', font=font_title)

    for i in range(16):
        row = i // cols
        col = i % cols

        x_pos = padding + col * (cell_w + padding)
        y_pos = title_h + padding + row * (cell_h + label_h + padding)

        # Sample lambda and blend pair (MixUp formula)
        lam = beta_dist.sample().item()
        mixed = lam * images[i] + (1.0 - lam) * images[i + 16]

        img_t = mixed * std + mean
        img_t = img_t.clamp(0.0, 1.0)

        pil_img = transforms.ToPILImage()(img_t)
        try:
            resample = Image.Resampling.NEAREST
        except AttributeError:
            resample = Image.NEAREST
        pil_img = pil_img.resize((cell_w, cell_h), resample)

        canvas.paste(pil_img, (x_pos, y_pos))

        draw.rectangle(
            [x_pos - 1, y_pos - 1, x_pos + cell_w, y_pos + cell_h],
            outline='#CCCCCC', width=1)

        cls_a = CIFAR10_CLASSES[labels[i]]
        cls_b = CIFAR10_CLASSES[labels[i + 16]]
        caption = f"lam={lam:.2f}: {cls_a}+{cls_b}"
        draw.text((x_pos + 2, y_pos + cell_h + 3),
                  caption, fill='#333333', font=font_sm)

    canvas.save(filename)
    print(f"Saved: {filename}")


# Technical analysis
def print_technical_analysis(bl_clean, mx_clean, bl_noisy, mx_noisy, config):
    """Print ~500-word technical justification covering MixUp and Label Smoothing.

    Discusses (1) why MixUp prevents memorisation via vicinal risk minimisation,
    (2) how Label Smoothing prevents overshooting by bounding optimal logits,
    and (3) quantitative robustness results comparing baseline and MixUp+LS.

    Args:
        bl_clean  (float): Baseline clean test accuracy.
        mx_clean  (float): MixUp+LS clean test accuracy.
        bl_noisy  (dict):  Mapping {noise_std (float): accuracy (float)}
                           for the baseline model.
        mx_noisy  (dict):  Mapping {noise_std (float): accuracy (float)}
                           for the MixUp+LS model.
        config    (dict):  Training configuration dictionary.
    """
    
    table_lines = []
    for sigma in sorted(bl_noisy.keys()):
        table_lines.append(
            f"    {sigma:.2f}        {bl_noisy[sigma]:.1%}          {mx_noisy[sigma]:.1%}")
    table = "\n".join(table_lines)

    eps = config['epsilon']
    # Finite optimal logit: z* = log((1-eps+eps/K) / (eps/K))
    target_correct = 1.0 - eps + eps / 10.0
    target_other = eps / 10.0
    optimal_logit = torch.tensor(target_correct / target_other).log().item()

    text = f"""\
================================================================================
TECHNICAL JUSTIFICATION — MIXUP AND LABEL SMOOTHING FOR ROBUSTNESS
================================================================================

1. WHY MIXUP PREVENTS MEMORISATION

Standard empirical risk minimisation (ERM) minimises the expected loss under the
training data distribution p_data(x, y) = (1/M) * sum delta(x = x_m, y = y_m),
which places all probability mass on individual training samples (Lecture 4: data
resampling). A high-capacity network can drive the training loss to zero by
memorising each sample, producing sharp, oscillating decision boundaries that
generalise poorly.

MixUp replaces ERM with vicinal risk minimisation (VRM), where the vicinity
distribution blends training pairs (Lecture 4: mixup regularisation):
    x_tilde = lambda * x_i + (1 - lambda) * x_j
    y_tilde = lambda * y_i + (1 - lambda) * y_j
with lambda ~ Beta({config['alpha']}, {config['alpha']}).

This prevents memorisation through three mechanisms. First, it generates
synthetic training points between existing samples, expanding the support of the
training distribution. A model that memorises individual samples cannot produce
coherent predictions at these intermediate points, so memorisation is directly
penalised. Second, the linear label blending incentivises predictions that vary
linearly between training samples, encouraging smooth, low-complexity decision
boundaries — a direct reduction of model variance (Lecture 4: complexity).
Third, the stochastic lambda sampling creates a combinatorially larger set of
virtual examples at every epoch, preventing overfitting to any fixed dataset
(Lecture 4: affinity and diversity).

Our implementation samples one lambda per batch from Beta({config['alpha']},
{config['alpha']}), following the original MixUp paper. Pairs are formed via
random permutation, and both inputs and one-hot labels are blended using basic
tensor operations (scatter_ for one-hot encoding, element-wise arithmetic).

2. HOW LABEL SMOOTHING PREVENTS OVERSHOOTING

With hard one-hot targets, minimising cross-entropy requires pushing the correct-
class logit toward +infinity. This drives overconfident predictions and causes
gradient saturation — the loss landscape becomes flat and updates become
negligibly small (Lecture 4: label smoothing under randomness).

Our Label Smoothing converts hard targets to soft targets:
    y_smooth = (1 - epsilon) * y + epsilon / K
where epsilon = {eps} and K = 10. The correct class receives a target of
{target_correct:.3f} instead of 1.0. Crucially, the optimal logit is now FINITE:
z* = log((1-eps+eps/K) / (eps/K)) = {optimal_logit:.2f}. Consequently:

(a) Gradients remain informative throughout training — the optimizer never needs
    to push weights to extreme values, preventing numerical instability.
(b) Predicted distributions are better calibrated — allocating some probability
    to incorrect classes reflects genuine uncertainty, acting as output
    regularisation (Lecture 4: randomness as regularisation) that complements
    MixUp's input-space regularisation.

The log-softmax is computed from scratch with the max-subtraction trick for
numerical stability, and the final loss is the negative dot product of smoothed
targets with log-probabilities, averaged over the batch.

3. QUANTITATIVE ROBUSTNESS RESULTS

    Noise Std    Baseline       MixUp+LS
    ─────────────────────────────────────
    clean        {bl_clean:.1%}          {mx_clean:.1%}
{table}

The MixUp+LS model achieves {mx_clean:.1%} clean accuracy vs the baseline's
{bl_clean:.1%}, confirming that MixUp's vicinal risk minimisation and Label
Smoothing's soft targets improve generalisation. At low noise intensities
(sigma <= 0.10), MixUp+LS retains its accuracy advantage: the smoother decision
boundaries learned through vicinal training are less sensitive to small input
perturbations. However, at higher noise (sigma >= 0.20), the baseline degrades
more gracefully. This is an expected consequence of the bias-variance trade-off
(Lecture 4): Label Smoothing distributes probability mass across all classes,
producing well-calibrated but less peaky predictions. Under severe noise, these
flatter softmax distributions are more easily disrupted — a small shift in logit
space is more likely to change the argmax when classes have similar probabilities.
The baseline's overconfident (high-variance) predictions, while poorly calibrated,
produce sharper peaks that are more resistant to random perturbation. Additionally,
with alpha = {config['alpha']}, the Beta({config['alpha']}, {config['alpha']})
distribution is U-shaped, producing lambdas concentrated near 0 and 1, so the
effective MixUp regularisation is mild — the model still learns sharp features
that are vulnerable to large noise.

This illustrates a fundamental tension: the same smoothing that improves
generalisation and calibration can reduce robustness to extreme input corruption.
Increasing alpha would produce more aggressive blending and potentially improve
high-noise robustness, but at the cost of clean accuracy.

Together with early stopping (patience = {config['patience']}), which halts
training when validation loss increases (Lecture 4: early stopping), a StepLR
learning rate schedule that halves the rate every 20 epochs (Lecture 2:
adaptive learning rates) for stable convergence, and gradient clipping
(max_norm = {config.get('grad_clip', 5.0)}) to prevent gradient explosions,
these techniques shift the model toward a balanced position on the bias-variance
curve — improving generalisation on clean and mildly-corrupted data while
accepting a trade-off at extreme noise levels.

GenAI Usage: Claude assisted with code structure and drafting. All
from-scratch implementations and theoretical claims were verified by the author.
"""
    print(text)


def main():
    """Load saved models, evaluate robustness, generate visualisation,
    print analysis."""

    # Load training history 
    with open('training_history.json', 'r') as f:
        data = json.load(f)
    config = data['config']

    # Setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
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

    # Load models
    ckpt = torch.load('models.pth', map_location=device, weights_only=True)

    baseline = ConvNet(num_classes=num_classes, dropout_rate=0.3).to(device)
    baseline.load_state_dict(ckpt['baseline_state'])

    mixup_model = ConvNet(num_classes=num_classes, dropout_rate=0.3).to(device)
    mixup_model.load_state_dict(ckpt['mixup_state'])

    param_count = sum(p.numel() for p in baseline.parameters())
    print(f"Loaded both models ({param_count:,} params each)")

    # Clean test accuracy
    _, bl_clean = evaluate(baseline, test_loader, criterion, device)
    _, mx_clean = evaluate(mixup_model, test_loader, criterion, device)
    print(f"Baseline     clean accuracy: {bl_clean:.4f}")
    print(f"MixUp+LS     clean accuracy: {mx_clean:.4f}\n")

    # Noisy test evaluation
    print("=" * 70)
    print("ROBUSTNESS EVALUATION — Gaussian Noise")
    print("=" * 70)

    noise_levels = [0.05, 0.1, 0.2, 0.3, 0.5]
    bl_noisy = {}
    mx_noisy = {}

    for sigma in noise_levels:
        bl_acc = evaluate_noisy(baseline, test_loader, device, noise_std=sigma)
        mx_acc = evaluate_noisy(mixup_model, test_loader, device, noise_std=sigma)
        bl_noisy[sigma] = bl_acc
        mx_noisy[sigma] = mx_acc
        print(f"  sigma={sigma:.2f}  Baseline: {bl_acc:.4f}  MixUp+LS: {mx_acc:.4f}")

    # Generate robustness demo image
    print()
    create_robustness_demo(test_data, alpha=config['alpha'],
                           filename='robustness_demo.png')

    # Print technical analysis
    print()
    print_technical_analysis(bl_clean, mx_clean, bl_noisy, mx_noisy, config)


if __name__ == '__main__':
    main()
