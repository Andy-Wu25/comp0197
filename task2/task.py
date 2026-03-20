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
Claude was used as an assistive tool for code and terminal output structuring 
and Pillow-based visualisation. All code implemented by Claude was reviewed, 
tested, and edited by the author to ensure correctness.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
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
    target_correct = 1.0 - eps + eps / 10.0
    target_other = eps / 10.0
    optimal_logit = torch.tensor(target_correct / target_other).log().item()

    text = f"""\
================================================================================
Technical Justification — MixUp and label smoothing for robustness
================================================================================

1. Why MixUp prevents memorisation

Standard training minimises the empirical risk over the training distribution
p_data(x,y) = (1/M) sum delta(x=x_m, y=y_m), which places all probability mass
on individual samples. A high-capacity network can drive this loss to zero by 
memorising each sample, producing sharp decision boundaries that do not generalise.

MixUp replaces this with vicinal risk minimisation, where the training distribution 
is replaced by a vicinity distribution that blends pairs:
    x_tilde = lambda * x_i + (1 - lambda) * x_j
    y_tilde = lambda * y_i + (1 - lambda) * y_j
with lambda ~ Beta({config['alpha']}, {config['alpha']}).

This prevents memorisation because a model that has memorised individual samples
cannot produce coherent predictions at the interpolated points between them.
The linear label blending forces predictions to vary smoothly between training
examples, encouraging low-complexity decision boundaries. Additionally, the stochastic lambda
sampling also creates a combinatorially larger set of virtual training examples each epoch,
making it much harder to overfit to any fixed set of inputs.

Our implementation samples one lambda per batch, pairs are formed by random
permutation, and both images and one-hot labels are blended using basic tensor
operations (scatter_ for one-hot encoding, element-wise arithmetic).

2. How Label Smoothing Prevents Overshooting

With hard one-hot targets, cross-entropy loss is minimised by pushing the
correct-class logit toward +infinity. This causes gradient saturation where updates
become negligibly small as the loss landscape flattens.

Label Smoothing replaces hard targets with soft targets:
    y_smooth = (1 - epsilon) * y + epsilon / K
where epsilon = {eps} and K = 10. The correct class target becomes
{target_correct:.3f} instead of 1.0. The key consequence is that the optimal
logit is now finite: z* = log((1-eps+eps/K) / (eps/K)) = {optimal_logit:.2f}.
The optimiser no longer needs to push weights to extreme values, allowing gradients to 
remain informative throughout training. The resulting predictions are also better
calibrated. Allocating some probability to incorrect classes shows realistic
uncertainty, acting as output regularisation that complements MixUp's input-space regularisation.

Our loss function computes log-softmax from scratch using the max-subtraction
trick for numerical stability. Then, we take the negative dot product of smoothed
targets with log-probabilities which is averaged over the batch.

3. Quantitative results

    Noise Std    Baseline       MixUp+LS
    ─────────────────────────────────────
    clean        {bl_clean:.1%}          {mx_clean:.1%}
{table}

MixUp+LS achieves {mx_clean:.1%} clean accuracy vs {bl_clean:.1%} for the
baseline and retains an advantage at mild noise (sigma <= 0.10). At higher
noise (sigma >= 0.20) the baseline degrades more gracefully. This is because
Label Smoothing produces flatter softmax distributions. At extreme noise levels,
a small shift in logit space is more likely to flip the argmax when class
probabilities are closer together. The baseline's overconfident but sharper peaks
are more resistant to random perturbation, even though they are poorly calibrated.
With alpha={config['alpha']}, Beta({config['alpha']},{config['alpha']}) is
U-shaped (most lambdas near 0 or 1), so the MixUp regularisation is mild.

Early stopping (patience={config['patience']}) halts training when validation
loss stops improving. A StepLR schedule halves the learning rate every 20 epochs 
for stable convergence.
"""
    print(text)


def main():
    """Load saved models, evaluate robustness, generate visualisation,
    print analysis."""

    # Training configuration (matches train.py)
    config = {
        'alpha': 0.2,
        'epsilon': 0.1,
        'patience': 10,
    }

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
