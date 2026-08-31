"""Style Matching (SM) module -- L_MM and L_CM.

The style of a feature map is summarised by its channel-wise first and second moments
(``moments_matching_loss``, L_MM) and by the correlations between its channels
(``correlation_matching_loss``, L_CM).  Matching those two statistics between the real and
the condensed batch is the SM module of the paper; see docs/method.md.
"""
import torch
import torch.nn as nn


def calc_mean_std(feat, eps=1e-5):
    """
    Calculates the mean and standard deviation of the feature maps.

    Args:
    - feat (torch.Tensor): Input feature map tensor of shape (N, C, H, W).
    - eps (float): A small value added to the variance to avoid divide-by-zero.

    Returns:
    - feat_mean (torch.Tensor): The mean of the feature maps, reshaped to (N, C, 1, 1).
    - feat_std (torch.Tensor): The standard deviation of the feature maps, reshaped to (N, C, 1, 1).
    """
    size = feat.size()
    assert (len(size) == 4)  # Ensure the input tensor has 4 dimensions (batch, channels, height, width)
    N, C = size[:2]  # Extract batch size (N) and number of channels (C)

    # Calculate the variance for each feature map and add a small epsilon to avoid division by zero
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    # Calculate the standard deviation from the variance
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    # Calculate the mean for each feature map
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)

    return feat_mean, feat_std  # Return the mean and standard deviation, reshaped for broadcasting



def gram_matrix(x, should_normalize=True):
    """
    Computes the Gram matrix, capturing correlations between channels for each spatial location.

    Args:
    - x (torch.Tensor): Input tensor of shape (batch, channels, height, width).
    - should_normalize (bool): Whether to normalize the Gram matrix by the number of elements.

    Returns:
    - torch.Tensor: The Gram matrix of the input tensor.
    """
    (b, ch, h, w) = x.size()  # Unpack dimensions: batch size, channels, height, width
    features = x.view(b, ch, w * h)  # Reshape to (batch, channels, width * height)
    features_t = features.transpose(1, 2)  # Transpose to (batch, width * height, channels)
    gram = features.bmm(features_t)  # Batch matrix multiplication to compute Gram matrix

    if should_normalize:
        gram /= ch * h * w  # Normalize by the number of elements

    return gram  # Output size: (batch, channels, channels), capturing channel correlations


def style_vector(feat, eps=1e-5):
    """Per-sample channel-wise style descriptor of a feature map.

    Args:
        feat: (N, C, H, W)
    Returns:
        mu, sd: (N, C) each -- the spatial mean and std of every channel of every sample.
    """
    n, c = feat.shape[:2]
    x = feat.reshape(n, c, -1)
    mu = x.mean(dim=2)
    sd = (x.var(dim=2, unbiased=False) + eps).sqrt()
    return mu, sd


def _sq_diff(a, b, relative, eps=1e-8):
    """Mean squared difference, optionally divided by the magnitude of the target.

    The relative form makes the term invariant to the scale of the feature maps, which
    matters once the style is read before normalisation: there the magnitudes depend on
    the random initialisation and grow with depth, so an absolute loss silently weights
    the layers by their activation scale and its coefficient has to be retuned for every
    tap, architecture and dataset.
    """
    num = (a - b).pow(2).mean()
    if not relative:
        return num
    return num / (b.detach().pow(2).mean() + eps)


def moments_matching_loss(feat_syn, feat_real, mode='persample', relative=False, eps=1e-5):
    """First/second-moment style matching between a synthetic and a real feature map.

    mode='batchavg' reproduces the published implementation: the feature maps are
        averaged over the batch first and the spatial mean/std of that average are
        compared.  Because the spatial variance of a batch-average shrinks with the
        batch size, the target computed from ``batch_real`` real images is not on the
        same scale as the value computed from ``ipc`` synthetic images.
    mode='persample' compares E_x[mu(x)] and E_x[sd(x)] instead, which are sample means
        of a per-sample quantity and therefore unbiased with respect to the batch size.
    """
    if mode == 'batchavg':
        s = torch.mean(feat_syn, dim=0, keepdim=True)
        r = torch.mean(feat_real, dim=0, keepdim=True)
        s_mu, s_sd = calc_mean_std(s, eps)
        r_mu, r_sd = calc_mean_std(r, eps)
        return (_sq_diff(s_mu, r_mu, relative) + _sq_diff(s_sd, r_sd, relative)) / 2

    s_mu, s_sd = style_vector(feat_syn, eps)
    r_mu, r_sd = style_vector(feat_real, eps)
    return (_sq_diff(s_mu.mean(0), r_mu.mean(0), relative)
            + _sq_diff(s_sd.mean(0), r_sd.mean(0), relative)) / 2


def correlation_matching_loss(feat_syn, feat_real):
    """Gram-matrix (channel correlation) matching, L_CM of the paper."""
    g_s = gram_matrix(feat_syn).mean(dim=0)
    g_r = gram_matrix(feat_real).mean(dim=0)
    return nn.MSELoss(reduction='sum')(g_s, g_r)
