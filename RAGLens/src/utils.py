import torch
import numpy as np
from typing import Tuple, Optional

def compute_conditional_mutual_information_chunked(
    features: torch.Tensor,
    labels: torch.Tensor,
    cond: torch.Tensor,
    *,
    n_bins: int = 10,
    n_cond_bins: int = 8,
    chunk_size: int = 1000,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Approximate conditional mutual information I(X;Y|C) via binning C:
      I(X;Y|C) = sum_c p(c) I(X;Y | C=c)

    Implementation:
      - Discretize `cond` into `n_cond_bins` quantile bins.
      - For each bin, compute MI(features_bin, labels_bin) in chunks (vectorized).
      - Weight by bin frequency.

    Args:
      features: (n_samples, n_features) continuous SAE activations (after pooling)
      labels:   (n_samples,) binary labels (0/1)
      cond:     (n_samples,) continuous proxy H
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    features = features.detach()
    labels = labels.detach()
    cond = cond.detach()

    if features.ndim != 2:
        raise ValueError(f"features must be 2D (n_samples, n_features), got {features.shape}")
    if labels.ndim != 1 or cond.ndim != 1:
        raise ValueError(f"labels/cond must be 1D, got labels={labels.shape}, cond={cond.shape}")
    if features.shape[0] != labels.shape[0] or labels.shape[0] != cond.shape[0]:
        raise ValueError("features, labels, cond must have same number of samples")

    n_samples, n_features = features.shape
    mi_scores = torch.zeros(n_features, dtype=torch.float32)

    # Quantile binning for cond (robust to scale)
    cond_np = cond.cpu().numpy().astype(np.float32)
    quantiles = np.quantile(cond_np, q=np.linspace(0, 1, n_cond_bins + 1))
    # Avoid duplicate edges due to constant cond
    quantiles = np.unique(quantiles)
    if len(quantiles) <= 2:
        # cond is (almost) constant -> fallback to unconditional MI
        return compute_mutual_information_chunked(features, labels, n_bins=n_bins, chunk_size=chunk_size, device=device)

    # Use np.digitize to assign bins [0, n_bins-1]
    bin_ids = np.digitize(cond_np, quantiles[1:-1], right=False)

    for b in range(bin_ids.max() + 1):
        idx = np.where(bin_ids == b)[0]
        if len(idx) < 10:
            continue
        weight = float(len(idx)) / float(n_samples)
        bin_features = features[idx]
        bin_labels = labels[idx]
        # If label is constant in bin, MI is 0
        if len(torch.unique(bin_labels)) < 2:
            continue
        bin_mi = compute_mutual_information_chunked(
            bin_features, bin_labels, n_bins=n_bins, chunk_size=chunk_size, device=device
        ).to(torch.float32)
        mi_scores += weight * bin_mi

    return mi_scores

def compute_mutual_information_vectorized(
    features: torch.Tensor, 
    labels: torch.Tensor, 
    n_bins: int = 10,
    device: Optional[str] = None
) -> torch.Tensor:
    """
    Compute mutual information between ALL features and target labels simultaneously using PyTorch.
    This is a fully vectorized implementation that processes all features in parallel.
    
    Args:
        features: Feature tensor of shape (n_samples, n_features)
        labels: Label tensor of shape (n_samples,)
        n_bins: Number of bins for discretizing continuous features
        device: Device to run computations on ('cuda' or 'cpu')
    
    Returns:
        Mutual information scores for each feature of shape (n_features,)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    features = features.to(device)
    labels = labels.to(device)
    
    n_samples, n_features = features.shape
    n_classes = len(torch.unique(labels))
    
    # Vectorized discretization of ALL features at once
    feature_mins = features.min(dim=0, keepdim=True)[0]  # Shape: (1, n_features)
    feature_maxs = features.max(dim=0, keepdim=True)[0]  # Shape: (1, n_features)
    
    # Handle features where all values are the same
    feature_ranges = feature_maxs - feature_mins
    zero_range_mask = feature_ranges == 0
    feature_ranges = torch.where(zero_range_mask, torch.ones_like(feature_ranges), feature_ranges)
    
    # Normalize features to [0, 1] range
    normalized_features = (features - feature_mins) / feature_ranges
    
    # Discretize all features simultaneously
    # Clamp to ensure values are in [0, n_bins-1] range
    feature_bins = torch.clamp(
        (normalized_features * n_bins).long(), 
        0, n_bins - 1
    )  # Shape: (n_samples, n_features)
    
    # For features with zero range, set all bins to 0
    feature_bins = torch.where(
        zero_range_mask.expand_as(feature_bins), 
        torch.zeros_like(feature_bins), 
        feature_bins
    )
    
    # Compute joint histograms for ALL features simultaneously
    mi_scores = _compute_vectorized_mi(feature_bins, labels, n_bins, n_classes, n_samples)
    
    return mi_scores.cpu()

def _compute_vectorized_mi(
    feature_bins: torch.Tensor, 
    labels: torch.Tensor, 
    n_bins: int, 
    n_classes: int, 
    n_samples: int
) -> torch.Tensor:
    """
    Compute mutual information for ALL features simultaneously using advanced tensor operations.
    """
    n_features = feature_bins.shape[1]
    device = feature_bins.device
    
    # Create joint histograms for all features at once
    # Shape: (n_features, n_bins, n_classes)
    joint_hist = torch.zeros(n_features, n_bins, n_classes, device=device)
    
    # Vectorized histogram computation using scatter_add
    # We need to create indices for each feature separately
    feature_indices = torch.arange(n_features, device=device).unsqueeze(0).expand(n_samples, -1)  # (n_samples, n_features)
    
    # Create flat indices for scatter operation
    # Each element: feature_idx * (n_bins * n_classes) + bin_idx * n_classes + class_idx
    flat_indices = (
        feature_indices * (n_bins * n_classes) + 
        feature_bins * n_classes + 
        labels.unsqueeze(1).expand(-1, n_features)
    )  # Shape: (n_samples, n_features)
    
    # Flatten everything for scatter_add
    flat_joint = torch.zeros(n_features * n_bins * n_classes, device=device)
    flat_joint.scatter_add_(0, flat_indices.flatten(), torch.ones(n_samples * n_features, device=device))
    
    # Reshape back to (n_features, n_bins, n_classes)
    joint_hist = flat_joint.view(n_features, n_bins, n_classes)
    
    # Convert counts to probabilities
    joint_prob = joint_hist / n_samples  # Shape: (n_features, n_bins, n_classes)
    
    # Compute marginal probabilities
    feature_marginals = joint_prob.sum(dim=2, keepdim=True)  # P(X) for each feature: (n_features, n_bins, 1)
    label_marginals = joint_prob.sum(dim=1, keepdim=True)    # P(Y) for each feature: (n_features, 1, n_classes)
    
    # Compute mutual information vectorized across all features
    eps = 1e-10
    
    # Create masks for valid probabilities
    joint_mask = joint_prob > eps
    marginal_product = feature_marginals * label_marginals
    marginal_mask = marginal_product > eps
    valid_mask = joint_mask & marginal_mask
    
    # Initialize MI tensor
    mi_terms = torch.zeros_like(joint_prob)
    
    # Compute MI only for valid entries
    if valid_mask.any():
        log_ratio = torch.log(joint_prob[valid_mask] / marginal_product[valid_mask])
        mi_terms[valid_mask] = joint_prob[valid_mask] * log_ratio
    
    # Sum over bins and classes to get MI for each feature
    mi_scores = mi_terms.sum(dim=(1, 2))  # Shape: (n_features,)
    
    return mi_scores

def compute_mutual_information_chunked(
    features: torch.Tensor, 
    labels: torch.Tensor, 
    n_bins: int = 10,
    chunk_size: int = 1000,
    device: Optional[str] = None
) -> torch.Tensor:
    """
    Compute mutual information in chunks for memory efficiency while maintaining vectorization.
    
    Args:
        features: Feature tensor of shape (n_samples, n_features)
        labels: Label tensor of shape (n_samples,)
        n_bins: Number of bins for discretizing continuous features
        chunk_size: Number of features to process simultaneously
        device: Device to run computations on
    
    Returns:
        Mutual information scores for each feature
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    n_features = features.shape[1]
    mi_scores = torch.zeros(n_features)
    
    # Process features in chunks, but vectorized within each chunk
    for start_idx in range(0, n_features, chunk_size):
        end_idx = min(start_idx + chunk_size, n_features)
        chunk_features = features[:, start_idx:end_idx]
        
        chunk_mi = compute_mutual_information_vectorized(
            chunk_features, labels, n_bins, device
        )
        
        mi_scores[start_idx:end_idx] = chunk_mi
        
        # Clear cache to manage memory
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    return mi_scores